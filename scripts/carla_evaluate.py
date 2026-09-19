from __future__ import annotations

"""CARLA fixed-route evaluation runner with driving-event telemetry."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    spawn_index: int
    destination_index: int


@dataclass
class EpisodeRecord:
    episode_id: str
    route_id: str
    seed: int
    frames: int = 0
    completed: bool = False
    collision_count: int = 0
    lane_departures: int = 0
    red_light_violations: int = 0
    interventions: int = 0
    offroad_frames: int = 0
    distance_m: float = 0.0
    route_distance_m: float = 0.0
    inference_success: bool = False
    inference_errors: int = 0
    latency_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is None:
            self.latency_ms = []

    @property
    def route_completion(self) -> float:
        if self.route_distance_m <= 0:
            return 0.0
        return min(1.0, max(0.0, self.distance_m / self.route_distance_m))

    @property
    def trajectory_collision_rate(self) -> float:
        return float(self.collision_count > 0)

    @property
    def offroad_rate(self) -> float:
        return (
            0.0
            if self.frames <= 0
            else float(self.offroad_frames / self.frames)
        )

    def as_evaluation_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "route_id": self.route_id,
            "seed": self.seed,
            "route_completion_values": [self.route_completion],
            "trajectory_collision_flags": [self.collision_count > 0],
            "offroad_flags": [self.offroad_frames > 0],
            "collision_flags": [self.collision_count > 0],
            "red_light_flags": [self.red_light_violations > 0],
            "lane_departure_flags": [self.lane_departures > 0],
            "intervention_flags": [self.interventions > 0],
            "runtime_samples": [
                {
                    "preprocessing_ms": 0.0,
                    "inference_ms": value,
                    "rendering_ms": 0.0,
                }
                for value in (self.latency_ms or [])
            ],
            "episode_summary": {
                "collision_count": self.collision_count,
                "red_light_violations": self.red_light_violations,
                "lane_departures": self.lane_departures,
                "intervention_recovery_count": self.interventions,
                "route_completion": self.route_completion,
                "offroad_rate": self.offroad_rate,
                "trajectory_collision_rate": self.trajectory_collision_rate,
            },
        }


class CarlaEvaluationRunner:
    def __init__(self, host: str, port: int, timeout: float, seed: int) -> None:
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError(
                "CARLA Python API is required. Install it in a CARLA-compatible environment."
            ) from exc

        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def connect(self) -> Any:
        return self.client.get_world()

    @staticmethod
    def sensor_frame(image: Any) -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        return array.reshape(image.height, image.width, 4)[:, :, :3][:, :, ::-1].copy()

    @staticmethod
    def lidar_frame(measurement: Any) -> np.ndarray:
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        return raw.reshape(-1, 4)[:, :3].copy()

    def _spawn_sensor(
        self,
        world: Any,
        blueprint_id: str,
        transform: Any,
        vehicle: Any,
    ) -> Any:
        blueprint = world.get_blueprint_library().find(blueprint_id)
        return world.spawn_actor(blueprint, transform, attach_to=vehicle)

    def run_route(self, world: Any, route: RouteSpec, output_dir: Path) -> EpisodeRecord:
        import queue

        output_dir.mkdir(parents=True, exist_ok=True)
        record = EpisodeRecord(
            episode_id=f"episode_{route.route_id}",
            route_id=route.route_id,
            seed=self.seed,
        )

        previous_settings = world.get_settings()
        sync_settings = world.get_settings()
        sync_settings.synchronous_mode = True
        sync_settings.fixed_delta_seconds = 0.05
        world.apply_settings(sync_settings)

        actors: list[Any] = []
        queues: dict[str, queue.Queue] = {
            "rgb": queue.Queue(maxsize=4),
            "lidar": queue.Queue(maxsize=4),
        }

        try:
            blueprints = world.get_blueprint_library()
            vehicle_candidates = blueprints.filter("vehicle.tesla.model3")
            if not vehicle_candidates:
                raise RuntimeError("No Tesla Model 3 blueprint available.")
            vehicle_bp = vehicle_candidates[0]

            spawn_points = world.get_map().get_spawn_points()
            if max(route.spawn_index, route.destination_index) >= len(spawn_points):
                raise ValueError("Route index exceeds available CARLA spawn points.")

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[route.spawn_index])
            if vehicle is None:
                raise RuntimeError(f"Unable to spawn vehicle for route {route.route_id}.")
            actors.append(vehicle)

            camera = self._spawn_sensor(
                world,
                "sensor.camera.rgb",
                self.carla.Transform(self.carla.Location(x=1.3, z=2.3)),
                vehicle,
            )
            lidar = self._spawn_sensor(
                world,
                "sensor.lidar.ray_cast",
                self.carla.Transform(self.carla.Location(x=1.3, z=2.5)),
                vehicle,
            )
            collision_sensor = self._spawn_sensor(
                world,
                "sensor.other.collision",
                self.carla.Transform(),
                vehicle,
            )
            lane_sensor = self._spawn_sensor(
                world,
                "sensor.other.lane_invasion",
                self.carla.Transform(),
                vehicle,
            )
            actors.extend([camera, lidar, collision_sensor, lane_sensor])

            camera.listen(queues["rgb"].put)
            lidar.listen(queues["lidar"].put)

            event_state = {"collisions": 0, "lane_departures": 0}
            collision_sensor.listen(
                lambda _: event_state.__setitem__(
                    "collisions", event_state["collisions"] + 1
                )
            )
            lane_sensor.listen(
                lambda _: event_state.__setitem__(
                    "lane_departures", event_state["lane_departures"] + 1
                )
            )

            start = vehicle.get_transform().location
            destination = spawn_points[route.destination_index].location
            record.route_distance_m = float(start.distance(destination))

            for _ in range(600):
                world.tick()
                rgb_measurement = queues["rgb"].get(timeout=2.0)
                lidar_measurement = queues["lidar"].get(timeout=2.0)
                record.frames += 1

                frame_start = time.perf_counter()
                _ = self.sensor_frame(rgb_measurement)
                _ = self.lidar_frame(lidar_measurement)
                preprocess_ms = (time.perf_counter() - frame_start) * 1000.0

                transform = vehicle.get_transform()
                location = transform.location
                distance_to_goal = location.distance(destination)

                if distance_to_goal < 6.0:
                    record.completed = True
                    record.distance_m = record.route_distance_m
                    break

                road_wp = world.get_map().get_waypoint(location)
                next_wps = road_wp.next(8.0)
                target = next_wps[0] if next_wps else road_wp
                target_loc = target.transform.location

                dx = target_loc.x - location.x
                dy = target_loc.y - location.y
                yaw = math.radians(transform.rotation.yaw)
                local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
                local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                steer = max(
                    -1.0,
                    min(1.0, math.atan2(local_y, max(local_x, 0.1)) * 1.8),
                )

                throttle = 0.35 if distance_to_goal >= 15.0 else 0.15
                vehicle.apply_control(
                    self.carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=0.0,
                    )
                )

                location_after = vehicle.get_transform().location
                record.distance_m = max(
                    record.distance_m,
                    record.route_distance_m - location_after.distance(destination),
                )
                record.collision_count = event_state["collisions"]
                record.lane_departures = event_state["lane_departures"]

                # Traffic-light and off-road measurements require explicit
                # CARLA map/traffic semantics; preserve conservative counters here.
                is_offroad = road_wp.lane_type != self.carla.LaneType.Driving
                if is_offroad:
                    record.offroad_frames += 1

                # No model adapter is invoked in this baseline runner.
                inference_ms = 0.0
                rendering_ms = 0.0
                record.latency_ms.append(preprocess_ms + inference_ms + rendering_ms)

            record.collision_count = event_state["collisions"]
            record.lane_departures = event_state["lane_departures"]

            summary_path = output_dir / f"{route.route_id}.json"
            summary_path.write_text(
                json.dumps(asdict(record), indent=2),
                encoding="utf-8",
            )
            return record
        finally:
            for actor in reversed(actors):
                if actor.is_alive:
                    actor.destroy()
            world.apply_settings(previous_settings)


def parse_routes(value: str) -> list[RouteSpec]:
    routes: list[RouteSpec] = []
    for item in value.split(","):
        route_id, spawn, destination = item.split(":")
        routes.append(RouteSpec(route_id, int(spawn), int(destination)))
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-route CARLA evaluation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--routes",
        default="route_00:0:1,route_01:2:3",
        help="Comma-separated route_id:spawn_index:destination_index definitions.",
    )
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    args = parser.parse_args()

    runner = CarlaEvaluationRunner(args.host, args.port, args.timeout, args.seed)
    world = runner.connect()
    output_dir = Path(args.output_dir)
    records = [
        runner.run_route(world, route, output_dir)
        for route in parse_routes(args.routes)
    ]

    rows = [record.as_evaluation_row() for record in records]
    jsonl_path = output_dir / "evaluation_records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "episodes.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()

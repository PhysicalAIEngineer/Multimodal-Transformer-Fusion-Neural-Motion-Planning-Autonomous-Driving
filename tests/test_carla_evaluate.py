from __future__ import annotations

from scripts.carla_evaluate import RouteSpec, parse_routes


def test_parse_routes() -> None:
    routes = parse_routes("a:0:1,b:2:3")
    assert routes == [RouteSpec("a", 0, 1), RouteSpec("b", 2, 3)]


def test_route_spec_is_stable() -> None:
    route = RouteSpec("route_00", 0, 1)
    assert route.route_id == "route_00"
    assert route.spawn_index == 0
    assert route.destination_index == 1

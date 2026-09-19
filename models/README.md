# Model artifacts

The production server expects the trained checkpoint at models/transfuser_regnet032_seed1_39.pth.

The repository copy currently present is only 134 bytes, so the production server treats it as an incomplete artifact rather than silently serving untrained weights. Provide the real checkpoint through a secure artifact store, Git LFS, or a mounted volume before enabling /ready and /predict.

Record these metadata with every production artifact:

- model architecture and git commit SHA
- PyTorch and timm versions
- dataset version
- input tensor shapes and normalization
- evaluation metrics
- SHA-256 checksum
- training configuration

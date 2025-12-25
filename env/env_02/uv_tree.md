run
```bash
uv tree
```

```bash
Resolved 77 packages in 1ms
gdkvm_20251215 v0.1.0
├── hydra-core v1.3.2
│   ├── antlr4-python3-runtime v4.9.3
│   ├── omegaconf v2.3.0
│   │   ├── antlr4-python3-runtime v4.9.3
│   │   └── pyyaml v6.0.3
│   └── packaging v25.0
├── matplotlib v3.10.8
│   ├── contourpy v1.3.3
│   │   └── numpy v2.2.6
│   ├── cycler v0.12.1
│   ├── fonttools v4.61.1
│   ├── kiwisolver v1.4.9
│   ├── numpy v2.2.6
│   ├── packaging v25.0
│   ├── pillow v12.0.0
│   ├── pyparsing v3.2.5
│   └── python-dateutil v2.9.0.post0
│       └── six v1.17.0
├── monai v1.5.1
│   ├── numpy v2.2.6
│   └── torch v2.9.0+cu128
│       ├── filelock v3.20.1
│       ├── fsspec v2025.12.0
│       ├── jinja2 v3.1.6
│       │   └── markupsafe v3.0.3
│       ├── networkx v3.6.1
│       ├── nvidia-cublas-cu12 v12.8.4.1
│       ├── nvidia-cuda-cupti-cu12 v12.8.90
│       ├── nvidia-cuda-nvrtc-cu12 v12.8.93
│       ├── nvidia-cuda-runtime-cu12 v12.8.90
│       ├── nvidia-cudnn-cu12 v9.10.2.21
│       │   └── nvidia-cublas-cu12 v12.8.4.1
│       ├── nvidia-cufft-cu12 v11.3.3.83
│       │   └── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-cufile-cu12 v1.13.1.3
│       ├── nvidia-curand-cu12 v10.3.9.90
│       ├── nvidia-cusolver-cu12 v11.7.3.90
│       │   ├── nvidia-cublas-cu12 v12.8.4.1
│       │   ├── nvidia-cusparse-cu12 v12.5.8.93
│       │   │   └── nvidia-nvjitlink-cu12 v12.8.93
│       │   └── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-cusparse-cu12 v12.5.8.93 (*)
│       ├── nvidia-cusparselt-cu12 v0.7.1
│       ├── nvidia-nccl-cu12 v2.27.5
│       ├── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-nvshmem-cu12 v3.3.20
│       ├── nvidia-nvtx-cu12 v12.8.90
│       ├── setuptools v80.9.0
│       ├── sympy v1.14.0
│       │   └── mpmath v1.3.0
│       ├── triton v3.5.0
│       └── typing-extensions v4.15.0
├── numba v0.63.1
│   ├── llvmlite v0.46.0
│   └── numpy v2.2.6
├── numpy v2.2.6
├── opencv-python v4.12.0.88
│   └── numpy v2.2.6
├── pydantic v2.11.10
│   ├── annotated-types v0.7.0
│   ├── pydantic-core v2.33.2
│   │   └── typing-extensions v4.15.0
│   ├── typing-extensions v4.15.0
│   └── typing-inspection v0.4.2
│       └── typing-extensions v4.15.0
├── scipy v1.16.3
│   └── numpy v2.2.6
├── tensorboard v2.20.0
│   ├── absl-py v2.3.1
│   ├── grpcio v1.76.0
│   │   └── typing-extensions v4.15.0
│   ├── markdown v3.10
│   ├── numpy v2.2.6
│   ├── packaging v25.0
│   ├── pillow v12.0.0
│   ├── protobuf v6.33.2
│   ├── setuptools v80.9.0
│   ├── tensorboard-data-server v0.7.2
│   └── werkzeug v3.1.4
│       └── markupsafe v3.0.3
├── torch v2.9.0+cu128 (*)
├── torchvision v0.24.0+cu128
│   ├── numpy v2.2.6
│   ├── pillow v12.0.0
│   └── torch v2.9.0+cu128 (*)
├── tqdm v4.67.1
├── wandb v0.23.1
│   ├── click v8.3.1
│   ├── gitpython v3.1.45
│   │   └── gitdb v4.0.12
│   │       └── smmap v5.0.2
│   ├── packaging v25.0
│   ├── platformdirs v4.5.1
│   ├── protobuf v6.33.2
│   ├── pydantic v2.11.10 (*)
│   ├── pyyaml v6.0.3
│   ├── requests v2.32.5
│   │   ├── certifi v2025.11.12
│   │   ├── charset-normalizer v3.4.4
│   │   ├── idna v3.11
│   │   └── urllib3 v2.6.2
│   ├── sentry-sdk v2.47.0
│   │   ├── certifi v2025.11.12
│   │   └── urllib3 v2.6.2
│   └── typing-extensions v4.15.0
└── ruff v0.14.10 (group: dev)
(*) Package tree already displayed
```
# GDKVM 复现指南


## 1. 准备服务器项目目录
1. 进入实验根目录，以日期命名创建项目骨架：
   ```bash
   cd /data/username/Repo
   uv init -p 3.12.4 gdkvm_20251018
   ```
2. 将仓库克隆到同一目录，获取实际代码：
   ```bash
   git clone https://github.com/wangrui2025/GDKVM.git /data/username/Repo/gdkvm_20251018
   ```

## 2. 使用 uv 管理依赖
### 2.1 基于 uv 的项目
1. 依靠 GDKVM 的 `pyproject.toml` 与 `uv.lock` 配置环境。
2. 服务器端激活虚拟环境并安装依赖：
```bash
   cd /data/username/Repo/gdkvm_20251018
   source .venv/bin/activate
   uv sync
   uv pip check
```

## 数据集
- CAMUS PNG 256×256 10 帧子集：[Hugging Face 链接](https://huggingface.co/datasets/miyuki17/camus_png256x256_10f_20250709)
数据集：
camus：https://www.creatis.insa-lyon.fr/Challenge/camus/index.html
EchoNet-Dynamic数据集：https://echonet.github.io/dynamic/
处理好的camus，10 帧，png：https://huggingface.co/datasets/miyuki17/camus_png256x256_10f_20250709
- `data_path`：需改成你的本地或挂载路径，例如

  ```yaml
  data_path: "/data/username/dataset/camus_png256x256_15f/"
  ```


## 3. 运行训练脚本
```bash
chmod +x ./train.sh
./train.sh
```


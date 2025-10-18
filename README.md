# GDKVM 复现指南
2025 GDKVM：Gated Delta Rule 引导的时空键值记忆，用于超声心动图视频分割

## 快速概览
- 本地编写代码，配合 Copilot 等 AI 辅助修改。
- Git 管理版本，服务器侧由 `uv` 维护 Python 环境并独立运行实验。

## 准备工作
- 服务器终端工具：Termius 或 VS Code Remote Terminal。
- 本地工具：VS Code（可选）、Git。
- 服务器已安装 `uv` 与 Python 3.12.4（或可自行安装该版本）。

## 1. 服务器初始化 uv 项目
```bash
cd /data/username/Repo
uv init -p 3.12.4 gdkvm_20251018
```
- `username` 与日期按需替换。
- 该命令会创建基础项目结构与 `.venv` 环境（若未禁用）。
- 把项目 git clone 到相应目录下：
  ```bash
  git clone https://github.com/wangrui2025/GDKVM.git /data/username/Repo/gdkvm_20251018
  ```
  若目录已存在 `uv init` 生成的文件，可先备份或清理后再克隆；也可以在克隆后的仓库中执行 `uv init -p 3.12.4 .` 更新 `pyproject.toml` 与 `.venv`。

## 2. 使用 uv 管理环境
### 2.1 基于 uv 项目
1. 将服务器生成的 `pyproject.toml` 与 `uv.lock` 拷贝到本地项目，并提交至 Git。
2. 同步最新代码到服务器，然后执行：
   ```bash
   cd /data/username/Repo/gdkvm_20251018
   source .venv/bin/activate
   uv sync
   uv pip check
   ```
3. 如需镜像源，可改用：
   ```bash
   uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 2.2 基于 requirements.txt
若项目以 `requirements.txt` 记录依赖，可执行：
```bash
uv add $(cat requirements.txt)
```
或将依赖手动写入 `pyproject.toml` 后再运行 `uv sync`。

## 3. 获取实验代码
```bash
git clone https://github.com/wangrui2025/GDKVM.git
cd GDKVM
```
如需与远程实验目录整合，可将代码合并进 `gdkvm_20251018` 再同步到服务器。

## 4. 在服务器上运行实验
```bash
cd /data/username/Repo/gdkvm_20251018
source .venv/bin/activate
chmod +x ./train.sh
./train.sh
```
- 确保训练脚本需要的配置、数据路径已正确设置。
- 若脚本有额外依赖，请提前在 `pyproject.toml` 中声明并 `uv sync`。

## 数据集
- CAMUS PNG 256×256 10 帧子集：[Hugging Face 链接](https://huggingface.co/datasets/miyuki17/camus_png256x256_10f_20250709)

## 提示
- 建议在关键步骤后执行 `git status` 确认同步状态。
- 注意在 `.gitignore` 中忽略大文件或敏感数据，避免误传至远端或仓库。

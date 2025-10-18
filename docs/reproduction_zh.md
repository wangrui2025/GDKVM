# GDKVM 项目复现指南

## 1. 环境准备

### 1.1 Python 与依赖管理

* **Python**：建议使用 3.12（`pyproject.toml` 指定 `>=3.12`）。
* **uv**


### 1.2 创建项目目录

1. 进入实验根目录，创建项目骨架：

   ```bash
   cd /data/username/Repo
   uv init -p 3.12.4 gdkvm_20251018
   ```
2. 克隆 GDKVM 仓库（可替换为你的 fork 地址）：

   ```bash
   git clone https://github.com/wangrui2025/GDKVM.git /data/username/Repo/gdkvm_20251018
   ```

   或者：

   ```bash
   git clone https://github.com/your-org-or-user/gdkvm_20251018.git .
   ```

---

## 2. 使用 uv 管理依赖

1. 进入项目目录并激活虚拟环境：

   ```bash
   cd /data/username/Repo/gdkvm_20251018
   source .venv/bin/activate
   ```
2. 同步依赖并验证：

   ```bash
   uv sync
   uv pip check
   ```

---

## 3. 数据集准备

推荐数据集来源如下：

| 名称                    | 来源                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| CAMUS 原始数据            | [https://www.creatis.insa-lyon.fr/Challenge/camus/index.html](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) |
| EchoNet-Dynamic       | [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)                                                   |
| 处理好的 CAMUS 10帧 PNG 子集 | [Hugging Face 链接](https://huggingface.co/datasets/miyuki17/camus_png256x256_10f_20250709)                                  |

> 💡 建议将数据解压到 `/data/username/dataset/camus_png256x256_10f/` 。

```yaml
data_path: "/data/username/dataset/camus_png256x256_10f/"
```

---

## 4. 配置关键项

所有默认配置位于：

```
config/gdkvm_0709_2010.yaml
```

主要字段说明：

| 配置项                                                         | 说明                     |
| ----------------------------------------------------------- | ---------------------- |
| `data_path`                                                 | 数据集路径（可通过命令行 Hydra 覆盖） |
| `main_training.batch_size` / `num_workers`                  | 总量。脚本会按 GPU 数自动整除      |
| `main_training.seq_length` / `num_ref_frames` / `crop_size` | 与数据处理一致                |
| `exp_id` / `save_*`                                         | 实验命名、日志与权重保存控制         |

---

## 5. 启动训练

### 5.1 使用提供的脚本

默认脚本：`train.sh`

1. 编辑脚本：

   * 指定 GPU：

     ```bash
     export CUDA_VISIBLE_DEVICES=0,1,2,3
     ```
   * 设置 PYTHONPATH：

     ```bash
     export PYTHONPATH=/data/username/Repo/gdkvm_20251018${PYTHONPATH:+:$PYTHONPATH}
     ```
2. 执行：
   ```bash
   chmod +x ./train.sh
   ./train.sh
   ```

---

## 6. 日志与结果

* **Hydra 输出目录**：
  `outputs/gdkvm/<日期>/<时间>/`
* **Weights & Biases 日志**：
  保存在 `wandb/` 目录。可在本地执行：

  ```bash
  wandb sync wandb/
  ```

  上传结果至 Weights & Biases 云端查看训练曲线。

---

## 7. 离线场景说明

本指南专为 **无外网服务器** 设计，建议：

* 手动下载依赖包与数据集；
* 通过本地同步工具（如 `rsync`、`scp`）传输；
* WandB 结果可本地同步后再上传。

---

## 8. 帮助文档
1. 如何使用 uv [notion帮助文档](https://how2research.notion.site/uv-28fa44e48b9b80e396b3df9e2c8d9cdf)
2. 本项目中出现的 wandb 配置 todo
3. GDKVM wiki [github](https://github.com/wangrui2025/GDKVM_wiki)

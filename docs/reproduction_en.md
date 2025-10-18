# GDKVM Reproduction Guide

## 1. Environment Setup

### 1.1 Python and Dependency Manager

- **Python**: Python 3.12 is recommended (`pyproject.toml` requires `>=3.12`).
- **uv**: use `uv` to manage the virtual environment and dependencies.

### 1.2 Create the Project Directory

1. Enter your experiment root directory and scaffold the project (rename the directory as needed):

   ```bash
   cd /data/username/Repo
   uv init -p 3.12.4 gdkvm_20251018
   ```
2. Clone the GDKVM repository (replace with your fork if necessary):

   ```bash
   git clone https://github.com/wangrui2025/GDKVM.git /data/username/Repo/gdkvm_20251018
   ```

   or:

   ```bash
   git clone https://github.com/your-org-or-user/gdkvm_20251018.git .
   ```

---

## 2. Manage Dependencies with uv

1. Enter the project directory and activate the virtual environment:

   ```bash
   cd /data/username/Repo/gdkvm_20251018
   source .venv/bin/activate
   ```
2. Synchronize dependencies and validate the environment:

   ```bash
   uv sync
   uv pip check
   ```

---

## 3. Prepare the Datasets

Recommended datasets:

| Name                           | Source                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| CAMUS (original)               | [https://www.creatis.insa-lyon.fr/Challenge/camus/index.html](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) |
| EchoNet-Dynamic                | [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)                                                  |
| Processed CAMUS 10-frame PNG subset | [Hugging Face link](https://huggingface.co/datasets/miyuki17/camus_png256x256_10f_20250709)                           |

> 💡 Suggested extraction path: `/data/username/dataset/camus_png256x256_10f/`.

```yaml
data_path: "/data/username/dataset/camus_png256x256_10f/"
```

---

## 4. Key Configuration

Default configs live in:

```
config/gdkvm_0709_2010.yaml
```

Highlighted fields:

| Field                                                            | Description                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `data_path`                                                      | Dataset root path (override via Hydra CLI arguments if needed) |
| `main_training.batch_size` / `num_workers`                       | Global values; scripts will divide them across available GPUs  |
| `main_training.seq_length` / `num_ref_frames` / `crop_size`      | Keep consistent with your preprocessing pipeline               |
| `exp_id` / `save_*`                                              | Controls experiment naming, logging, and checkpoint storage    |

---

## 5. Launch Training

### 5.1 Provided Shell Script

Default script: `train.sh`

1. Edit the script:

   - Select GPUs:

     ```bash
     export CUDA_VISIBLE_DEVICES=0,1,2,3
     ```
   - Extend `PYTHONPATH`:

     ```bash
     export PYTHONPATH=/data/username/Repo/gdkvm_20251018${PYTHONPATH:+:$PYTHONPATH}
     ```
2. Run:

   ```bash
   chmod +x ./train.sh
   ./train.sh
   ```

---

## 6. Logs and Artifacts

- **Hydra outputs**: `outputs/gdkvm/<date>/<time>/`
- **Weights & Biases logs**: stored in the `wandb/` directory. Sync them locally and upload from your workstation (after copying `wandb/` to a machine with Internet access):

  ```bash
  wandb sync wandb/
  ```

---

## 7. Offline Server Notes

This guide targets servers **without direct Internet access**. Suggested workflow:

- Download Python wheels and datasets manually.
- Transfer files via tools such as `rsync`, `scp`, or SFTP.
- Copy the `wandb/` directory to a local machine and run `wandb sync` there to publish metrics.

---

## 8. Reference Material

1. uv quickstart (Notion) — [https://how2research.notion.site/uv-28fa44e48b9b80e396b3df9e2c8d9cdf](https://how2research.notion.site/uv-28fa44e48b9b80e396b3df9e2c8d9cdf)
2. wandb configuration checklist — (todo)
3. GDKVM wiki on GitHub — [https://github.com/wangrui2025/GDKVM_wiki](https://github.com/wangrui2025/GDKVM_wiki)

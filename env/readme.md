# ⚠️ Pydantic 2.12 与 W&B (Weights & Biases) 兼容性问题

**日期：2025-10-18**

## 🧩 问题概述

在今天（2025 年 10 月 18 日）的环境中，`pydantic` 最新版本 **2.12.x** 会与 **Weights & Biases (wandb)** 当前稳定版本产生兼容性问题。  
具体表现为在导入 `wandb` 时触发以下异常或告警：

```bash
pydantic.warnings.UnsupportedFieldAttributeWarning:
The 'repr' attribute with value False was provided to the `Field()` function,
which has no effect in the context it was used.
```

或者在你的训练脚本中出现类似：

```bash
pydantic.warnings.UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True...
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
```

这些问题源于 **Pydantic v2.12** 在内部 schema 生成逻辑中引入了新的校验机制，而 `wandb` 仍在部分类型声明中使用旧的 `Field(repr=False, frozen=True)` 写法。

---

## 📚 官方参考

W&B 官方在 GitHub 上已追踪该问题：

- **Issue**: [UnsupportedFieldAttributeWarning with Pydantic 2.12+](https://github.com/wandb/wandb/issues/10662)  
- **PR 相关修复（暂未完全解决该问题）**:  
  [fix(sdk): future-proof type hints on BaseModel.model_dump / model_dump_json](https://github.com/wandb/wandb/pull/10651)

---

## 💡 影响说明

- 当 `PYTHONWARNINGS` 或你的项目逻辑将告警升级为异常（`warnings.simplefilter("error")`）时，
  该警告会直接导致 `import wandb` 失败。
- 多 GPU 训练环境（如 `torchrun`）下，各子进程都会触发同样的导入异常，训练任务整体终止。

---

## ✅ 推荐解决方案

### ✅ 方案 1：降级 `pydantic`（最简单、最稳妥）

使用 **uv** 或 **pip** 降级到兼容版本：

```bash
# 使用 uv（推荐）
uv pip install -p .venv "pydantic<2.12"

# 或使用 pip
pip install "pydantic<2.12"
````

验证版本：

```bash
uv run -p .venv python -c "import pydantic; print(pydantic.__version__)"
# 应输出 2.11.x
```

此方法不会影响 wandb 功能，也无需改动项目代码。

---

### 🩹 方案 2：手动热补丁（临时修复）

如果必须使用 `pydantic==2.12`，可在虚拟环境中编辑：

```
.venv/lib/python3.12/site-packages/wandb/_pydantic/field_types.py
```

修改以下两行：

```python
# 原
Typename = Annotated[T, Field(repr=False, frozen=True, alias="__typename")]
GQLId    = Annotated[StrictStr, Field(repr=False, frozen=True)]

# 改
Typename = Annotated[T, Field(alias="__typename")]
GQLId    = Annotated[StrictStr, Field()]
```

---

## 🧭 总结

| 项目          | 推荐方案                   | 状态       |
| ----------- | ---------------------- | -------- |
| ⚙️ pydantic | 降级至 `<2.12`（例如 2.11.x） | ✅ 稳定     |
| 📦 wandb    | 等待官方发布修复版              | ⏳ 处理中    |
| 🚀 临时方案     | 热补丁或白名单告警过滤            | 可行但非长期方案 |

---

## 🗓️ 结论

截至 **2025-10-18**，`wandb` 官方尚未完全修复与 `pydantic 2.12+` 的兼容问题。
如果你的训练脚本在导入阶段出现 `UnsupportedFieldAttributeWarning`，
**建议立即将 pydantic 降级到 2.11.x**，即可恢复正常使用。

---

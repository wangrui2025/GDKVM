# GDKVM 设计规范

> 本文档描述 GDKVM 论文项目页面的设计原则与视觉规范。
> 任何修改必须遵守本规范，确保跨浏览器一致性。

---

## 1. 项目概述

| 属性 | 值 |
|------|-----|
| 论文标题 | GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule |
| 会议 | ICCV 2025 |
| 技术栈 | Astro 6.x + Tailwind CSS v4 + TypeScript |
| 部署 URL | `https://wangrui2025.github.io/GDKVM/` |
| i18n | `prefixDefaultLocale: true` — `/en/` 和 `/zh/` |

---

## 2. 页面结构

| 页面 | 路由 | 说明 |
|------|------|------|
| 首页 | `/en/` `/zh/` | 论文标题、作者、摘要、方法图、挑战、BibTeX |
| 工具页 | `/en/tool/` `/zh/tool/` | 在线工具（心肌分割可视化） |
| 404 | `/en/404/` | 错误页 |

---

## 3. 字体规范

### 3.1 字体栈

全站统一使用 Times New Roman 系列衬线字体（含 CJK 回退）：

```css
/* src/styles/global.css */
body, h1, h2, h3, h4, h5, h6 {
  font-family: 'Times New Roman', 'Noto Serif SC', Georgia, serif;
}
```

| 用途 | 字体栈 |
|------|--------|
| 全局正文/标题 | `Times New Roman`, `Noto Serif SC`, Georgia, serif |
| BibTeX / 等宽 | `Fira Code`, Consolas, monospace |
| 公式 | KaTeX (CDN `@0.16.47`) |

### 3.2 字号规范

| 元素 | 字号 |
|------|------|
| 论文标题 | 32pt |
| 作者名 | 18pt |
| 机构 | 14pt |
| Section 标题 | 18pt |
| 正文 | 16pt |
| BibTeX | 13pt |

**禁止改回** Inter / Plus Jakarta Sans 等无衬线字体。

---

## 4. 颜色规范

### 4.1 主色板（CSS 变量）

| Token | Light | Dark | 用途 |
|-------|-------|------|------|
| `--color-primary` | `#2563eb` | `#3b82f6` | 链接、强调 |
| `--color-primary-dark` | `#1d4ed8` | `#2563eb` | hover 状态 |
| `--color-bg` | `#ffffff` | `#111827` | 页面背景 |
| `--color-text` | `#1f2937` | `#f9fafb` | 主文本 |
| `--color-text-muted` | `#6b7280` | `#9ca3af` | 次文本 |
| `--color-border` | `#e5e7eb` | `#374151` | 边框 |
| `--color-surface` | `#f9fafb` | `#1f2937` | 卡片背景 |

### 4.2 暗色模式实现

Tailwind v4 `@custom-variant dark (&:where(.dark, .dark *))`。

```css
/* src/styles/global.css */
@custom-variant dark (&:where(.dark, .dark *));

@theme {
  --color-primary: oklch(55% 0.2 265);
  --color-bg: oklch(98% 0 0);
  --color-bg-dark: oklch(5% 0 0);
  /* ... */
}
```

切换机制：JS 切换 `html` 上的 `.dark` class，ThemeToggle 组件使用 `data-action="theme-toggle"` 事件委托。

---

## 5. a11y 规范

### 5.1 对比度要求

| 场景 | 最小对比度 |
|------|-----------|
| 正常文本 | ≥ 4.5:1 |
| 大文本（≥18pt 或 14pt bold） | ≥ 3:1 |
| UI 组件边界 | ≥ 3:1 |

### 5.2 Focus 可见性

禁止 `outline: none` 而不提供 `:focus-visible` 替代方案。所有交互按钮必须显示 focus 环。

### 5.3 图片 alt 文本

所有 `<img>` 必须有描述性 `alt` 属性，装饰性图片使用 `alt=""`。

### 5.4 语义化标题

每页仅一个 `<h1>`，标题层级不得跳级（如 `h1 → h3`）。

### 5.5 验证命令

```bash
# 检查缺失 alt
grep -rn "<img" src/ --include="*.astro" | grep -v "alt=" | grep -v "decorative"
# 检查 outline: none
grep -rn "outline: none\|outline:none" src/ --include="*.css" --include="*.astro"
# 检查 h1 数量
grep -rn "<h1" src/pages/ --include="*.astro"
```

---

## 6. 组件清单

| 组件 | 文件 | 说明 |
|------|------|------|
| HomePage | `src/components/HomePage.astro` | 首页核心内容（标题/作者/摘要/方法图/挑战/BibTeX） |
| Section | `src/components/Section.astro` | 内容区块（标题 + 内容） |
| ActionButton | `src/components/ActionButton.astro` | 操作按钮（Copy、Download） |
| CopyButton | `src/components/CopyButton.astro` | BibTeX 复制按钮 |
| Footer | `src/components/Footer.astro` | 页脚（版权、链接） |
| LangSwitcher | `src/components/LangSwitcher.astro` | 中英文切换 |
| ThemeToggle | `src/components/ThemeToggle.astro` | 暗黑模式切换 |

---

## 7. 组件事件处理

### 7.1 事件委托模式

所有交互按钮使用 `data-action` 属性 + 事件委托，**禁止** `onclick` 内联：

```javascript
// src/layouts/Layout.astro — 全局事件委托
document.addEventListener('click', (e) => {
  const target = e.target.closest('[data-action]');
  if (!target) return;
  const action = target.dataset.action;
  if (action === 'theme-toggle') toggleTheme();
  if (action === 'copy-bibtex') copyBibtex(target);
});
```

### 7.2 已知 data-action

| action 值 | 组件 | 行为 |
|-----------|------|------|
| `theme-toggle` | ThemeToggle.astro | 切换暗黑模式 |
| `copy-bibtex` | HomePage.astro | 复制 BibTeX 到剪贴板 |

### 7.3 ThemeToggle 规范

```astro
<!-- src/components/ThemeToggle.astro -->
<button id="theme-toggle" data-action="theme-toggle" ...>
  <Icon name="lucide:sun" class="w-4 h-4 hidden dark:block" />
  <Icon name="lucide:moon" class="w-4 h-4 block dark:hidden" />
</button>
```

Theme 脚本在 `Layout.astro` 中通过 `astro:page-load` 初始化，必须在 `astro:after-swap` 做 cleanup。

---

## 8. @media print 规范

### 8.1 核心规则

```css
@page { size: A4 portrait; margin: 15mm 20mm; }
```

### 8.2 打印时隐藏的元素

- 导航控件（ThemeToggle、LangSwitcher）
- 页脚
- 外部链接的 URL（`a[href]:after { content: none }`）
- 任何固定定位元素（`.fixed`）

### 8.3 打印时保留的元素

- 论文标题、作者、机构
- 摘要内容
- BibTeX 引用块
- 图片（`max-width: 100%; height: auto`）

### 8.4 验证方法

```bash
# 本地预览
npm run preview
# 浏览器 DevTools → ... → More tools → Rendering → Emulate CSS media print
# 或直接 Ctrl+P / Cmd+P 打印预览
```

---

## 9. 关键设计决策

### 9.1 外部 ICCV Poster PDF 链接

首页 `HomePage.astro` 引用 `https://iccv.thecvf.com/media/PosterPDFs/ICCV%202025/2134.png`。这是 ICCV 官方海报 PDF，为只读外链，**无需下载本地化**。

### 9.2 无本地 Poster/Slides 组件

GDKVM 不提供本地 poster 渲染和 slides 演示。外部海报 PDF 通过 `<a target="_blank" rel="noopener">` 链接引用，用户点击后在 ICCV 官网查看。

### 9.3 KaTeX CDN 样式

公式渲染使用 `https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css`，通过 CDN 加载，无需本地安装。

### 9.4 工具页独立样式

`src/styles/toolstyle.css` 和 `src/scripts/tool.ts` 专门服务于 `/tool/` 页面，不影响主页样式。

---

## 10. i18n 内容对等

所有文案必须同时存在于 `src/content/homepage/en.json` 和 `zh.json`，key 集合必须完全一致。

验证命令：

```bash
diff <(node -e "console.log(Object.keys(require('./src/content/homepage/en.json')).sort().join('\n'))") \
     <(node -e "console.log(Object.keys(require('./src/content/homepage/zh.json')).sort().join('\n'))")
```

---

## 11. 修改 Checklist

任何涉及 UI 样式或组件结构的修改，发布前必须验证：

1. [ ] `npm run build` 通过（0 errors）
2. [ ] `npx astro check` 通过（0 errors / 0 warnings）
3. [ ] 中英文页面内容对等（key 集合一致）
4. [ ] 外部链接可访问（arxiv、code、CVF paper）
5. [ ] 暗黑模式正常切换
6. [ ] Playwright E2E 通过：`npx playwright test`
7. [ ] a11y 验证（alt 文本、focus 可见性）

---

## 12. CI/CD

部署通过 GitHub Actions，workflow 文件在 `.github/workflows/`。Node.js 版本使用最新 LTS。

---

## 14. 学术图片 CDN 策略

### 14.1 背景

学术图片（论文图表、logo、海报）统一托管在 `mykcs/academic` 仓库，通过 jsDelivr CDN 分发。项目页面通过语义化版本标签（`@v1.1.0`）引用。

### 14.2 OSA 模式（当前采用）

| 维度 | 实现 |
|------|------|
| 图片组件 | `<img>`（原生 HTML） |
| CDN | `cdn.jsdelivr.net/gh/mykcs/academic@v1.1.0/...` |
| `remotePatterns` | 仅 `mykcs.github.io` + `raw.githubusercontent.com`（**不含** `cdn.jsdelivr.net`） |
| 构建时验证 | 不触发 — `<img>` 不走 Astro 图片管线 |
| 版本管理 | 语义化 tag，可读可预期 |

**为什么更优：**
- jsDelivr 全球边缘节点，延迟低；raw.githubusercontent.com 直连 GitHub 源站，无 CDN 层
- 语义化版本（`@v1.1.0`）优于 commit SHA（`84e996d...`）
- 架构上规避了 jsDelivr 301 重定向与 Astro 构建时验证的冲突

### 14.3 错误模式（已废弃）

| 维度 | 实现 |
|------|------|
| 图片组件 | `<Image>` from `astro:assets` |
| CDN | jsDelivr URL |
| `remotePatterns` | 包含 `cdn.jsdelivr.net` |

`<Image>` 组件在构建时会请求远程 URL 验证，jsDelivr 的 301 重定向可能导致构建警告或失败。

### 14.4 变更记录

- **2026-05-28**：从错误模式迁移至 OSA 模式，`HomePage.astro` 全部 `<Image>` 替换为 `<img>`，`astro.config.mjs` 的 `remotePatterns` 移除 `cdn.jsdelivr.net`

---

## 13. 变更历史

| 日期 | 变更 |
|------|------|
| 2026-05-28 | 新增 §14 学术图片 CDN 策略（OSA 模式：`<img>` + jsDelivr 语义化版本，优于 `<Image>` + raw.githubusercontent） |
| 2026-05-27 | 新增 §3 字体规范（含 CSS 实现）、§4.2 暗色模式实现、§5 a11y 规范、§7 组件事件处理（data-action 模式）、§8 @media print 规范、§10 i18n 对等验证命令、§11 checklist 新增 a11y 验证、§13 变更历史 |

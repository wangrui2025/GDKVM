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

全站统一 Times New Roman：

| 元素 | 字体 | 字号 |
|------|------|------|
| 论文标题 | Times New Roman | 32pt |
| 作者名 | Times New Roman | 18pt |
| 机构 | Times New Roman | 14pt |
| Section 标题 | Times New Roman | 18pt |
| 正文 | Times New Roman | 16pt |
| 公式 | KaTeX (CDN) | — |
| BibTeX | monospace | 13pt |

---

## 4. 颜色规范

| 用途 | 色值 |
|------|-----|
| 主色 | `#2563eb` |
| 主色深 | `#1d4ed8` |
| 背景 | `#ffffff` |
| 文字主色 | `#1f2937` |
| 文字次色 | `#6b7280` |
| 边框 | `#e5e7eb` |
| 卡片背景 | `#f9fafb` |
| 链接 | `#2563eb` |

---

## 5. 组件清单

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

## 6. 关键设计决策

### 6.1 外部 ICCV Poster PDF 链接

首页 `HomePage.astro` 引用 `https://iccv.thecvf.com/media/PosterPDFs/ICCV%202025/2134.png`。这是 ICCV 官方海报 PDF，为只读外链，**无需下载本地化**。

### 6.2 无本地 Poster/Slides 组件

GDKVM 不提供本地 poster 渲染和 slides 演示。外部海报 PDF 通过 `<a target="_blank" rel="noopener">` 链接引用，用户点击后在 ICCV 官网查看。

### 6.3 KaTeX CDN 样式

公式渲染使用 `https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css`，通过 CDN 加载，无需本地安装。

### 6.4 工具页独立样式

`src/styles/toolstyle.css` 和 `src/scripts/tool.ts` 专门服务于 `/tool/` 页面，不影响主页样式。

---

## 7. i18n 内容对等

所有文案必须同时存在于 `src/content/homepage/en.json` 和 `zh.json`，key 集合必须完全一致。

---

## 8. 修改 Checklist

任何涉及 UI 样式或组件结构的修改，发布前必须验证：

1. [ ] `npm run build` 通过（0 errors）
2. [ ] `npx astro check` 通过（0 errors / 0 warnings）
3. [ ] 中英文页面内容对等（key 集合一致）
4. [ ] 外部链接可访问（arxiv、code、CVF paper）
5. [ ] 暗黑模式正常切换
6. [ ] Playwright E2E 通过：`npx playwright test`

---

## 9. CI/CD

部署通过 GitHub Actions，workflow 文件在 `.github/workflows/`。Node.js 版本使用最新 LTS。

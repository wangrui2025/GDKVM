# GDKVM — 项目上下文与决策记录

> 本文件记录影响项目长期演进的技术决策。任何推翻既有决策的行为，必须在此文件中记录原因。

---

## Tailwind CSS v4 集成方式决策

**决策日期**: 2026-05-22
**决策**: 使用 `@tailwindcss/vite`（而非 `@tailwindcss/postcss`）

### 背景
项目最初使用 `@tailwindcss/postcss` + `postcss.config.mjs` 接入 Tailwind v4。在网站审计中发现此配置属于兼容性回退方案，非官方推荐路径。

### 迁移详情
- **Before**: `postcss.config.mjs` + `@tailwindcss/postcss`
- **After**: `astro.config.mjs` integrations 中直接引入 `tailwindcss()`（来自 `@tailwindcss/vite`）
- **删除文件**: `postcss.config.mjs`
- **修改文件**: `package.json`（替换依赖）、`astro.config.mjs`（添加 integration）

### 决策依据
1. **官方推荐**: Tailwind v4 官方文档明确推荐 `@tailwindcss/vite` 作为 Vite 项目的首选集成方式，描述为"最快、最可靠"。
2. **性能**: Vite 插件模式比 PostCSS 插件模式构建更快，且与 Astro 的 Vite 底层集成更紧密。
3. **一致性**: 与组织内其他活跃维护站点（OSA、mykcs.github.io）保持一致，降低跨站点维护成本。
4. **简化配置**: 无需额外的 `postcss.config.mjs` 文件，配置集中在 `astro.config.mjs`。

### 何时可以推翻此决策
只有在以下场景才允许回退到 `@tailwindcss/postcss`：
- 项目明确需要与其他 PostCSS 插件（如 `autoprefixer`、`postcss-preset-env`、`cssnano` 等）共存，且这些插件无法通过 Vite 插件链实现。
- `@tailwindcss/vite` 出现与 Astro 版本的兼容性 bug，且官方未在 2 个 patch 版本内修复。

### 验证记录
- `npx astro check`: 0 errors / 0 warnings / 0 hints
- `npm run build`: 通过（9 pages）
- 构建时间: ~850ms（与迁移前持平）

---

## 学术资产库化状态

**状态**: 未迁移（本地托管）
**位置**: `public/paper/fig/`（约 23MB，43 个文件）

### 未迁移原因
- 图片为论文专属图表（GDKVM 架构图、实验结果图、热图等），通用性低。
- `mykcs/academic` 仓库目前主要托管头像、学校/会议 logo 等通用学术资产。
- 迁移需将 23MB 图片推送到 academic 仓库并生成版本化 CDN URL，工作量与收益比不高。

### 如果未来迁移
1. 在 `mykcs/academic/images/publications/gdkvm/` 下建立目录结构
2. 更新 `HomePage.astro` 中所有 `${import.meta.env.BASE_URL}/paper/fig/...` 引用为 jsDelivr CDN URL
3. 删除 `public/paper/fig/` 目录
4. 注意：`<Image>` 组件引用远程图片时，需在 `astro.config.mjs` 中声明 `image.remotePatterns`

---

## 内联 CSS 集成设计说明

**集成**: `src/integrations/inline-critical-css.mjs`
**行为**: 在 `astro:build:done` 阶段，将 `_astro/*.css` 链接替换为内联 `<style>` 标签

### 设计意图
消除 render-blocking CSS 请求，提升 LCP（Largest Contentful Paint）。对于静态站点部署在 GitHub Pages（无 CDN 边缘缓存）的场景，内联 CSS 可以减少一次 RTT。

### 已知现象
- 每页 HTML 内联 59-65KB CSS（这是整站共享的 Tailwind utility CSS + 组件样式）
- `/index.html` 和 `/reprod/index.html` 为纯 redirect 页面（`<meta http-equiv="refresh">`），无 CSS 可内联，日志显示 "No CSS links" —— 此为预期行为

### 注意事项
- 这不是"CSS 重复"bug。每个 HTML 文件独立，内联是设计行为。
- 如果站点页面数增长到 50+，内联 CSS 的总部署体积会显著膨胀，届时应考虑：
  - 提取公共 CSS 到外部文件（恢复 `<link rel="stylesheet">`）
  - 或使用 HTTP/2 push / 边缘缓存

---

## 技术栈快照

| 组件 | 版本 | 备注 |
|------|------|------|
| Astro | ^6.3.6 | `experimental.rustCompiler: true` |
| Tailwind CSS | ^4.3.0 | `@tailwindcss/vite` 集成 |
| TypeScript | ^5.9.3 | |
| KaTeX | ^0.16.47 | CDN auto-render 用于公式渲染 |
| astro-icon | ^1.1.5 | Iconify 图标集 |

---

## 相关仓库

- **学术资产库**: `~/Repo/webs/academic/`（`mykcs/academic` GitHub 仓库的本地副本）
- **主站**: `~/Repo/webs/mykcs.github.io/`
- **OSA 项目页**: `~/Repo/webs/OSA/`

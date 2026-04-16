# GDKVM — Astro 6.x 项目速查

> 技术栈：Astro v6.1.6 + Tailwind v4 + TypeScript
> 部署目标：GitHub Pages (`https://wangrui2025.github.io/GDKVM/`)

---

## 快速验证（任何修改后必须执行）

```bash
npx astro check        # 0 errors / 0 warnings / 0 hints
npm run build          # 必须成功
npx playwright test    # 4 tests passed
```

---

## Astro 6.x 现代化守则

### URL 构造
- ✅ 使用 `new URL(path, Astro.site).href`
- ❌ 禁止字符串拼接：`` `${Astro.site}${path}` ``

### i18n 路由
- 配置已冻结：`prefixDefaultLocale: true` + `redirectToDefaultLocale: true`
- 跳转链接用 `getRelativeLocaleUrl(locale, path)`（`astro:i18n`）
- `src/pages/index.astro` 必须保留（Astro i18n 要求），但保持最小化（空 frontmatter 即可）

### 图片
- ✅ 使用 `astro:assets` 的 `<Image />`
- ❌ 禁止 `<Image format="webp" />`（Astro 6 已废弃 format 属性）

### 脚本与样式
- 页面级 CSS/JS 必须放在 `src/` 下，通过 `import` 引入
- ❌ 禁止在 `public/` 放页面专属资源
- `is:inline` 仅允许用于：第三方 CDN 脚本、JSON-LD、SW 注册、FOUC theme 脚本
- 事件交互统一用事件委托，禁止 `__xxxBound` 标记

### View Transitions
- ✅ 使用 `<ClientRouter />`
- ❌ 禁止 `<ViewTransitions />`（Astro 6 废弃）

---

## 已知限制（不可修复）

`npm run build` 会出现以下 warning：

```
[WARN] [build] Could not render `` from route `/` as it conflicts with higher priority route `/`.
```

**原因**：`redirectToDefaultLocale: true` 时，Astro 自动生成根路由 `/` 的 redirect，而 `src/pages/index.astro` 的存在会与该自动路由冲突。同时 Astro **要求** `src/pages/index.astro` 必须存在，否则 build 报错 `MissingIndexForInternationalizationError`。

**结论**：此为 Astro by-design 行为，保留空 `src/pages/index.astro` 即可，warning 无害。

---

## Build 后回归检查（可选脚本）

```bash
# 1. Build 产物
ls -d dist/

# 2. ClientRouter 激活
grep -r "astro-route-announcer" dist/

# 3. 结构化数据
grep -r "application/ld+json" dist/

# 4. SW 注册
grep -r "navigator.serviceWorker.register" dist/

# 5. OG 标签
grep -r "og:image" dist/
```

---

## 项目结构要点

- `src/pages/[lang]/` — 所有多语言页面
- `src/layouts/Layout.astro` — 全局布局（SEO、ClientRouter、theme/SW 脚本）
- `src/components/HomePage.astro` — 首页内容组件
- `src/scripts/tool.ts` — tool 页面交互脚本
- `src/styles/toolstyle.css` — tool 页面样式
- `e2e/smoke.spec.ts` — Playwright 关键流程测试

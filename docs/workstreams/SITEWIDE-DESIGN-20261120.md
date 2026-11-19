# GDKVM 全站排版与年月日统一

> Draft · Implementation candidate ready · Browser/merge acceptance pending
> Size: M · Confidence: medium

## 目标与来源

2026年11月20日维护者要求本站所有网页实际消费统一字体、字号、字重、行距、间距与年月日规则；跨网站保留独立风格。必须实施、验收、合并 main；规范文件不代表完成。

## 当前事实与去重

工具页和主站样式各自定义；public/404.html 绕过 Astro 布局；footer 日期写死且未格式化。无重叠开放 PR。

## 既有 owner 与边界

website/src/styles/global.css、toolstyle.css、layouts/Layout.astro；website/DESIGN.md

保持现有论文网站 serif、KaTeX 数学字体、工具功能和科学内容；独立 404、双语页面及 /reprod 跳转纳入；出版日期不改动。

## 验收合同

- [ ] 在实际共享 owner 管理语义字体、字号、字重、行高、间距；旧重复定义迁移，必要论文/数学/海报/幻灯片/代码角色明确记录。
- [ ] 全部生成 HTML 路由包含普通页面、双语、独立 HTML、404 和重定向；实际加载及计算值可检查，新增漏接页面必须失败。
- [ ] 完整日期为中文 YYYY年M月D日、英文 YYYY-MM-DD；只有年/月的保留精度；timestamp 保留明确时区；源码数据、URL、代码、datetime不改写。
- [ ] 桌面和440×956 CSS px 手机模拟覆盖所有路由；保持主题、语言、导航、交互；无未解释的页面横向溢出/文字裁切；必要画布/打印模式单独验收。
- [ ] 检查具有漏接样式与非标准日期负例，不以规则文件存在冒充真实消费。
- [ ] cd website && pnpm install --frozen-lockfile && pnpm run build && pnpm exec playwright test；验证 GitHub Pages /GDKVM/ 的发布 commit。
- [ ] 当前 head 必要检查通过后更新证据、Ready、合并 main；读回 merge SHA，部署身份另行核实。

## 执行与恢复

单一写入 owner，按总优先级逐站推进。已获实现、验证、合并授权；不改科学结果、隐私、访问保护、现有发布平台。先读本仓当前文档和源码。回退使用本PR逆向变更。

## 证据

目前仅合同；实现 NOT DONE；构建/浏览器 NOT RUN。完成后补命令、覆盖数、结果和证据。


## 实施记录（2026年11月20日）

- 单一 `website/src/styles/tokens.css` → `global.css` → `Layout` / `FallbackLayout`。Tailwind 字级、行高、字重、间距及容器宽度均映射到 owner；普通 UI 的独立间距/字号声明已迁移。
- 保留学术衬线身份，正文统一为 16px / 1.65，图注 14px，元信息 13px；同级 section 28px、小节 20px，三个内容宽度角色。打印、数学与原图几何单独说明。
- 原 `public/404.html` 改为 Astro 生成同名真实主机 fallback，去掉页面中误显示的 frontmatter 注释；两个历史跳转保持目的地并共享 fallback 布局。
- 日期只在显示处投影，JSON 原始记录不变。普通页脚的 2026-06-04 与复现历史页脚的 2025-12-15 各自保留其真实日期；完整日期、年月、年份分别保留精度。
- 本地 `npx --yes pnpm@10.33.0 run build` PASS：11/11 HTML 样式依赖及消费者声明，日期/缺失样式正反例 PASS。当前环境阻止脚本浏览器启动，cloud 浏览器也不能连接本地 preview；未将静态审计称为渲染验收。
- 正式 `Deploy to GitHub Pages / test` 将运行全部生成路由 × Chromium/WebKit × 440×956/1440×1000 × light/dark，实际扰动 font/size/leading/spacing，同时对真实 Tailwind padding、caption、heading、content width 做见证；保留原语言/主题/工具计算/ClientRouter 回归，并提供候选截图。
- 待当前 head CI 通过后完成 Ready、受规则约束的 merge 和 Pages 正式部署身份读回。此条在正式验收前保持未完成。

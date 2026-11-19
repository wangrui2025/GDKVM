# GDKVM 全站排版与年月日统一

> Draft · TASK SPEC ONLY · Implementation: NOT DONE
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

# Website development and acceptance

This page routes maintenance to the existing website implementation; it does not
replace the paper, source data, or design specification.

| Area | Local owner and acceptance |
| --- | --- |
| Agent reading | Source-derived localized guide, llms.txt, agent/site.json and agent/SKILL.md; `scripts/check-agent-reader.mjs` validates the built files. Follow the cited paper and project source before claiming scientific results. |
| Design language | [DESIGN.md](DESIGN.md), `src/styles/tokens.css`, shared components and existing design guards; preserve the paper/poster/print roles. |
| Responsive | Desktop and Chrome 440×956 CSS-pixel portrait, DPR3/touch when emulated; exercise navigation, theme, copy, overflow and relevant project-specific layouts. |
| Human expression | Localized content and paper source are authoritative. Preserve evidence, dates, figures, reproducibility limitations and the scientific distinction between examples and results. |
| Engineering | From the site root run `pnpm install --frozen-lockfile && pnpm build`. Reuse the current lockfile, framework and output checks. |
| CI/release | Existing GitHub Actions and Pages workflow own validation and main publication; verify the exact candidate and actual served version before claiming release. No second provider. |
| Public output | Use approved public paper and project materials only. Do not add account credentials, private data, or inaccessible sources to published output. |
| Accessibility/performance | Existing design/browser suite plus `e2e/reader-accessibility.spec.ts` verifies contrast and keyboard access of paper controls. Check relevant page-level accessibility and loading behavior, and distinguish unrun checks. |
| Governance | Repository AGENTS, local design and this development route locate the actual code and tests. Check active PRs and current branch before making changes. |

This is an implementation map, not a blanket PASS; current test and deployment
evidence must be obtained for each change. Real device and independent-reader
behavior are separate from browser simulation and build results.

Feature-level goals and protected semantics: [Feature Design](../docs/design/features/README.md). This file retains its existing ownership and history.

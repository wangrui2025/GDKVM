// Content Collections config
//
// Intentionally empty: GDKVM's UI text is managed as plain JSON imports
// in `src/i18n/index.ts` rather than via Astro's content collections.
// Rationale:
//   - We use type-narrowed `t(lang, key)` lookups (not frontmatter).
//   - We co-locate en/zh pairs in `src/content/<section>/{en,zh}.json`
//     so the bilingual-sync protocol stays explicit per section.
//   - Adding a `defineCollection({ loader: glob(), schema })` here would
//     force all JSON through zod validation, which is overkill for
//     human-authored bilingual copy.
//
// If/when a non-i18n data source (e.g. a publications list sourced from
// a CMS) is added, register it here.
export const collections = {};

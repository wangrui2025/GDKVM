export type Locale = 'en' | 'zh';

import enHome from '../content/homepage/en.json';
import zhHome from '../content/homepage/zh.json';
import enReprod from '../content/reprod/en.json';
import zhReprod from '../content/reprod/zh.json';
import enTool from '../content/tool/en.json';
import zhTool from '../content/tool/zh.json';
import enFooter from '../content/footer/en.json';
import zhFooter from '../content/footer/zh.json';
import en404 from '../content/404/en.json';
import zh404 from '../content/404/zh.json';

const dict = {
  en: { home: enHome, reprod: enReprod, tool: enTool, footer: enFooter, '404': en404 },
  zh: { home: zhHome, reprod: zhReprod, tool: zhTool, footer: zhFooter, '404': zh404 },
};

type Dictionary = typeof dict.en;

type NestedKeyOf<ObjectType extends object> = {
  [Key in keyof ObjectType & (string | number)]: ObjectType[Key] extends object
    ? `${Key}` | `${Key}.${NestedKeyOf<ObjectType[Key]>}`
    : `${Key}`;
}[keyof ObjectType & (string | number)];

export type TranslationKey = NestedKeyOf<Dictionary>;

/**
 * Runtime type guard for locale values. Use this to narrow
 * `Astro.params.lang` (typed as `string | undefined`) to the
 * `Locale` type — avoids unsafe `as` casts at call sites.
 */
export function isLocale(value: string | undefined): value is Locale {
  return value === 'en' || value === 'zh';
}

export function t(locale: Locale | string, key: TranslationKey): string {
  const keys = key.split('.');
  let val: unknown = dict[locale as Locale] ?? dict.en;
  for (const k of keys) {
    if (val && typeof val === 'object' && k in val) {
      val = (val as Record<string, unknown>)[k];
    } else {
      return key;
    }
  }
  return typeof val === 'string' ? val : key;
}

/**
 * Strip the site's `BASE_URL` prefix and the leading locale segment
 * (e.g. `/en` or `/zh`) from an absolute Astro pathname, returning the
 * locale-free path expected by `getRelativeLocaleUrl`.
 *
 * Centralized here so the regex isn't duplicated between Layout.astro
 * (SEO hreflang) and LangSwitcher.astro (locale toggle). Trailing slashes
 * are normalized; the result falls back to `/` for the site root.
 */
export function stripLocaleFromPath(pathname: string, baseUrl: string): string {
  const base = baseUrl.replace(/\/$/, '');
  return pathname.replace(new RegExp(`^${base}/(en|zh)`), '').replace(/\/$/, '') || '/';
}

/**
 * Shared `getStaticPaths` for any page routed under `[lang]/`.
 * Emits the canonical en + zh pairs. Use as:
 *   export { getStaticPaths } from '../../i18n';
 */
export const SUPPORTED_LOCALES: readonly Locale[] = ['en', 'zh'] as const;
export function getStaticPaths() {
  return SUPPORTED_LOCALES.map((lang) => ({ params: { lang } }));
}

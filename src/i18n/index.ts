export type Locale = 'en' | 'zh';

import enHome from '../content/homepage/en.json';
import zhHome from '../content/homepage/zh.json';
import enReprod from '../content/reprod/en.json';
import zhReprod from '../content/reprod/zh.json';
import enTool from '../content/tool/en.json';
import zhTool from '../content/tool/zh.json';
import enFooter from '../content/footer/en.json';
import zhFooter from '../content/footer/zh.json';

const dict = {
  en: { home: enHome, reprod: enReprod, tool: enTool, footer: enFooter },
  zh: { home: zhHome, reprod: zhReprod, tool: zhTool, footer: zhFooter },
};

type Dictionary = typeof dict.en;

type NestedKeyOf<ObjectType extends object> = {
  [Key in keyof ObjectType & (string | number)]: ObjectType[Key] extends object
    ? `${Key}` | `${Key}.${NestedKeyOf<ObjectType[Key]>}`
    : `${Key}`;
}[keyof ObjectType & (string | number)];

export type TranslationKey = NestedKeyOf<Dictionary>;

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

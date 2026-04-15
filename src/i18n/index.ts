import en from './en.json';
import zh from './zh.json';

const dict = { en, zh };

type Dictionary = typeof dict.en;
type NestedKeyOf<ObjectType extends object> = {
  [Key in keyof ObjectType & (string | number)]: ObjectType[Key] extends object
    ? `${Key}` | `${Key}.${NestedKeyOf<ObjectType[Key]>}`
    : `${Key}`;
}[keyof ObjectType & (string | number)];

export type Locale = keyof typeof dict;
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

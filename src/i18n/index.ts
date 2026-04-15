import en from './en.json';
import zh from './zh.json';

const dict = { en, zh };

export function t(locale: string, key: string): string {
  const keys = key.split('.');
  let val: any = dict[locale as keyof typeof dict] ?? dict.en;
  for (const k of keys) val = val?.[k];
  return typeof val === 'string' ? val : key;
}

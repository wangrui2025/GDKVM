/** Calendar display only: no timezone parsing and no changes to source data. */
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const monthPattern = '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\.?';
const before = '(?<![A-Za-z0-9_/+.-])';
const after = '(?![A-Za-z0-9_/+-]|\\.[A-Za-z0-9]{1,12})';
const pad = number => String(number).padStart(2, '0');
function parse(value) {
  const match = /^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$/.exec(value);
  if (!match) return null;
  const [, year, month, day] = match;
  if (!month) return { year };
  const y = Number(year), m = Number(month), d = Number(day);
  if (m < 1 || m > 12) return null;
  const days = [31, y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (day && (d < 1 || d > days[m - 1])) return null;
  return { year, month: m, day: day ? d : undefined };
}
export function formatCalendarDate(value, locale = 'en') {
  const parsed = parse(value);
  if (!parsed) return value;
  const { year, month, day } = parsed;
  if (!month) return year;
  return locale === 'zh' ? `${year}年${month}月${day ? `${day}日` : ''}` : value;
}
function project(original, year, month, day, locale) {
  const iso = `${year}-${pad(month)}${day == null ? '' : `-${pad(day)}`}`;
  return parse(iso) ? formatCalendarDate(iso, locale) : original;
}
/** Use at prose boundaries only; never rewrite code, attributes or metadata. */
export function formatCalendarText(value, locale = 'en') {
  const monthNumber = name => months.findIndex(month => month.toLowerCase() === name.slice(0, 3).toLowerCase()) + 1;
  let result = value.replace(new RegExp(before + monthPattern + '\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,\\s*|\\s+)(\\d{4})' + after, 'gi'),
    (all, month, day, year) => project(all, year, monthNumber(month), Number(day), locale));
  result = result.replace(new RegExp(before + '(\\d{1,2})(?:st|nd|rd|th)?\\s+' + monthPattern + '\\s+(\\d{4})' + after, 'gi'),
    (all, day, month, year) => project(all, year, monthNumber(month), Number(day), locale));
  result = result.replace(new RegExp(before + monthPattern + '\\s+(\\d{4})' + after, 'gi'),
    (all, month, year, offset, source) => /\d{1,2}(?:st|nd|rd|th)?\s+$/i.test(source.slice(0, offset)) ? all : project(all, year, monthNumber(month), null, locale));
  result = result.replace(new RegExp(before + '(\\d{4})\\s*年\\s*(\\d{1,2})\\s*月\\s*(\\d{1,2})\\s*日' + after, 'g'),
    (all, year, month, day) => project(all, year, Number(month), Number(day), locale));
  result = result.replace(new RegExp(before + '(\\d{4})\\s*年\\s*(\\d{1,2})\\s*月(?!\\s*\\d{1,2}\\s*日)' + after, 'g'),
    (all, year, month) => project(all, year, Number(month), null, locale));
  return result.replace(new RegExp(before + '\\d{4}-\\d{2}(?:-\\d{2})?' + after, 'g'), date => formatCalendarDate(date, locale));
}

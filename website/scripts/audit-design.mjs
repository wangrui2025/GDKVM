import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { resolve, relative, dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { formatCalendarText } from '../src/utils/calendar-date.mjs';

export const htmlFiles = dir => readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
  const file = join(dir, entry.name);
  return entry.isDirectory() ? htmlFiles(file) : entry.name.endsWith('.html') ? [file] : [];
});

export function inspectHTML(html, loadStylesheet = () => '') {
  const inline = [...html.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/gi)].map(match => match[1]);
  const links = [...html.matchAll(/<link\b[^>]*>/gi)].filter(match => /rel=["']stylesheet["']/.test(match[0]));
  const css = inline.concat(links.map(match => loadStylesheet(/href=["']([^"']+)/.exec(match[0])?.[1] || ''))).join('\n');
  const errors = [];
  if (!/data-design-system=["']gdkvm["']/.test(html)) errors.push('missing shared layout marker');
  for (const token of ['design-system', 'font-body', 'size-root', 'leading-body', 'space-unit', 'width-content']) {
    if (!new RegExp(`--${token}\\s*:`).test(css)) errors.push(`missing token ${token}`);
  }
  if (!/font-family\s*:\s*var\(--font-body\)/.test(css)) errors.push('missing body font consumer');
  if (!/line-height\s*:\s*var\(--leading-body\)/.test(css)) errors.push('missing body leading consumer');
  if (!/(?:padding|gap)[\w-]*\s*:[^;}]*var\(--space-/.test(css)) errors.push('missing spacing consumer');
  const locale = /<html\b[^>]*lang=["']zh(?:-[^"']*)?["']/i.test(html) ? 'zh' : 'en';
  const prose = html.replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(script|style|pre|code|svg|math)\b[^>]*>[\s\S]*?<\/\1>/gi, '')
    .replace(/<[^>]*>/g, ' ').replace(/&nbsp;|&#160;/g, ' ');
  if (formatCalendarText(prose, locale) !== prose) errors.push('nonstandard displayed date');
  return errors;
}

export function auditDesign(dist = resolve('dist')) {
  const files = htmlFiles(dist), errors = [];
  if (files.length < 11) errors.push(`incomplete route inventory (${files.length} HTML)`);
  const cache = new Map();
  for (const file of files) {
    const html = readFileSync(file, 'utf8');
    const load = href => {
      const path = href.startsWith('/GDKVM/') ? join(dist, href.slice('/GDKVM/'.length)) : resolve(dirname(file), href);
      if (!path.startsWith(dist + '/') || !existsSync(path)) return '';
      if (!cache.has(path)) cache.set(path, readFileSync(path, 'utf8'));
      return cache.get(path);
    };
    for (const error of inspectHTML(html, load)) errors.push(`${relative(dist, file)}: ${error}`);
  }
  if (errors.length) throw new Error(errors.join('\n'));
  return { html: files.length, routes: files.map(file => relative(dist, file)) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const result = auditDesign();
  console.log(`PASS: shared design stylesheet and consumer declarations on all ${result.html} HTML outputs. Actual consumption is checked in Playwright.`);
}

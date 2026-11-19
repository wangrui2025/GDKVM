import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatCalendarDate, formatCalendarText } from '../src/utils/calendar-date.mjs';
import { inspectHTML } from './audit-design.mjs';

test('calendar display is year-first and preserves source precision', () => {
  assert.equal(formatCalendarDate('2026-06-04', 'zh'), '2026年6月4日');
  assert.equal(formatCalendarDate('2026-06-04', 'en'), '2026-06-04');
  assert.equal(formatCalendarDate('2025-10', 'zh'), '2025年10月');
  assert.equal(formatCalendarDate('2025', 'zh'), '2025');
  assert.equal(formatCalendarDate('2024-02-29', 'zh'), '2024年2月29日');
  assert.equal(formatCalendarText('Updated 2026-06-04.', 'zh'), 'Updated 2026年6月4日.');
  assert.equal(formatCalendarText('Updated December 15, 2025', 'en'), 'Updated 2025-12-15');
  assert.equal(formatCalendarText('更新：2025 年 12 月 15 日', 'zh'), '更新：2025年12月15日');
  assert.equal(formatCalendarText('September 2025', 'zh'), '2025年9月');
  assert.equal(formatCalendarText('15 December 2025', 'en'), '2025-12-15');
});
test('invalid dates, machine labels and timestamps are never repaired', () => {
  for (const value of ['2026-02-29', '2026-13-10', '2026-02-31', '2026-00', '2026-06-04T12:00:00Z']) {
    assert.equal(formatCalendarDate(value, 'zh'), value);
  }
  for (const value of ['run-2026-06-04', '/2026-06-04/', '2026-06-04.json', 'v2026-06', '2026-06-04T12:00:00Z', 'December 15, 2025.json', '15 December 2025.csv', 'October 2025.txt', '31 April 2025', 'February 29, 2025']) {
    assert.equal(formatCalendarText(value, 'zh'), value);
  }
  assert.equal(formatCalendarText('发布于 2026-06-04，版本 (2025-10)。', 'zh'), '发布于 2026年6月4日，版本 (2025年10月)。');
});
test('new standalone HTML cannot silently omit the shared design', () => {
  assert.ok(inspectHTML('<html><body>A page</body></html>').length > 0);
  const tokens = ['design-system:gdkvm','font-body:serif','size-root:16px','leading-body:1.65','space-unit:4px','width-content:56rem'].map(value => '--' + value).join(';');
  const fixture = `<body data-design-system="gdkvm"><style>:root{${tokens}}</style></body>`;
  assert.ok(inspectHTML(fixture).includes('missing body font consumer'));
  assert.ok(inspectHTML(fixture).includes('missing spacing consumer'));
});

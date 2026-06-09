import { t, type Locale } from '../i18n';

const SUPPORTED_LOCALES: Locale[] = ['en', 'zh'];
function getLocale(): Locale {
  const lang = document.documentElement.lang as Locale;
  return SUPPORTED_LOCALES.includes(lang) ? lang : 'en';
}

export function calculateBsLr(modelSize: number, trainingTokens: number) {
  const logBs = Math.log(0.58) + 0.571 * Math.log(trainingTokens);
  const logLr = Math.log(1.79) - 0.713 * Math.log(modelSize) + 0.307 * Math.log(trainingTokens);
  return {
    batchSize: Math.exp(logBs),
    learningRate: Math.exp(logLr)
  };
}

export function formatLargeNumber(num: number) {
  if (num >= 1e6) {
    return `${(num / 1e6).toFixed(1)}M`;
  }
  return num.toLocaleString();
}

export function formatSmallNumber(num: number) {
  if (num < 0.001) {
    return num.toExponential(2).replace('e-', '×10⁻');
  }
  return num.toFixed(6).replace(/\.?0+$/, '');
}

export function showError(message: string) {
  const resultDiv = document.getElementById('result');
  if (!resultDiv) return;
  const lang = getLocale();
  // Use DOM API instead of innerHTML to avoid XSS from any future i18n
  // strings that may contain HTML metacharacters. See audit finding
  // "src/scripts/tool.ts:38 innerHTML XSS pattern" (low severity).
  while (resultDiv.firstChild) resultDiv.removeChild(resultDiv.firstChild);
  const err = document.createElement('div');
  err.className = 'error';
  err.textContent = `⚠️ ${message}`;
  resultDiv.appendChild(err);
  setTimeout(() => {
    while (resultDiv.firstChild) resultDiv.removeChild(resultDiv.firstChild);
    const h3 = document.createElement('h3');
    h3.textContent = t(lang, 'tool.results');
    const pBs = document.createElement('p');
    pBs.id = 'bsValue';
    pBs.textContent = t(lang, 'tool.bsDefault');
    const pLr = document.createElement('p');
    pLr.id = 'lrValue';
    pLr.textContent = t(lang, 'tool.lrDefault');
    resultDiv.appendChild(h3);
    resultDiv.appendChild(pBs);
    resultDiv.appendChild(pLr);
  }, 2000);
}

export function initToolPage() {
  const lang = getLocale();
  const modelForm = document.getElementById('modelForm');
  if (!modelForm) return;
  modelForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const modelSizeInputEl = document.getElementById('modelSize') as HTMLInputElement;
    const trainingTokensInputEl = document.getElementById('trainingTokens') as HTMLInputElement;
    if (!modelSizeInputEl || !trainingTokensInputEl) return;
    const modelSizeInput = modelSizeInputEl.value;
    const trainingTokensInput = trainingTokensInputEl.value;
    const preprocessInput = (str: string) => {
      return str.replace(/,/g, '').replace(/×10\^?/g, 'e');
    };
    const parseNumber = (str: string) => {
      const cleaned = preprocessInput(str);
      return Number(cleaned);
    };
    const isValidFormat = (str: string) => {
      return /^[+-]?\d*\.?\d+([eE][+-]?\d+)?$/.test(preprocessInput(str));
    };
    if (!isValidFormat(modelSizeInput) || !isValidFormat(trainingTokensInput)) {
      showError(t(lang, 'tool.formatError'));
      return;
    }
    const modelSize = parseNumber(modelSizeInput);
    const trainingTokens = parseNumber(trainingTokensInput);
    if (isNaN(modelSize) || isNaN(trainingTokens) || modelSize <= 0 || trainingTokens <= 0) {
      showError(t(lang, 'tool.positiveNumberError'));
      return;
    }
    const { batchSize, learningRate } = calculateBsLr(modelSize, trainingTokens);
    const bsValue = document.getElementById('bsValue');
    const lrValue = document.getElementById('lrValue');
    if (bsValue) bsValue.textContent = `${t(lang, 'tool.bsLabel')}${formatLargeNumber(batchSize)}`;
    if (lrValue) lrValue.textContent = `${t(lang, 'tool.lrLabel')}${formatSmallNumber(learningRate)}`;
  });
}

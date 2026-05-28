// i18n helper
const i18n = {
  en: {
    formatError: 'Format error, Supporting examples: 1e8, 250000, 3.5×10^6',
    positiveNumberError: 'Please Input Positive Numbers',
    optimalBs: 'Optimal Token Wise BatchSize: ',
    learningRate: 'Learning Rate: ',
    results: 'Results',
    bsDefault: 'BS: -',
    lrDefault: 'LR: -',
  },
  zh: {
    formatError: '格式错误，支持示例：1e8, 250000, 3.5×10^6',
    positiveNumberError: '请输入正数',
    optimalBs: '最优Token级BatchSize: ',
    learningRate: '学习率: ',
    results: '计算结果',
    bsDefault: '批量大小: -',
    lrDefault: '学习率: -',
  },
};

export function t(key: string): string {
  const lang = document.documentElement.lang || 'en';
  return (i18n[lang as keyof typeof i18n] || i18n.en)[key as keyof typeof i18n.en] || key;
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
  resultDiv.innerHTML = `<div class="error">⚠️ ${message}</div>`;
  setTimeout(() => {
    resultDiv.innerHTML = `<h3>${t('results')}</h3>
      <p id="bsValue">${t('bsDefault')}</p>
      <p id="lrValue">${t('lrDefault')}</p>`;
  }, 2000);
}

export function initToolPage() {
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
      showError(t('formatError'));
      return;
    }
    const modelSize = parseNumber(modelSizeInput);
    const trainingTokens = parseNumber(trainingTokensInput);
    if (isNaN(modelSize) || isNaN(trainingTokens) || modelSize <= 0 || trainingTokens <= 0) {
      showError(t('positiveNumberError'));
      return;
    }
    const { batchSize, learningRate } = calculateBsLr(modelSize, trainingTokens);
    const bsValue = document.getElementById('bsValue');
    const lrValue = document.getElementById('lrValue');
    if (bsValue) bsValue.textContent = `${t('optimalBs')}${formatLargeNumber(batchSize)}`;
    if (lrValue) lrValue.textContent = `${t('learningRate')}${formatSmallNumber(learningRate)}`;
  });
}

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
    failed: 'Failed',
    checkParams: 'Please check the parameters：',
    loading: 'Loading...',
  },
  zh: {
    formatError: '格式错误，支持示例：1e8, 250000, 3.5×10^6',
    positiveNumberError: '请输入正数',
    optimalBs: '最优Token级BatchSize: ',
    learningRate: '学习率: ',
    results: '计算结果',
    bsDefault: '批量大小: -',
    lrDefault: '学习率: -',
    failed: '加载失败',
    checkParams: '请检查参数：',
    loading: '加载中...',
  },
};

export function t(key: string): string {
  const lang = document.documentElement.lang || 'en';
  return (i18n[lang as keyof typeof i18n] || i18n.en)[key as keyof typeof i18n.en] || key;
}

export function initTabs() {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => {
      const tabId = button.getAttribute('data-tab');
      document.querySelectorAll('.tab-button').forEach((btn) => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((content) => content.classList.remove('active'));
      button.classList.add('active');
      const pane = document.getElementById(tabId || '');
      if (pane) pane.classList.add('active');
    });
  });
}

const dependencies = {
  Dense: {
    N: {
      "214663680": ["4000000000", "11400000000", "20000000000", "100000000000"],
      "268304384": ["5000000000", "14200000000", "25000000000", "80000000000"],
      "429260800": ["8000000000", "22700000000", "40000000000", "50000000000"],
      "536872960": ["10000000000", "28400000000", "50000000000"],
      "1073741824": ["20000000000", "56900000000", "100000000000"]
    }
  },
  Moe: {
    N: {
      "2150612992": {
        Na: {
          "187973632": ["2000000000", "4000000000", "8000000000", "20000000000"],
          "232579072": ["2000000000", "4000000000", "8000000000", "20000000000"]
        }
      },
      "2155174912": {
        Na: {
          "590436352": ["2000000000", "4000000000", "8000000000", "20000000000"]
        }
      },
      "2156188672": {
        Na: {
          "1241270272": ["2000000000", "4000000000", "8000000000", "20000000000"]
        }
      }
    }
  }
};

export function updateNOptions(modelType: HTMLSelectElement, nValue: HTMLSelectElement, naValue: HTMLSelectElement, dValue: HTMLSelectElement) {
  const type = modelType.value;
  nValue.innerHTML = '';
  naValue.innerHTML = '';
  dValue.innerHTML = '';
  const deps = type === 'Dense' ? dependencies.Dense : dependencies.Moe;
  const options = Object.keys(deps.N);
  options.forEach((val) => {
    const option = document.createElement('option');
    option.value = val;
    option.textContent = val;
    nValue.appendChild(option);
  });
  nValue.value = options[0] || '';
  nValue.dispatchEvent(new Event('change'));
}

export function updateNaOptions(modelType: HTMLSelectElement, nValue: HTMLSelectElement, naValue: HTMLSelectElement, selectorGroup: HTMLElement) {
  const type = modelType.value;
  const nVal = nValue.value;
  const naItem = document.getElementById('naItem');
  naValue.innerHTML = '';
  if (type === 'Moe') {
    naItem?.classList.remove('hidden');
    selectorGroup.classList.add('has-na');
    const moeNode = (dependencies.Moe.N as any)[nVal];
    const naOptions = moeNode ? Object.keys(moeNode.Na || {}) : [];
    naOptions.forEach((val: string) => {
      const option = document.createElement('option');
      option.value = val;
      option.textContent = val;
      naValue.appendChild(option);
    });
    naValue.value = naOptions[0] || '';
    naValue.dispatchEvent(new Event('change'));
  } else {
    naItem?.classList.add('hidden');
    selectorGroup.classList.remove('has-na');
    naValue.value = '';
  }
}

export function updateDOptions(modelType: HTMLSelectElement, nValue: HTMLSelectElement, naValue: HTMLSelectElement, dValue: HTMLSelectElement) {
  const type = modelType.value;
  dValue.innerHTML = '';
  let options: string[] = [];
  if (type === 'Dense') {
    options = (dependencies.Dense.N as any)[nValue.value] || [];
  } else {
    const moeNode = (dependencies.Moe.N as any)[nValue.value];
    options = moeNode && moeNode.Na ? (moeNode.Na[naValue.value] || []) : [];
  }
  options.forEach((val) => {
    const option = document.createElement('option');
    option.value = val;
    option.textContent = val;
    dValue.appendChild(option);
  });
}

export function initDependentSelects() {
  const modelType = document.getElementById('modelType') as HTMLSelectElement;
  const nValue = document.getElementById('nValue') as HTMLSelectElement;
  const naValue = document.getElementById('naValue') as HTMLSelectElement;
  const dValue = document.getElementById('dValue') as HTMLSelectElement;
  const selectorGroup = document.getElementById('selectorGroup');
  if (!modelType || !nValue || !naValue || !dValue || !selectorGroup) return;
  modelType.addEventListener('change', () => {
    updateNOptions(modelType, nValue, naValue, dValue);
  });
  nValue.addEventListener('change', () => {
    modelType.value === 'Moe'
      ? updateNaOptions(modelType, nValue, naValue, selectorGroup)
      : updateDOptions(modelType, nValue, naValue, dValue);
  });
  naValue.addEventListener('change', () => {
    updateDOptions(modelType, nValue, naValue, dValue);
  });
  updateNOptions(modelType, nValue, naValue, dValue);
}

export function initVisualization() {
  const generateBtn = document.getElementById('generateBtn');
  const vizContainer = document.getElementById('visualization');
  if (!generateBtn || !vizContainer) return;
  generateBtn.addEventListener('click', () => {
    const modelTypeEl = document.getElementById('modelType') as HTMLSelectElement;
    const nValueEl = document.getElementById('nValue') as HTMLSelectElement;
    const dValueEl = document.getElementById('dValue') as HTMLSelectElement;
    if (!modelTypeEl || !nValueEl || !dValueEl) return;
    const modelType = modelTypeEl.value;
    const nValue = nValueEl.value;
    const dValue = dValueEl.value;
    vizContainer.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'generated-image-container';
    let fileName: string;
    if (modelType === 'Dense') {
      fileName = `heatmap_N${nValue}_D${dValue}.png`;
    } else {
      const naValueEl = document.getElementById('naValue') as HTMLSelectElement;
      const naValue = naValueEl?.value || '';
      fileName = `heatmap_N${nValue}_D${dValue}_Na${naValue}.png`;
    }
    const imagePath = `./static/images/${fileName}`;
    const img = document.createElement('img');
    img.className = 'generated-image';
    img.src = imagePath;
    img.alt = `Parameters: ${fileName}`;
    img.style.maxWidth = '100%';
    img.style.height = 'auto';
    img.style.borderRadius = '8px';
    img.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
    const errorMsg = document.createElement('div');
    errorMsg.className = 'image-error';
    errorMsg.style.display = 'none';
    errorMsg.innerHTML = `
      <div class="error-message">
        <span class="icon">⚠️</span>
        <div>
          <p>${t('failed')}</p>
          <small>${t('checkParams')}${fileName}</small>
        </div>
      </div>
    `;
    const loading = document.createElement('div');
    loading.className = 'image-loading';
    loading.innerHTML = `
      <div class="loading-spinner"></div>
      <p>${t('loading')}</p>
    `;
    container.style.position = 'relative';
    container.style.minHeight = '500px';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.justifyContent = 'center';
    img.onload = () => {
      loading.style.display = 'none';
      errorMsg.style.display = 'none';
      container.style.minHeight = 'auto';
    };
    img.onerror = () => {
      loading.style.display = 'none';
      errorMsg.style.display = 'flex';
      container.style.minHeight = '500px';
    };
    container.appendChild(loading);
    container.appendChild(img);
    container.appendChild(errorMsg);
    vizContainer.appendChild(container);
  });
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
  initTabs();
  initDependentSelects();
  initVisualization();
  if (!document.querySelector('.tab-button.active')) {
    document.querySelector('.tab-button')?.classList.add('active');
    document.querySelector('.tab-pane')?.classList.add('active');
  }
  const modelForm = document.getElementById('modelForm');
  if (modelForm) {
    modelForm.addEventListener('submit', function (e) {
      e.preventDefault();
      document.querySelector('.tab-button')?.classList.add('active');
      document.querySelector('.tab-pane')?.classList.add('active');
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
}

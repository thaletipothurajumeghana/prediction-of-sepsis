/* SepsisAI — Frontend Application */

const API = '';   // same origin; change to 'http://localhost:5000' for dev
let state = {
  sessionId:    null,
  extracted:    {},
  allFeatures:  {},
  completeness: null,
  userValues:   {},
  predResult:   null,
};

// ═══════════════════════════════
// INIT
// ═══════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  checkAPIStatus();
  setupDragDrop();
  document.getElementById('file-input').addEventListener('change', onFileSelect);
});

async function checkAPIStatus() {
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    const dot  = document.getElementById('api-status-dot');
    const text = document.getElementById('api-status-text');
    if (d.status === 'ok') {
      dot.classList.add('online');
      text.textContent = d.model?.loaded
        ? `Model ready · ${d.model.model_name || 'Ensemble'}`
        : 'Server online · No model';
    } else {
      dot.classList.add('offline');
      text.textContent = 'Server error';
    }
  } catch {
    document.getElementById('api-status-dot').classList.add('offline');
    document.getElementById('api-status-text').textContent = 'Offline';
  }
}

// ═══════════════════════════════
// NAVIGATION
// ═══════════════════════════════
function scrollToApp() {
  document.getElementById('app-section').scrollIntoView({ behavior: 'smooth' });
}

function showManualEntry() {
  scrollToApp();
  // Build form with all features, nothing pre-extracted
  state.sessionId   = null;
  state.extracted   = {};
  state.userValues  = {};
  buildReviewForm({}, {
    total_features: 30,
    found_count: 0,
    completeness_pct: 0,
    has_enough: false,
    critical_missing: ['HR','WBC','Lactate','Creatinine','Platelets','Temp','Resp','SBP'],
    important_missing: ['HR','WBC','Lactate','Creatinine','Platelets','Temp','Resp','SBP'],
    optional_missing: [],
  }, {});
  showPanel('review');
  setStep(2);
}

function goBack() {
  showPanel('upload');
  setStep(1);
}

function startOver() {
  state = { sessionId:null, extracted:{}, allFeatures:{}, completeness:null, userValues:{}, predResult:null };
  document.getElementById('upload-file-name').hidden = true;
  document.getElementById('upload-actions').hidden   = true;
  document.getElementById('file-input').value = '';
  showPanel('upload');
  setStep(1);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showPanel(name) {
  ['upload','review','result'].forEach(p => {
    document.getElementById(`panel-${p}`).classList.toggle('active', p === name);
  });
}

function setStep(n) {
  [1,2,3].forEach(i => {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove('active','done');
    if (i < n)  el.classList.add('done');
    if (i === n) el.classList.add('active');
  });
}

// ═══════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════
function setupDragDrop() {
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type === 'application/pdf') setFile(f);
  });
  zone.addEventListener('click', () => document.getElementById('file-input').click());
}

function onFileSelect(e) {
  const f = e.target.files[0];
  if (f) setFile(f);
}

function setFile(file) {
  const badge = document.getElementById('upload-file-name');
  badge.textContent = `📄 ${file.name}`;
  badge.hidden = false;

  const actions = document.getElementById('upload-actions');
  actions.hidden = false;
  actions.dataset.file = file.name;

  // Store file reference
  window._selectedFile = file;
}

async function analyzeUpload() {
  if (!window._selectedFile) return;

  const MSGS = [
    'Reading PDF content...',
    'Extracting lab values...',
    'Identifying biomarkers...',
    'Assessing completeness...',
  ];

  showLoading(MSGS);

  const formData = new FormData();
  formData.append('file', window._selectedFile);

  try {
    const r = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
    const d = await r.json();
    hideLoading();

    if (!d.success) {
      showError(d.error || 'Upload failed');
      return;
    }

    state.sessionId   = d.session_id;
    state.extracted   = d.extracted    || {};
    state.allFeatures = d.all_features || {};
    state.completeness= d.completeness;

    buildReviewForm(d.extracted, d.completeness, d.all_features);
    showPanel('review');
    setStep(2);

  } catch (err) {
    hideLoading();
    showError('Could not connect to server. Is Flask running?');
  }
}

// ═══════════════════════════════
// REVIEW FORM
// ═══════════════════════════════
const FEATURE_GROUPS = {
  vitals:      { label: 'Vital Signs',         color: '#ef4444' },
  critical:    { label: 'Critical Lab Values', color: '#f59e0b' },
  cbc:         { label: 'Complete Blood Count',color: '#22c55e' },
  abg:         { label: 'ABG / Blood Gas',     color: '#3b82f6' },
  metabolic:   { label: 'Metabolic Panel',     color: '#8b5cf6' },
  liver:       { label: 'Liver Function',      color: '#ec4899' },
  coag:        { label: 'Coagulation',         color: '#14b8a6' },
  cardiac:     { label: 'Cardiac Markers',     color: '#f97316' },
  demographic: { label: 'Patient Demographics',color: '#6b7280' },
};

const ALL_FEATURES_FALLBACK = {
  HR:           {label:'Heart Rate',          unit:'bpm',      normal:[60,100],    group:'vitals'},
  O2Sat:        {label:'Oxygen Saturation',   unit:'%',        normal:[95,100],    group:'vitals'},
  Temp:         {label:'Temperature',         unit:'°C',       normal:[36.1,37.2], group:'vitals'},
  SBP:          {label:'Systolic BP',         unit:'mmHg',     normal:[90,140],    group:'vitals'},
  MAP:          {label:'Mean Arterial Pressure',unit:'mmHg',   normal:[70,100],    group:'vitals'},
  DBP:          {label:'Diastolic BP',        unit:'mmHg',     normal:[60,90],     group:'vitals'},
  Resp:         {label:'Respiratory Rate',    unit:'/min',     normal:[12,20],     group:'vitals'},
  WBC:          {label:'White Blood Cells',   unit:'x10³/µL', normal:[4.5,11],    group:'critical'},
  Lactate:      {label:'Lactate',             unit:'mmol/L',   normal:[0.5,2],     group:'critical'},
  Creatinine:   {label:'Creatinine',          unit:'mg/dL',    normal:[0.6,1.2],   group:'critical'},
  Platelets:    {label:'Platelet Count',      unit:'x10³/µL', normal:[150,400],   group:'critical'},
  Bilirubin_total:{label:'Total Bilirubin',   unit:'mg/dL',    normal:[0.2,1.2],   group:'critical'},
  Hgb:          {label:'Hemoglobin',          unit:'g/dL',     normal:[12,17.5],   group:'cbc'},
  Hct:          {label:'Hematocrit',          unit:'%',        normal:[37,52],     group:'cbc'},
  pH:           {label:'Blood pH',            unit:'',         normal:[7.35,7.45], group:'abg'},
  PaCO2:        {label:'PaCO2',               unit:'mmHg',     normal:[35,45],     group:'abg'},
  HCO3:         {label:'Bicarbonate (HCO3)',  unit:'mmol/L',   normal:[22,26],     group:'abg'},
  BaseExcess:   {label:'Base Excess',         unit:'mmol/L',   normal:[-2,2],      group:'abg'},
  FiO2:         {label:'FiO2',               unit:'fraction', normal:[0.21,0.40],  group:'abg'},
  Glucose:      {label:'Glucose',             unit:'mg/dL',    normal:[70,110],    group:'metabolic'},
  BUN:          {label:'Blood Urea Nitrogen', unit:'mg/dL',    normal:[7,20],      group:'metabolic'},
  Sodium:       {label:'Sodium',              unit:'mEq/L',    normal:[136,145],   group:'metabolic'},
  Potassium:    {label:'Potassium',           unit:'mEq/L',    normal:[3.5,5],     group:'metabolic'},
  Chloride:     {label:'Chloride',            unit:'mEq/L',    normal:[98,107],    group:'metabolic'},
  Calcium:      {label:'Calcium',             unit:'mg/dL',    normal:[8.5,10.5],  group:'metabolic'},
  AST:          {label:'AST (Liver)',         unit:'U/L',      normal:[10,40],     group:'liver'},
  Alkalinephos: {label:'Alkaline Phosphatase',unit:'U/L',      normal:[44,147],    group:'liver'},
  Bilirubin_direct:{label:'Direct Bilirubin', unit:'mg/dL',   normal:[0,0.3],     group:'liver'},
  PTT:          {label:'PTT (Clotting)',      unit:'sec',      normal:[25,35],     group:'coag'},
  Fibrinogen:   {label:'Fibrinogen',          unit:'mg/dL',    normal:[200,400],   group:'coag'},
  TroponinI:    {label:'Troponin I',          unit:'ng/mL',    normal:[0,0.04],    group:'cardiac'},
  Magnesium:    {label:'Magnesium',           unit:'mEq/L',    normal:[1.7,2.2],   group:'metabolic'},
  Phosphate:    {label:'Phosphate',           unit:'mg/dL',    normal:[2.5,4.5],   group:'metabolic'},
  Age:          {label:'Age',                 unit:'years',    normal:[18,100],    group:'demographic'},
  Gender:       {label:'Gender (0=F, 1=M)',   unit:'',         normal:[0,1],       group:'demographic'},
  ICULOS:       {label:'ICU Length of Stay',  unit:'hours',    normal:[0,720],     group:'demographic'},
};

const CRITICAL_FEATURES = ['HR','WBC','Lactate','Creatinine','Platelets','Temp','Resp','SBP'];

function buildReviewForm(extracted, completeness, allFeatures) {
  const features = Object.keys(allFeatures).length ? allFeatures : ALL_FEATURES_FALLBACK;

  // Update completeness UI
  const pct   = completeness.completeness_pct || 0;
  const fill  = document.getElementById('completeness-fill');
  const pctEl = document.getElementById('completeness-pct');
  pctEl.textContent = `${pct}%`;
  fill.style.width  = `${pct}%`;
  fill.className    = 'completeness-fill' + (pct < 30 ? ' low' : pct < 60 ? ' warn' : '');

  const note = document.getElementById('completeness-note');
  note.textContent = `${completeness.found_count || 0} of ${completeness.total_features || 36} fields extracted from PDF`;

  const subtitle = document.getElementById('review-subtitle');
  if (Object.keys(extracted).length === 0) {
    subtitle.textContent = 'No values could be extracted. Please enter them manually below.';
  } else {
    subtitle.textContent = `${completeness.found_count} values extracted. Fill in any missing fields for a more accurate prediction.`;
  }

  // Warning
  const warn = document.getElementById('data-warning');
  const warnText = document.getElementById('data-warning-text');
  if (!completeness.has_enough) {
    warn.hidden = false;
    const missing = completeness.critical_missing || CRITICAL_FEATURES;
    warnText.textContent = `Not enough critical data yet. Please fill in at least 5 of: ${missing.join(', ')}.`;
  } else {
    warn.hidden = true;
  }

  // Group features
  const groups = {};
  for (const [key, meta] of Object.entries(features)) {
    const g = meta.group || 'other';
    if (!groups[g]) groups[g] = [];
    groups[g].push([key, meta]);
  }

  const formEl = document.getElementById('values-form');
  formEl.innerHTML = '';

  for (const [groupKey, groupItems] of Object.entries(groups)) {
    if (!groupItems.length) continue;
    const gMeta = FEATURE_GROUPS[groupKey] || { label: groupKey, color: '#6b7280' };

    const section = document.createElement('div');
    section.className = 'form-section';
    section.innerHTML = `
      <div class="form-section-title">
        <span class="section-badge" style="background:${gMeta.color}"></span>
        ${gMeta.label}
      </div>
      <div class="form-grid" id="grid-${groupKey}"></div>
    `;
    formEl.appendChild(section);

    const grid = section.querySelector(`#grid-${groupKey}`);

    for (const [key, meta] of groupItems) {
      const isExtracted = key in extracted;
      const isCritical  = CRITICAL_FEATURES.includes(key);
      const val         = isExtracted ? extracted[key] : '';
      const [lo, hi]    = meta.normal || [0, 999];
      const isAbnormal  = isExtracted && val !== '' && (parseFloat(val) < lo || parseFloat(val) > hi);

      let statusText  = '';
      let statusClass = '';
      let inputClass  = '';

      if (isExtracted && isAbnormal) {
        statusText  = `⚠ Abnormal (${lo}–${hi} ${meta.unit})`;
        statusClass = 'status-abnormal';
        inputClass  = 'extracted abnormal';
      } else if (isExtracted) {
        statusText  = '✓ Extracted from PDF';
        statusClass = 'status-extracted';
        inputClass  = 'extracted';
      } else if (isCritical) {
        statusText  = '★ Critical — please enter';
        statusClass = 'status-critical';
        inputClass  = 'critical-missing';
      } else {
        statusText  = 'Optional';
        statusClass = 'status-missing';
        inputClass  = '';
      }

      const field = document.createElement('div');
      field.className = 'field-wrap';
      field.innerHTML = `
        <label class="field-label" for="field-${key}">
          <span>${meta.label}${isCritical ? ' <span style="color:var(--red)">*</span>' : ''}</span>
          <span class="unit">${meta.unit}</span>
        </label>
        <input
          type="number" step="any"
          id="field-${key}"
          class="field-input ${inputClass}"
          placeholder="${lo}–${hi}"
          value="${val !== '' ? parseFloat(val).toFixed(meta.unit === '' ? 2 : 1) : ''}"
          data-feature="${key}"
          oninput="onFieldInput(this)"
        >
        <span class="field-status ${statusClass}">${statusText}</span>
      `;
      grid.appendChild(field);
    }
  }
}

function onFieldInput(input) {
  const key = input.dataset.feature;
  const val = input.value.trim();
  if (val === '') {
    delete state.userValues[key];
    return;
  }
  state.userValues[key] = parseFloat(val);

  // Live completeness update
  const criticalEntered = CRITICAL_FEATURES.filter(f => {
    const el = document.getElementById(`field-${f}`);
    return el && el.value.trim() !== '';
  });

  if (criticalEntered.length >= 5) {
    document.getElementById('data-warning').hidden = true;
    document.getElementById('predict-btn').disabled = false;
  }

  // Update field status
  const features = Object.keys(state.allFeatures).length ? state.allFeatures : ALL_FEATURES_FALLBACK;
  if (features[key]) {
    const [lo, hi] = features[key].normal;
    const v = parseFloat(val);
    const statusEl = input.nextElementSibling;
    if (v < lo || v > hi) {
      statusEl.className  = 'field-status status-abnormal';
      statusEl.textContent = `⚠ Abnormal (${lo}–${hi})`;
      input.classList.add('abnormal');
    } else {
      statusEl.className  = 'field-status status-extracted';
      statusEl.textContent = '✓ Normal range';
      input.classList.remove('abnormal');
    }
  }
}

// ═══════════════════════════════
// PREDICTION
// ═══════════════════════════════
async function runPrediction() {
  // Collect all field values
  const allValues = { ...state.extracted };
  document.querySelectorAll('.field-input').forEach(input => {
    if (input.value.trim() !== '') {
      allValues[input.dataset.feature] = parseFloat(input.value);
    }
  });

  const MSGS = [
    'Running ML inference...',
    'Computing risk score...',
    'Analyzing biomarkers...',
    'Generating assessment...',
  ];
  showLoading(MSGS);

  try {
    const r = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values: allValues, session_id: state.sessionId }),
    });
    const d = await r.json();
    hideLoading();

    if (!d.success) {
      if (d.error === 'not_enough_data') {
        const warn = document.getElementById('data-warning');
        const warnText = document.getElementById('data-warning-text');
        warn.hidden = false;
        warnText.textContent = d.message;
        warn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      showError(d.error || 'Prediction failed');
      return;
    }

    state.predResult = d;
    renderResults(d);
    showPanel('result');
    setStep(3);

  } catch (err) {
    hideLoading();
    showError('Could not connect to server.');
  }
}

// ═══════════════════════════════
// RENDER RESULTS
// ═══════════════════════════════
function renderResults(data) {
  const pred  = data.prediction;
  const flags = data.abnormal_flags || [];
  const recs  = data.recommendations || [];
  const comp  = data.completeness;
  const vals  = data.values_used || {};

  const prob      = pred.probability;
  const riskLevel = pred.risk_level.toLowerCase().replace(' ', '-');
  const isSepsis  = pred.sepsis_detected;

  const verdictMessages = {
    'low':       ['No Sepsis Detected',               'Blood markers are within acceptable ranges. Low probability of sepsis.'],
    'moderate':  ['Moderate Risk — Monitor Closely',  'Some indicators present. Continued monitoring and clinical correlation advised.'],
    'high':      ['High Sepsis Risk',                 'Significant sepsis indicators present. Prompt clinical review is recommended.'],
    'very-high': ['Very High Risk — Immediate Review','Strong sepsis indicators. Immediate clinical evaluation required.'],
    'critical':  ['CRITICAL — Emergency Attention',   'Critical sepsis risk detected. Immediate emergency intervention required.'],
  };

  const [title, desc] = verdictMessages[riskLevel] || verdictMessages['high'];

  // Key metrics — take most relevant available
  const SHOW_METRICS = ['WBC','Lactate','Creatinine','Platelets','HR','Temp','SBP','Resp'];
  const metricsHtml = SHOW_METRICS.map(k => {
    if (!(k in vals)) return '';
    const meta = ALL_FEATURES_FALLBACK[k];
    const v    = vals[k];
    const [lo, hi] = meta.normal;
    const status = v < lo ? 'low' : v > hi ? 'high' : 'normal';
    const statusLabel = v < lo ? '↓ Low' : v > hi ? '↑ High' : '● Normal';
    return `
      <div class="metric-tile">
        <div class="mt-label">${meta.label}</div>
        <div class="mt-val">${parseFloat(v).toFixed(1)}</div>
        <div class="mt-status ${status}">${statusLabel} · ${meta.unit}</div>
      </div>
    `;
  }).filter(Boolean).slice(0, 8).join('');

  // Findings
  const findingsHtml = flags.length
    ? flags.map(f => {
        const dir = f.direction === 'high' ? 'fd-high' : 'fd-medium';
        return `<li class="finding-item">
          <span class="finding-dot ${dir}"></span>
          <span>${f.label}: <strong>${f.value}</strong> ${f.unit} (${f.direction}; normal ${f.normal})</span>
        </li>`;
      }).join('')
    : '<li class="finding-item"><span class="finding-dot fd-low"></span><span>All checked markers within normal reference ranges</span></li>';

  const recsHtml = recs.map((r, i) => `
    <li class="rec-item"><span class="rec-num">${String(i+1).padStart(2,'0')}</span><span>${r}</span></li>
  `).join('');

  // Risk bars
  const riskBarsHtml = (pred.risk_components || []).map(rc => `
    <div class="risk-row">
      <div class="risk-meta">
        <span>${rc.label}</span>
        <span style="color:${rc.level==='critical'?'var(--red)':rc.level==='high'?'#ea580c':rc.level==='medium'?'var(--amber)':'var(--green)'}">${rc.value}%</span>
      </div>
      <div class="risk-track">
        <div class="risk-fill ${rc.level}" style="width:0%" data-w="${rc.value}%"></div>
      </div>
    </div>
  `).join('');

  document.getElementById('result-content').innerHTML = `
    <div class="result-verdict verdict-${riskLevel}">
      <div class="verdict-conf">${prob}%</div>
      <div class="verdict-tag">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor"></span>
        ${pred.risk_level} RISK · ${pred.confidence}% CONFIDENCE · ${pred.model_name || 'Ensemble'}
      </div>
      <h2 class="verdict-title">${title}</h2>
      <p class="verdict-desc">${desc}</p>
    </div>

    <div class="result-grid">
      <div class="result-card">
        <h3>Risk Breakdown</h3>
        ${riskBarsHtml || '<p style="color:var(--text-muted);font-size:13px">Insufficient data for breakdown</p>'}
      </div>
      <div class="result-card">
        <h3>Key Biomarkers</h3>
        <div class="metrics-grid">${metricsHtml || '<p style="color:var(--text-muted);font-size:13px">No key metrics available</p>'}</div>
      </div>
    </div>

    <div class="result-grid">
      <div class="result-card">
        <h3>Abnormal Findings</h3>
        <ul class="findings-list">${findingsHtml}</ul>
      </div>
      <div class="result-card">
        <h3>Clinical Recommendations</h3>
        <ul class="rec-list">${recsHtml}</ul>
      </div>
    </div>

    <div class="result-disclaimer">
      <strong>Clinical Disclaimer:</strong> This AI-assisted assessment is for informational and decision-support purposes only.
      It is not a substitute for professional medical diagnosis. Results must be reviewed and acted upon by a qualified
      healthcare provider. Model: ${pred.model_name} · ROC-AUC: ${pred.model_auc} · Data completeness: ${comp?.completeness_pct || '?'}%
    </div>
  `;

  // Animate risk bars after render
  setTimeout(() => {
    document.querySelectorAll('.risk-fill[data-w]').forEach(el => {
      el.style.width = el.dataset.w;
    });
  }, 100);
}

// ═══════════════════════════════
// LOADING
// ═══════════════════════════════
let _loadingInterval = null;
function showLoading(messages) {
  const overlay = document.getElementById('loading-overlay');
  const msgEl   = document.getElementById('loading-msgs');
  overlay.hidden = false;
  let i = 0;
  msgEl.innerHTML = `<span>${messages[0]}</span>`;
  _loadingInterval = setInterval(() => {
    i = (i + 1) % messages.length;
    msgEl.innerHTML = `<span>${messages[i]}</span>`;
  }, 1800);
}
function hideLoading() {
  clearInterval(_loadingInterval);
  document.getElementById('loading-overlay').hidden = true;
}

// ═══════════════════════════════
// ERROR
// ═══════════════════════════════
function showError(msg) {
  const panel = document.querySelector('.panel.active');
  let errDiv = document.getElementById('global-err');
  if (!errDiv) {
    errDiv = document.createElement('div');
    errDiv.id = 'global-err';
    errDiv.className = 'alert alert-err';
    errDiv.style.marginBottom = '20px';
    panel.insertBefore(errDiv, panel.firstChild);
  }
  errDiv.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> ${msg}`;
  errDiv.hidden = false;
  setTimeout(() => { if (errDiv) errDiv.hidden = true; }, 6000);
}
/* ── State ── */
let selectedTask = 'copying';
let hasFile = false;

/* ── Pipeline ── */
const PIPE_IDS = ['ps0','ps1','ps2','ps3','ps4','ps5'];
function setPipe(active) {
  PIPE_IDS.forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
    if (i < active) el.classList.add('done');
    if (i === active) el.classList.add('active');
  });
}

/* ── File ── */
function onFileChange(input) {
  if (!input.files[0]) return;
  hasFile = true;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('imgPreview');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('uploadHint').style.display = 'none';
  };
  reader.readAsDataURL(input.files[0]);
  setPipe(0);
  document.getElementById('goBtn').disabled = false;
}

/* ── Task ── */
function pickTask(el) {
  document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  selectedTask = el.dataset.task;
}

/* ── Analysis ── */
function runAnalysis() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files[0]) return;

  const btn = document.getElementById('goBtn');
  btn.textContent = 'Analyzing...';
  btn.disabled = true;

  const fd = new FormData();
  fd.append('image', fileInput.files[0]);
  fd.append('task_type', selectedTask);

  // Animate pipeline
  let step = 0;
  const timer = setInterval(() => {
    if (step < PIPE_IDS.length - 1) setPipe(step++);
  }, 380);

  fetch('/predict', { method:'POST', body:fd })
    .then(r => r.json())
    .then(data => {
      clearInterval(timer);
      setPipe(PIPE_IDS.length - 1);
      if (data.error) {
        alert('Error: ' + data.error);
      } else {
        renderResults(data);
        loadHistory();
      }
    })
    .catch(err => {
      clearInterval(timer);
      alert('Request failed: ' + err.message);
    })
    .finally(() => {
      btn.textContent = 'Analyze Cognitive Load';
      btn.disabled = false;
    });
}

/* ── Render results ── */
function renderResults(data) {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('results').style.display = 'block';

  const level = data.load;
  const conf  = data.confidence;

  // Circle
  const circle = document.getElementById('loadCircle');
  circle.textContent = level;
  circle.className = 'load-circle ' + level;

  // Title / sub
  const descs = {
    LOW:    'Writing patterns consistent with low-demand task execution.',
    MEDIUM: 'Moderate graphological variability — brain actively engaged.',
    HIGH:   'High stroke irregularity, tremor, and spacing disruption — heavy cognitive effort.'
  };
  document.getElementById('loadTitle').textContent = level + ' Cognitive Load';
  document.getElementById('loadSub').textContent   = descs[level] || '';

  // Status pill
  const pill = document.getElementById('statusPill');
  pill.textContent = (data.status || '').replace(/_/g,' ');
  pill.className   = 'spill ' + (data.status || '');
  document.getElementById('expLoad').textContent = data.expected_load || '—';

  // Confidence
  const pct = Math.round(conf * 100);
  document.getElementById('confTxt').textContent = pct + '%';
  const fill = document.getElementById('confFill');
  fill.style.background = level === 'HIGH' ? '#E24B4A' : level === 'MEDIUM' ? '#EF9F27' : '#639922';
  setTimeout(() => { fill.style.width = pct + '%'; }, 60);

  // Class probas
  const probRow = document.getElementById('probRow');
  probRow.innerHTML = '';
  if (data.class_probabilities) {
    Object.entries(data.class_probabilities).sort().forEach(([cls, p]) => {
      const d = document.createElement('div');
      d.className = 'prob-item';
      d.innerHTML = `<div class="prob-lbl">${cls}</div><div class="prob-val">${Math.round(p*100)}%</div>`;
      probRow.appendChild(d);
    });
  }

  // Feature groups
  renderFeatureGroups(data.grouped_features || {});

  // Insights
  renderInsights(data.insights || [], level);

  document.getElementById('results').scrollIntoView({ behavior:'smooth', block:'start' });
}

/* ── Feature group cards ── */
const GROUP_META = {
  'Pressure':   { color:'#3266ad', label:'Pressure patterns' },
  'Slant':      { color:'#854F0B', label:'Slant patterns' },
  'Spacing':    { color:'#0F6E56', label:'Spacing patterns' },
  'Form/Shape': { color:'#533AB7', label:'Form & shape patterns' },
};

// Approximate max value per feature for bar scaling
const FEAT_MAX = {
  pressure_mean:3.5, pressure_variance:2, ink_density:1,
  slant_mean_angle:180, slant_deviation:30, slant_skewness:4,
  letter_spacing_mean:20, letter_spacing_var:15, baseline_deviation:15,
  form_regularity:1, tremor_index:2.5, pen_lift_fragmentation:30,
  letter_height_var:10, pixel_entropy:6, zone_ratio:0.5,
};

function renderFeatureGroups(grouped) {
  const container = document.getElementById('featGroups');
  container.innerHTML = '';

  Object.entries(GROUP_META).forEach(([grp, meta]) => {
    const feats = grouped[grp] || {};
    const card = document.createElement('div');
    card.className = 'fg-card';
    card.innerHTML = `
      <div class="fg-title">
        <span class="fg-dot" style="background:${meta.color}"></span>
        ${meta.label}
      </div>
    `;
    Object.entries(feats).forEach(([name, val]) => {
      const max  = FEAT_MAX[name] || 10;
      const pct  = Math.min((Math.abs(val) / max) * 100, 100).toFixed(0);
      const disp = name.replace(/_/g,' ');
      const row  = document.createElement('div');
      row.className = 'feat-row';
      row.innerHTML = `
        <div class="feat-name" title="${disp}">${disp}</div>
        <div class="feat-track">
          <div class="feat-fill" data-pct="${pct}" style="width:0;background:${meta.color};opacity:.75"></div>
        </div>
        <div class="feat-val">${val.toFixed(3)}</div>
      `;
      card.appendChild(row);
    });
    container.appendChild(card);
  });

  // Animate bars
  setTimeout(() => {
    container.querySelectorAll('.feat-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  }, 80);
}

/* ── Insights ── */
function renderInsights(insights, level) {
  const list = document.getElementById('insightsList');
  const dotColor = level === 'HIGH' ? '#E24B4A' : level === 'MEDIUM' ? '#EF9F27' : '#639922';
  if (!insights.length) {
    list.innerHTML = '<li><span class="ins-dot" style="background:#aaa"></span>No significant anomalies detected.</li>';
    return;
  }
  list.innerHTML = insights.map(ins => {
    const c = ins.triggered ? (level === 'HIGH' ? '#E24B4A' : '#EF9F27') : '#639922';
    return `<li>
      <span class="ins-dot" style="background:${c}"></span>
      <span>${ins.text} <em style="font-size:11px;color:#aaa">(${ins.feature.replace(/_/g,' ')} = ${ins.value})</em></span>
    </li>`;
  }).join('');
}

/* ── History ── */
function loadHistory() {
  fetch('/history').then(r => r.json()).then(rows => {
    const area = document.getElementById('histArea');
    if (!rows.length) {
      area.className = 'hist-empty';
      area.innerHTML = 'No predictions yet.';
      return;
    }
    area.className = '';
    const table = document.createElement('table');
    table.className = 'hist-table';
    table.innerHTML = `
      <thead><tr>
        <th>#</th><th>File</th><th>Task</th><th>Load</th>
        <th>Confidence</th><th>Status</th><th>Time</th>
      </tr></thead>
      <tbody>${rows.map(r => `
        <tr>
          <td>${r.id}</td>
          <td>${r.filename}</td>
          <td>${r.task_type.replace('_',' ')}</td>
          <td><span class="hbadge ${r.load_level}">${r.load_level}</span></td>
          <td>${Math.round(r.confidence*100)}%</td>
          <td>${(r.status||'').replace(/_/g,' ')}</td>
          <td>${(r.timestamp||'').slice(0,16).replace('T',' ')}</td>
        </tr>`).join('')}
      </tbody>`;
    area.innerHTML = '';
    area.appendChild(table);
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', loadHistory);

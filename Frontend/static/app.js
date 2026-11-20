/* Heart Disease Prediction System: browser app.
   This file only draws the interface. Every prediction, the metrics, the nearest neighbors and
   the batch results come from the Python server (server.py), which loads the real model files. */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  var INFO = null;   // schema, ranges, defaults, metrics: all from GET /api/info

  /* ---------- API ---------- */
  function api(path, opts) {
    return fetch(path, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (body) {
        if (!res.ok) { var e = new Error(body.error || ('Request failed (' + res.status + ')')); e.details = body.details; throw e; }
        return body;
      });
    });
  }
  function postJSON(path, data) { return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }); }
  function postFile(path, blob, name) { var fd = new FormData(); fd.append('file', blob, name || 'patients.csv'); return api(path, { method: 'POST', body: fd }); }

  /* ---------- Helpers ---------- */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }
  function smooth(a, b, x) { var t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
  function gauss(x, m, s) { var d = (x - m) / s; return Math.exp(-0.5 * d * d); }
  function ordinal(n) { var s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); }
  function pct1(x) { return (Math.round(x * 10) / 10).toFixed(1) + '%'; }
  var SEX = { M: 'Male', F: 'Female' };

  function votesHTML(votes, k, small) {
    var h = '<span class="votes' + (small ? ' sm' : '') + '" role="img" aria-label="' + votes + ' of ' + k + ' similar patients had heart disease">';
    for (var i = 0; i < k; i++) h += '<i class="dot ' + (i < votes ? 'yes' : 'no') + '"></i>';
    return h + '</span>';
  }

  function tween(el, to, ms, fmt) {
    if (!el) return;
    if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(to); return; }
    var t0 = performance.now();
    (function step(t) {
      var f = clamp((t - t0) / ms, 0, 1), e = 1 - Math.pow(1 - f, 3);
      el.textContent = fmt(to * e);
      if (f < 1) requestAnimationFrame(step);
    })(t0);
  }

  /* ---------- ECG trace (an illustration drawn from the inputs) ----------
     8 px = 1 mm on the paper. 25 mm/s gives 200 px per second, 10 mm/mV gives 80 px per mV. */
  var PX_S = 200, PX_MV = 80;

  function beat(p, o) {
    var scale = Math.min(1, o.T / 0.62), u = p / scale;
    var L0 = -o.st * 0.1;
    var sl = o.slope === 'Up' ? 0.12 : o.slope === 'Down' ? -0.12 : 0;
    var v = 0.11 * gauss(u, 0.095, 0.028);
    if (u >= 0.17 && u < 0.285) {
      var pts = [[0.17, 0], [0.20, -0.10], [0.225, 1.15 * o.amp], [0.255, -0.28], [0.285, L0]];
      for (var i = 0; i < pts.length - 1; i++) {
        if (u >= pts[i][0] && u < pts[i + 1][0]) { var f = (u - pts[i][0]) / (pts[i + 1][0] - pts[i][0]); v = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f; break; }
      }
    } else if (u >= 0.285) {
      var lvl = (L0 + sl * clamp((u - 0.285) / 0.115, 0, 1)) * (1 - smooth(0.5, 0.6, u));
      var tAmp = o.flatT ? -0.1 : clamp(0.34 - 0.05 * Math.max(0, o.st), -0.1, 0.34);
      v = lvl + tAmp * gauss(u, 0.44, 0.05);
    }
    return v;
  }

  function ecgPath(o, W, baseY, H) {
    var T = 60 / clamp(o.hr, 30, 240), d = '';
    for (var x = 0; x <= W; x += 1.5) {
      var t = x / PX_S + 0.02, ph = t - Math.floor(t / T) * T;
      var y = clamp(baseY - beat(ph, { T: T, amp: o.amp, slope: o.slope, st: o.st, flatT: o.flatT }) * PX_MV, 6, H - 6);
      d += (x === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
    }
    return d;
  }

  function ecgParams(p) {
    return {
      hr: +p.MaxHR || 150,
      slope: p.ST_Slope || 'Flat',
      st: clamp(+p.Oldpeak || 0, -3, 6),
      amp: p.RestingECG === 'LVH' ? 1.45 : 1,
      flatT: p.RestingECG === 'ST'
    };
  }

  function ecgSVG(p, W, H, id) {
    var baseY = Math.round(H * 0.62 / 8) * 8;
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Illustrative ECG trace at ' + (+p.MaxHR || 150) + ' beats per minute">' +
      '<defs><pattern id="gm' + id + '" width="8" height="8" patternUnits="userSpaceOnUse"><path class="grid-minor" d="M8 0H0V8"/></pattern>' +
      '<pattern id="gM' + id + '" width="40" height="40" patternUnits="userSpaceOnUse"><path class="grid-major" d="M40 0H0V40"/></pattern></defs>' +
      '<rect width="' + W + '" height="' + H + '" fill="url(#gm' + id + ')"/><rect width="' + W + '" height="' + H + '" fill="url(#gM' + id + ')"/>' +
      '<path class="trace" pathLength="1" d="' + ecgPath(ecgParams(p), W, baseY, H) + '"/>' +
      '<text class="scale" x="12" y="' + (H - 10) + '">25 mm/s  10 mm/mV</text></svg>';
  }

  var ecgDrawn = false;
  function updateEcg() {
    var host = $('#ecgLive'); if (!host || !INFO) return;
    var p = readForm(true);
    host.innerHTML = ecgSVG(p, 640, 208, 'L');
    $('#ecgMeta').textContent = Math.round(ecgParams(p).hr) + ' bpm';
    if (!ecgDrawn) {
      ecgDrawn = true;
      var tr = $('.trace', host); tr.classList.add('draw');
      tr.addEventListener('animationend', function () { tr.classList.remove('draw'); }, { once: true });
    }
  }

  /* ---------- Predict form ----------
     Labels live here. Options, ranges and defaults come from the server schema. */
  var OPT_LABEL = {
    Sex: { M: 'Male', F: 'Female' },
    FastingBS: { 0: 'No', 1: 'Yes' },
    ChestPainType: { ATA: 'Atypical angina', NAP: 'Non-anginal pain', ASY: 'Asymptomatic', TA: 'Typical angina' },
    RestingECG: { Normal: 'Normal', ST: 'ST', LVH: 'LVH' },
    ExerciseAngina: { N: 'No', Y: 'Yes' },
    ST_Slope: { Up: 'Up', Flat: 'Flat', Down: 'Down' }
  };
  var FIELDS_A = [
    { k: 'Age', type: 'num', label: 'Age', unit: 'years' },
    { k: 'Sex', type: 'seg', label: 'Sex' },
    { k: 'RestingBP', type: 'num', label: 'Resting blood pressure', unit: 'mm Hg' },
    { k: 'Cholesterol', type: 'num', label: 'Cholesterol', unit: 'mg/dl' },
    { k: 'MaxHR', type: 'num', label: 'Maximum heart rate', unit: 'bpm' },
    { k: 'FastingBS', type: 'seg', label: 'Fasting blood sugar above 120 mg/dl' }
  ];
  var FIELDS_B = [
    { k: 'ChestPainType', type: 'opts', label: 'Chest pain type', wide: true },
    { k: 'RestingECG', type: 'seg', label: 'Resting ECG', hint: 'ST: ST-T wave abnormality. LVH: left ventricular hypertrophy' },
    { k: 'ExerciseAngina', type: 'seg', label: 'Exercise-induced angina' },
    { k: 'Oldpeak', type: 'num', label: 'Oldpeak (ST depression)', unit: 'mm', hint: 'The model was trained on whole numbers, so decimals are dropped' },
    { k: 'ST_Slope', type: 'seg', label: 'ST slope' }
  ];

  var minus = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12h12"/></svg>';
  var plus = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v12M6 12h12"/></svg>';

  function fieldHTML(f) {
    var id = 'f-' + f.k, h = '<div class="field' + (f.wide ? ' wide' : '') + '" data-k="' + f.k + '">';
    if (f.type === 'num') {
      var n = INFO.numeric[f.k];
      h += '<label class="label" for="' + id + '">' + f.label + '</label>' +
        '<div class="num"><input id="' + id + '" name="' + f.k + '" type="number" inputmode="decimal" min="' + n.min + '" max="' + n.max + '" step="' + n.step + '" value="' + INFO.defaults[f.k] + '">' +
        '<span class="unit">' + f.unit + '</span>' +
        '<button type="button" data-d="-1" aria-label="Decrease ' + f.label + '">' + minus + '</button>' +
        '<button type="button" data-d="1" aria-label="Increase ' + f.label + '">' + plus + '</button></div>' +
        '<p class="err">Enter a value from ' + n.min + ' to ' + n.max + '</p>';
    } else {
      var cls = f.type === 'seg' ? 'seg' : 'opts';
      h += '<span class="label" id="' + id + '-l">' + f.label + '</span><div class="' + cls + '" role="radiogroup" aria-labelledby="' + id + '-l">';
      INFO.categories[f.k].forEach(function (val) {
        var chk = String(INFO.defaults[f.k]) === String(val) ? ' checked' : '', text = (OPT_LABEL[f.k] || {})[val] || val;
        if (f.type === 'seg') h += '<label><input type="radio" name="' + f.k + '" value="' + val + '"' + chk + '><span>' + text + '</span></label>';
        else h += '<label class="opt"><input type="radio" name="' + f.k + '" value="' + val + '"' + chk + '><span><b>' + val + '</b><em>' + text + '</em></span></label>';
      });
      h += '</div>';
    }
    if (f.hint) h += '<p class="hint">' + f.hint + '</p>';
    return h + '</div>';
  }

  function readForm(lenient) {
    var f = $('#predictForm'), p = {}, ok = true;
    INFO.input_columns.forEach(function (k) {
      var n = INFO.numeric[k];
      if (n) {
        var el = f.elements[k], raw = el.value.trim(), num = parseFloat(raw);
        var bad = raw === '' || isNaN(num) || num < n.min || num > n.max;
        el.closest('.field').classList.toggle('invalid', bad && !lenient);
        if (bad) ok = false;
        p[k] = isNaN(num) ? INFO.defaults[k] : (lenient ? clamp(num, n.min, n.max) : num);
      } else {
        var val = f.elements[k].value;
        p[k] = k === 'FastingBS' ? +val : val;
      }
    });
    p._valid = ok;
    return p;
  }

  function payload(p) { var o = {}; INFO.input_columns.forEach(function (k) { o[k] = p[k]; }); return o; }

  function neighborsHTML(res) {
    var h = '<ul class="neigh">';
    res.neighbors.forEach(function (n) {
      var t = n.profile, yes = n.label === 1;
      var who = t ? SEX[t.Sex] + ', ' + t.Age : 'Similar patient';
      var meta = t ? '<span class="meta">' + t.ChestPainType + ' · ST ' + t.ST_Slope + ' · BP ' + Math.round(t.RestingBP) + ' · Chol ' + Math.round(t.Cholesterol) + ' · HR ' + t.MaxHR + '</span>' : '';
      h += '<li><span class="votes sm"><i class="dot ' + (yes ? 'yes' : 'no') + '"></i></span>' +
        '<div><span class="who">' + who + '<span class="muted" style="font-weight:400"> · ' + (yes ? 'Heart disease' : 'No heart disease') + '</span></span>' + meta + '</div>' +
        '<span class="d" title="Distance in standardized units">d ' + n.distance.toFixed(2) + '</span></li>';
    });
    return h + '</ul>';
  }

  function compareHTML(list) {
    var h = '<ul class="cmp">';
    list.forEach(function (c) {
      var d = INFO.distributions[c.key]; if (!d) return;
      var mx = Math.max.apply(null, d.bins), x = clamp((c.value - d.lo) / (d.hi - d.lo), 0, 1) * 100;
      h += '<li><span class="k">' + c.label + '</span><div class="strip" aria-hidden="true">' +
        d.bins.map(function (b) { return '<i style="height:' + Math.max(8, b / mx * 100) + '%"></i>'; }).join('') +
        '<span class="mk" style="left:' + x + '%"></span></div>' +
        '<span class="v"><b>' + (c.key === 'Oldpeak' ? (+c.value).toFixed(1) : Math.round(c.value)) + '</b>' + ordinal(clamp(Math.round(c.percentile), 1, 99)) + ' pct</span></li>';
    });
    return h + '</ul>';
  }

  function renderResult(host, res) {
    var risk = res.prediction === 1, conf = res.confidence * 100;
    host.innerHTML =
      '<div class="verdict-row"><div><p class="verdict ' + (risk ? 'risk' : 'ok') + '">' + (risk ? 'Heart disease risk detected' : 'No heart disease detected') + '</p>' +
      '<p class="verdict-sub">' + res.votes + ' of ' + res.k + ' most similar patients had heart disease</p></div>' +
      '<div class="conf"><b data-conf>0%</b><span>Model confidence</span></div></div>' +
      '<div class="vote-line">' + votesHTML(res.votes, res.k) + '<span class="small">Neighbor votes, filled means heart disease</span></div>' +
      '<h3 class="block-title">Five most similar patients</h3>' + neighborsHTML(res) +
      (res.compare && res.compare.length ? '<h3 class="block-title">Where this patient sits among the ' + INFO.dataset.total + ' records</h3>' + compareHTML(res.compare) : '') +
      '<p class="disclaimer">A screening estimate for education. It is not a medical diagnosis.</p>';
    tween($('[data-conf]', host), conf, 600, function (x) { return Math.round(x) + '%'; });
  }

  function renderResultError(host, err) {
    var list = err.details && err.details.length ? '<ul class="small muted" style="margin-top:8px">' + err.details.map(function (d) { return '<li>' + esc(d) + '</li>'; }).join('') + '</ul>' : '';
    host.innerHTML = '<div class="msg err"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.2v.1"/></svg><div>' + esc(err.message) + list + '</div></div>';
  }

  var EMPTY = '<div class="result-empty"><span class="votes votes-empty" aria-hidden="true"><i class="dot"></i><i class="dot"></i><i class="dot"></i><i class="dot"></i><i class="dot"></i></span><p>The result and the five most similar patients appear here</p></div>';

  function buildPredict() {
    $('#fieldsA').innerHTML = FIELDS_A.map(fieldHTML).join('');
    $('#fieldsB').innerHTML = FIELDS_B.map(fieldHTML).join('');
    var form = $('#predictForm'), raf = 0, btn = $('button[type=submit]', form);

    form.addEventListener('input', function () { cancelAnimationFrame(raf); raf = requestAnimationFrame(updateEcg); });
    form.addEventListener('change', function () { updateEcg(); });
    form.addEventListener('click', function (e) {
      var b = e.target.closest('.num button'); if (!b) return;
      var inp = $('input', b.closest('.num')), step = parseFloat(inp.step) || 1, d = +b.dataset.d;
      var cur = parseFloat(inp.value); if (isNaN(cur)) cur = INFO.defaults[inp.name];
      var next = clamp(Math.round((cur + d * step) * 10) / 10, +inp.min, +inp.max);
      inp.value = next; inp.dispatchEvent(new Event('input', { bubbles: true }));
    });
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var p = readForm(false), host = $('#result');
      if (!p._valid) { var bad = $('.field.invalid input', form); if (bad) bad.focus(); return; }
      btn.disabled = true; btn.textContent = 'Analyzing';
      postJSON('/api/predict', payload(p)).then(function (res) {
        renderResult(host, res);
        if (window.matchMedia('(max-width: 1100px)').matches) host.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }).catch(function (err) { renderResultError(host, err); })
        .then(function () { btn.disabled = false; btn.textContent = 'Predict heart disease risk'; });
    });
    $('#resetBtn').addEventListener('click', function () {
      form.reset();
      $$('.field.invalid', form).forEach(function (f) { f.classList.remove('invalid'); });
      updateEcg();
      $('#result').innerHTML = EMPTY;
    });
    updateEcg();
  }

  /* ---------- Home preview: a real prediction from the server for an example patient ---------- */
  function buildHero() {
    var p = { Age: 58, Sex: 'M', ChestPainType: 'ASY', RestingBP: 140, Cholesterol: 240, FastingBS: 0, RestingECG: 'ST', MaxHR: 112, ExerciseAngina: 'Y', Oldpeak: 2, ST_Slope: 'Flat' };
    var host = $('#heroView');
    host.innerHTML = '<div class="ecg">' + ecgSVG(p, 640, 280, 'H') + '</div>' +
      '<div class="hero-meta"><span>Example patient</span><span class="mono">58 · M · ASY · ST Flat · Oldpeak 2 · HR 112</span></div>' +
      '<div class="hero-result" id="heroResult"><p class="sub">Running the model</p></div>';
    var tr = $('#heroView .trace'); tr.classList.add('draw');
    tr.addEventListener('animationend', function () { tr.classList.remove('draw'); }, { once: true });
    postJSON('/api/predict', p).then(function (res) {
      var risk = res.prediction === 1;
      $('#heroResult').innerHTML =
        '<div><p class="verdict" style="color:var(' + (risk ? '--accent' : '--ok') + ')">' + (risk ? 'Heart disease risk detected' : 'No heart disease detected') + '</p>' +
        '<p class="sub">' + res.votes + ' of ' + res.k + ' most similar patients had heart disease</p></div>' +
        '<div class="right">' + votesHTML(res.votes, res.k) + '<div class="conf"><b>' + Math.round(res.confidence * 100) + '%</b><span>Model confidence</span></div></div>';
    }).catch(function () { $('#heroResult').innerHTML = '<p class="sub">The example could not be calculated</p>'; });
  }

  /* ---------- Batch ---------- */
  var batch = { file: null, name: '', result: null, filter: 'all', page: 0 };
  var PAGE = 12;

  function setStatus(kind, html) {
    var icon = kind === 'ok'
      ? '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16 9.5"/></svg>'
      : '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.2v.1"/></svg>';
    $('#batchStatus').innerHTML = html ? '<div class="msg ' + kind + '">' + icon + '<div>' + html + '</div></div>' : '';
  }

  function loadBatch(blob, name) {
    batch.file = null; batch.result = null; $('#batchResults').innerHTML = ''; $('#batchPreview').innerHTML = '';
    setStatus('', 'Checking the file');
    postFile('/api/batch/validate', blob, name).then(function (res) {
      batch.file = blob; batch.name = name;
      setStatus('ok', 'Loaded <b>' + res.rows + '</b> records' + (name ? ' from <code>' + esc(name) + '</code>' : ''));
      renderPreview(res);
    }).catch(function (err) { setStatus('err', esc(err.message)); });
  }

  function renderPreview(res) {
    var cols = INFO.input_columns;
    $('#batchPreview').innerHTML = '<div class="section"><div class="section-head"><h2>Preview</h2><button class="btn primary lg" id="runBatch" type="button">Run batch prediction</button></div>' +
      '<div class="tbl-wrap"><table><thead><tr>' + cols.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead><tbody>' +
      res.preview.map(function (r) { return '<tr>' + cols.map(function (c) { return '<td>' + esc(r[c]) + '</td>'; }).join('') + '</tr>'; }).join('') +
      '</tbody></table></div><p class="small muted preview-note">Showing the first ' + res.preview.length + ' of ' + res.rows + ' rows</p></div>';
    $('#runBatch').addEventListener('click', runBatch);
  }

  function runBatch() {
    var btn = $('#runBatch'); btn.disabled = true; btn.textContent = 'Processing';
    postFile('/api/batch/predict', batch.file, batch.name).then(function (res) {
      batch.result = res; batch.filter = 'all'; batch.page = 0;
      renderBatchResults();
      $('#batchResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }).catch(function (err) { setStatus('err', esc(err.message)); })
      .then(function () { btn.disabled = false; btn.textContent = 'Run batch prediction'; });
  }

  function renderBatchResults() {
    var r = batch.result, s = r.summary, n = s.total, counts = r.votes_histogram, mx = Math.max.apply(null, counts) || 1, k = r.k;
    var h = '<div class="section"><div class="section-head"><h2>Results summary</h2></div>' +
      '<div class="stats">' +
      '<article class="stat"><p class="stat-label">Total patients</p><p class="stat-fig">' + n + '</p><p class="stat-note">records processed</p></article>' +
      '<article class="stat"><p class="stat-label">At risk</p><p class="stat-fig" style="color:var(--accent)">' + s.at_risk + '</p><p class="stat-note">' + pct1(s.at_risk / n * 100) + ' of all patients</p></article>' +
      '<article class="stat"><p class="stat-label">Healthy</p><p class="stat-fig" style="color:var(--ok)">' + s.healthy + '</p><p class="stat-note">' + pct1(s.healthy / n * 100) + ' of all patients</p></article>' +
      '</div></div>';

    var half = Math.floor(k / 2);
    h += '<div class="section"><div class="bar-chart"><div class="section-head" style="margin-bottom:24px"><div><h2>Risk score distribution</h2><p class="chart-sub">Patients by how many of the ' + k + ' nearest neighbors had heart disease</p></div></div>' +
      '<div class="bars" style="grid-template-columns:repeat(' + counts.length + ',minmax(0,1fr))">' + counts.map(function (c, i) {
        return '<div class="bar' + (i > half ? ' hot' : '') + '"><span class="n">' + c + '</span><div class="col" style="height:' + (c / mx * 100 * 0.82) + '%;animation-delay:' + (i * 50) + 'ms"></div></div>';
      }).join('') + '</div>' +
      '<div class="xlabels" style="grid-template-columns:repeat(' + counts.length + ',minmax(0,1fr))">' + counts.map(function (_, i) { return '<span>' + Math.round(i / k * 100) + '%</span>'; }).join('') + '</div>' +
      '<div class="xzones"><span>Healthy: 0 to ' + half + ' votes</span><span>At risk: ' + (half + 1) + ' to ' + k + ' votes</span></div></div></div>';

    h += '<div class="section"><div class="section-head"><h2>Detailed results</h2><div class="filters" role="group" aria-label="Filter results">' +
      [['all', 'All'], ['risk', 'At risk'], ['ok', 'Healthy']].map(function (f) { return '<button type="button" data-f="' + f[0] + '" aria-pressed="' + (batch.filter === f[0]) + '">' + f[1] + '</button>'; }).join('') +
      '</div></div><div id="tableHost"></div>' +
      '<div style="margin-top:16px"><button class="btn secondary" id="dlBtn" type="button"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M4 19h16"/></svg>Download results as CSV</button></div></div>';

    $('#batchResults').innerHTML = h;
    renderTable();
    $$('.filters button').forEach(function (b) {
      b.addEventListener('click', function () {
        batch.filter = b.dataset.f; batch.page = 0;
        $$('.filters button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        renderTable();
      });
    });
    $('#dlBtn').addEventListener('click', download);
  }

  function renderTable() {
    var cols = INFO.input_columns, k = batch.result.k;
    var rows = batch.result.rows.map(function (r, i) { return { r: r, i: i }; }).filter(function (x) {
      var risk = x.r.Prediction === 'Heart Disease';
      return batch.filter === 'all' || (batch.filter === 'risk' ? risk : !risk);
    });
    var pages = Math.max(1, Math.ceil(rows.length / PAGE)); batch.page = clamp(batch.page, 0, pages - 1);
    var slice = rows.slice(batch.page * PAGE, batch.page * PAGE + PAGE);
    var h = '<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Prediction</th><th>Neighbor votes</th><th class="r">Risk score</th>' +
      cols.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead><tbody>' +
      slice.map(function (x) {
        var r = x.r, isRisk = r.Prediction === 'Heart Disease';
        return '<tr><td>' + (x.i + 1) + '</td><td class="sans"><span class="tag ' + (isRisk ? 'risk' : 'ok') + '">' + esc(r.Prediction) + '</span></td>' +
          '<td>' + votesHTML(r.Votes, k, true) + '</td><td class="r">' + pct1(r.Risk_Score) + '</td>' +
          cols.map(function (c) { return '<td>' + esc(r[c]) + '</td>'; }).join('') + '</tr>';
      }).join('') + '</tbody></table></div>' +
      '<div class="pager"><span>' + (rows.length ? (batch.page * PAGE + 1) + ' to ' + Math.min(rows.length, batch.page * PAGE + PAGE) : '0') + ' of ' + rows.length + '</span><div>' +
      '<button class="btn secondary" data-p="-1"' + (batch.page === 0 ? ' disabled' : '') + '>Previous</button>' +
      '<button class="btn secondary" data-p="1"' + (batch.page >= pages - 1 ? ' disabled' : '') + '>Next</button></div></div>';
    var host = $('#tableHost'); host.innerHTML = h;
    $$('.pager button', host).forEach(function (b) { b.addEventListener('click', function () { batch.page += +b.dataset.p; renderTable(); }); });
  }

  function download() {
    var blob = new Blob([batch.result.csv], { type: 'text/csv' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'heart_disease_predictions.csv';
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  function buildBatch() {
    $('#reqCols').innerHTML = INFO.input_columns.map(function (c) { return '<li>' + c + '</li>'; }).join('');
    var input = $('#csvInput'), drop = $('#drop');
    input.addEventListener('change', function () { if (input.files[0]) loadBatch(input.files[0], input.files[0].name); input.value = ''; });
    ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
    ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
    drop.addEventListener('drop', function (e) { var f = e.dataTransfer.files[0]; if (f) loadBatch(f, f.name); });
    $('#sampleBtn').addEventListener('click', function () {
      fetch('/api/sample').then(function (res) { if (!res.ok) throw new Error('The sample file is not on the server.'); return res.blob(); })
        .then(function (blob) { loadBatch(blob, 'bangladesh_heart_patients.csv'); })
        .catch(function (err) { setStatus('err', esc(err.message)); });
    });
  }

  /* ---------- Routing, theme, shortcuts ---------- */
  var PAGES = ['home', 'predict', 'batch'];
  function route() {
    var name = (location.hash.replace(/^#\/?/, '') || 'home'); if (PAGES.indexOf(name) < 0) name = 'home';
    $$('.page').forEach(function (s) { s.hidden = s.dataset.page !== name; });
    $$('[data-nav]').forEach(function (a) { if (a.dataset.nav === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    window.scrollTo(0, 0);
    if (name === 'predict') updateEcg();
  }

  function initTheme() {
    var root = document.documentElement, btn = $('#themeBtn');
    function sync() { var dark = root.getAttribute('data-theme') === 'dark'; btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme'); $('.theme-label').textContent = dark ? 'Light theme' : 'Dark theme'; }
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next); try { localStorage.setItem('hd-theme', next); } catch (e) {}
      sync();
    });
    sync();
  }

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target; if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    var i = '123'.indexOf(e.key); if (i >= 0) { location.hash = '#/' + PAGES[i]; }
  });

  /* ---------- Boot ---------- */
  function fillInfo() {
    var m = INFO.metrics || {}, mod = INFO.model, dash = '–';
    var v = {
      k: mod.k, train: mod.n_train != null ? mod.n_train : dash,
      test: m.n_test != null ? m.n_test : dash,
      accuracy: m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : dash,
      f1: m.f1 != null ? m.f1.toFixed(2) : dash
    };
    $$('[data-m]').forEach(function (el) { el.textContent = v[el.dataset.m]; });
  }

  initTheme();
  window.addEventListener('hashchange', route);
  route();
  api('/api/info').then(function (info) {
    INFO = info;
    fillInfo(); buildHero(); buildPredict(); buildBatch();
    route();
  }).catch(function () { $('#modelError').hidden = false; });
})();
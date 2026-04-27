"""
SepsisAI — Flask Backend
CSS and JS are inlined into the HTML response so browser caching cannot break the UI.
"""

import os
from flask import Flask, Response
from flask_cors import CORS

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models_store')
os.makedirs(UPLOAD_DIR, exist_ok=True)

_CSS_PATH = os.path.join(BASE_DIR, '..', 'frontend', 'static', 'css', 'main.css')
_JS_PATH  = os.path.join(BASE_DIR, '..', 'frontend', 'static', 'js',  'app.js')

def _read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def build_page():
    css = _read(_CSS_PATH)
    js  = _read(_JS_PATH)
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SepsisAI \u2014 Clinical Prediction Platform</title>
<style>
:root {
  --font-head: -apple-system,"Segoe UI Semibold","Segoe UI",system-ui,sans-serif;
  --font-body: -apple-system,"Segoe UI",system-ui,sans-serif;
  --font-mono: "Cascadia Code","Consolas","Fira Code","SF Mono","Courier New",monospace;
}
""" + css + """
</style>
</head>
<body>
<nav class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle cx="14" cy="14" r="13" stroke="#ef4444" stroke-width="1.5"/>
        <path d="M9 14h3l2-5 2 10 2-5h1" stroke="#ef4444" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <span class="nav-name">SepsisAI</span>
    <span class="nav-tag">Clinical Decision Support</span>
  </div>
  <div class="nav-status">
    <div class="status-dot" id="api-status-dot"></div>
    <span class="status-text" id="api-status-text">Connecting...</span>
    <span class="nav-version">v1.0</span>
  </div>
</nav>

<section class="hero" id="home-section">
  <div class="hero-bg"><div class="hero-grid"></div><div class="hero-glow"></div></div>
  <div class="hero-content">
    <div class="hero-badge"><span class="badge-dot"></span>AI-Powered &middot; Sepsis-3 Protocol &middot; Real-time Analysis</div>
    <h1 class="hero-title">Early Sepsis<br><span class="hero-accent">Detection</span></h1>
    <p class="hero-sub">Upload a blood report PDF for instant sepsis risk prediction.<br>Missing values? Fill them in &mdash; we guide you every step.</p>
    <div class="hero-actions">
      <button class="btn-primary" onclick="scrollToApp()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Upload Blood Report
      </button>
      <button class="btn-ghost" onclick="showManualEntry()">Enter Values Manually</button>
    </div>
    <div class="hero-stats">
      <div class="stat"><span class="stat-num">40+</span><span class="stat-lbl">Biomarkers</span></div>
      <div class="stat-sep"></div>
      <div class="stat"><span class="stat-num">4</span><span class="stat-lbl">ML Models</span></div>
      <div class="stat-sep"></div>
      <div class="stat"><span class="stat-num">Sepsis-3</span><span class="stat-lbl">Protocol</span></div>
    </div>
  </div>
  <div class="hero-visual">
    <div class="monitor-card">
      <div class="monitor-header"><span class="monitor-label">LIVE ECG</span><span class="monitor-hr">72 bpm</span></div>
      <svg class="ecg-line" viewBox="0 0 300 80" preserveAspectRatio="none">
        <polyline class="ecg-path" fill="none" points="0,40 30,40 40,40 50,10 60,60 70,40 90,40 120,40 130,40 140,10 150,60 160,40 180,40 210,40 220,10 230,60 240,40 270,40 300,40"/>
      </svg>
      <div class="monitor-vals">
        <div class="mval"><span class="mval-num" style="color:#ef4444">98%</span><span class="mval-lbl">SpO2</span></div>
        <div class="mval"><span class="mval-num" style="color:#22c55e">120/80</span><span class="mval-lbl">BP</span></div>
        <div class="mval"><span class="mval-num" style="color:#f59e0b">37.1deg</span><span class="mval-lbl">Temp</span></div>
      </div>
    </div>
  </div>
</section>

<main class="app-section" id="app-section">
  <div class="app-container">
    <div class="step-bar">
      <div class="step active" id="step-1"><div class="step-num">1</div><span>Upload / Enter</span></div>
      <div class="step-line"></div>
      <div class="step" id="step-2"><div class="step-num">2</div><span>Review &amp; Complete</span></div>
      <div class="step-line"></div>
      <div class="step" id="step-3"><div class="step-num">3</div><span>Prediction</span></div>
    </div>

    <div class="panel active" id="panel-upload">
      <div class="panel-header"><h2>Upload Blood Report</h2><p>PDF will be scanned for lab values automatically</p></div>
      <div class="upload-zone" id="drop-zone">
        <input type="file" id="file-input" accept=".pdf" style="display:none">
        <div class="upload-icon-wrap">
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14,2 14,8 20,8"/>
            <line x1="12" y1="18" x2="12" y2="12"/>
            <polyline points="9,15 12,12 15,15"/>
          </svg>
        </div>
        <h3 class="upload-title">Drop your PDF here</h3>
        <p class="upload-sub">Blood report &middot; Lab results &middot; Pathology report</p>
        <button class="btn-outline" id="browse-btn" type="button">Browse File</button>
        <div id="upload-file-name" style="display:none;margin-top:16px;padding:6px 14px;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);border-radius:20px;font-size:13px;color:#93c5fd;font-family:monospace"></div>
      </div>
      <div class="divider"><span>or</span></div>
      <button class="btn-ghost-full" onclick="showManualEntry()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        Enter values manually (no PDF)
      </button>
      <div id="upload-actions" style="display:none;margin-top:24px;justify-content:flex-end">
        <button class="btn-primary" onclick="analyzeUpload()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Analyze Report
        </button>
      </div>
    </div>

    <div class="panel" id="panel-review">
      <div class="panel-header"><h2>Review Extracted Values</h2><p id="review-subtitle">Fill in any missing fields below.</p></div>
      <div class="completeness-bar-wrap">
        <div class="completeness-info"><span>Data completeness</span><span id="completeness-pct">0%</span></div>
        <div class="completeness-track"><div class="completeness-fill" id="completeness-fill"></div></div>
        <div class="completeness-note" id="completeness-note"></div>
      </div>
      <div class="alert alert-warn" id="data-warning" style="display:none">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span id="data-warning-text">Not enough data for reliable prediction.</span>
      </div>
      <div class="values-form" id="values-form"></div>
      <div class="review-actions">
        <button class="btn-ghost" onclick="goBack()">&larr; Back</button>
        <button class="btn-primary" id="predict-btn" onclick="runPrediction()">Run Prediction <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg></button>
      </div>
    </div>

    <div class="panel" id="panel-result">
      <div id="result-content"></div>
      <div class="result-actions">
        <button class="btn-ghost" onclick="startOver()">&larr; New Prediction</button>
        <button class="btn-outline" onclick="window.print()">Print Report</button>
      </div>
    </div>
  </div>
</main>

<div id="loading-overlay" style="display:none;position:fixed;inset:0;z-index:9999;align-items:center;justify-content:center;background:rgba(10,10,11,0.92)">
  <div style="text-align:center">
    <svg viewBox="0 0 50 50" width="56" height="56" style="display:block;margin:0 auto 16px">
      <circle cx="25" cy="25" r="20" fill="none" stroke="#ef4444" stroke-width="3" stroke-dasharray="100 26" stroke-linecap="round">
        <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
      </circle>
    </svg>
    <div id="loading-msgs" style="font-family:monospace;font-size:14px;color:#a1a1aa">Analyzing...</div>
  </div>
</div>

<footer class="footer">
  <div class="footer-inner">
    <div class="footer-brand">SepsisAI</div>
    <div class="footer-disclaimer">For clinical decision support only. Not a substitute for professional medical judgement.</div>
    <div class="footer-right">Sepsis-3 Protocol</div>
  </div>
</footer>

<script>
// Patch .hidden property to use style.display (avoids CSS override issues)
['upload-file-name','upload-actions','data-warning'].forEach(id => {
  document.addEventListener('DOMContentLoaded', () => {
    const el = document.getElementById(id);
    if (!el) return;
    Object.defineProperty(el, 'hidden', {
      get() { return this.style.display === 'none'; },
      set(v) { this.style.display = v ? 'none' : (id === 'upload-actions' ? 'flex' : ''); }
    });
  });
});
</script>
<script>
""" + js + """
</script>
</body>
</html>"""


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    app.config['UPLOAD_FOLDER']      = UPLOAD_DIR
    app.config['MODELS_DIR']         = MODELS_DIR
    app.config['SECRET_KEY']         = os.environ.get('SECRET_KEY', 'sepsis-ai-dev-key-2024')

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    from routes.predict_route import predict_bp
    from routes.report_route  import report_bp
    from routes.health_route  import health_bp

    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(report_bp,  url_prefix='/api')
    app.register_blueprint(health_bp,  url_prefix='/api')

    @app.route('/')
    @app.route('/<path:path>')
    def serve_frontend(path=''):
        if path.startswith('api'):
            from flask import abort; abort(404)
        resp = Response(build_page(), mimetype='text/html')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma']        = 'no-cache'
        resp.headers['Expires']       = '0'
        return resp

    return app

app = create_app()
if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}\n  SepsisAI — http://localhost:{port}\n{'='*50}\n")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
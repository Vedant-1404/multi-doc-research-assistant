"""
styles.py — Custom CSS injected into Streamlit.

Design direction: technical/editorial — dark slate bg, warm amber accent,
monospace data elements. Feels like a research tool, not a chatbot.
"""

CUSTOM_CSS = """
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg: #0f1117;
    --surface: #1a1f2e;
    --surface-2: #222840;
    --border: #2d3452;
    --text: #e8eaf0;
    --text-muted: #8892a4;
    --accent: #f5a623;
    --accent-dim: rgba(245,166,35,0.15);
    --green: #4ade80;
    --red: #f87171;
    --blue: #60a5fa;
    --radius: 8px;
    --font-body: 'Inter', sans-serif;
    --font-serif: 'Libre Baskerville', serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* ── App shell ── */
.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
}

/* ── Header / Title ── */
h1 {
    font-family: var(--font-serif) !important;
    color: var(--text) !important;
    font-size: 2rem !important;
    letter-spacing: -0.02em;
}

h2, h3 {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--text) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stChatMessage"] p {
    font-family: var(--font-body);
    line-height: 1.7;
    color: var(--text) !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}

.stChatInput textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Source cards ── */
.source-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

.source-card .source-header {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--accent);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.source-card .score-bar {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}

.source-card .excerpt {
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.55;
    font-style: italic;
    border-top: 1px solid var(--border);
    padding-top: 0.4rem;
    margin-top: 0.4rem;
}

/* ── Metric pills ── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent);
}

/* ── Document badge ── */
.doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    margin: 0.2rem;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0f1117 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
    transition: opacity 0.15s !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Text inputs ── */
.stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Confidence badge variants ── */
.badge-high   { color: var(--green); border-color: var(--green); }
.badge-medium { color: var(--accent); border-color: var(--accent); }
.badge-low    { color: var(--red); border-color: var(--red); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Spinner ── */
.stSpinner { color: var(--accent) !important; }

/* ── Success / Error / Warning ── */
.stSuccess { background: rgba(74,222,128,0.1) !important; border-color: var(--green) !important; }
.stError   { background: rgba(248,113,113,0.1) !important; border-color: var(--red) !important; }
.stWarning { background: rgba(245,166,35,0.1) !important; border-color: var(--accent) !important; }
</style>
"""

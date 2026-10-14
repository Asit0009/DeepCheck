import streamlit as st

CSS = """
<style>
:root {
  --bg: #0B1220; --panel: #131B2B; --panel-2: #182236; --border: #26324A;
  --text: #F2F5FA; --muted: #93A1BC; --accent: #4C8DFF;
  --ai: #E5A23B; --real: #3DBE8B; --unsure: #93A1BC;
}
.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
.dfd-brand { color: var(--muted); font-size: 0.95rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.dfd-lead { color: var(--muted); font-size: 1.05rem; line-height: 1.6; margin: -.3rem 0 1.4rem 0; }
.dfd-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1.3rem; margin: .8rem 0; }
.dfd-chip { display: inline-block; padding: .35rem .85rem; border-radius: 999px; font-weight: 650; font-size: 1.05rem;
            border: 1px solid currentColor; }
.dfd-chip.likely_ai { color: var(--ai); background: rgba(229,162,59,.10); }
.dfd-chip.likely_real { color: var(--real); background: rgba(61,190,139,.10); }
.dfd-chip.inconclusive { color: var(--unsure); background: rgba(147,161,188,.10); }
.dfd-score { font-size: 2.1rem; font-weight: 700; margin: .5rem 0 .1rem 0; }
.dfd-meter { height: 8px; border-radius: 99px; background: var(--panel-2); overflow: hidden; margin: .4rem 0 .5rem 0; }
.dfd-meter > div { height: 100%; background: var(--accent); border-radius: 99px; }
.dfd-muted { color: var(--muted); font-size: .92rem; line-height: 1.5; }
.dfd-stat { display: inline-block; margin-right: 1.8rem; }
.dfd-stat b { font-size: 1.35rem; display: block; }
.dfd-warn { border-left: 3px solid var(--ai); background: rgba(229,162,59,.07); padding: .6rem .9rem; border-radius: 6px;
            color: #D9DEE8; font-size: .92rem; margin-top: .8rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1.5px dashed var(--border); border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p { font-size: 0; margin: 0; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after { content: "Choose Image"; font-size: .95rem; }
.stButton > button[kind="primary"] { background: var(--accent); border: none; font-weight: 650; padding: .55rem 1.4rem; }
.stButton > button { border-radius: 10px; }
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)  # inject stylesheet into app

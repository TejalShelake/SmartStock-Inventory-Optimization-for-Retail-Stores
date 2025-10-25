import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess, sys
import matplotlib.pyplot as plt
import random

# --------------------------------------------------------
# Streamlit Config
# --------------------------------------------------------
st.set_page_config(page_title="SmartStock – Retail Dashboard", layout="wide")

# --------------------------------------------------------
# Theme Toggle
# --------------------------------------------------------
theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Light 🌞", "Teal 🌊"],
    index=1
)

# --------------------------------------------------------
# Dynamic Theme Colors
# --------------------------------------------------------
if "🌞" in theme:
    background = "linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%)"
    sidebar = "linear-gradient(180deg, #1E3A8A, #2563EB, #38BDF8)"
    accent = "#0077b6"
    text_color = "#1b263b"
elif "🌙" in theme:
    background = "linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)"
    sidebar = "linear-gradient(180deg, #000000, #1a1a1a)"
    accent = "#00b4d8"
    text_color = "#e0e0e0"
else:
    background = "linear-gradient(135deg, #c8e7f5 0%, #a1f2d0 100%)"
    sidebar = "linear-gradient(180deg, #004d40, #00796b, #26a69a)"
    accent = "#00b4d8"
    text_color = "#0f172a"

# --------------------------------------------------------
# CSS Styling (Final Fixed Version – Visible Uploader Text)
# --------------------------------------------------------
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: {background};
        color: {text_color};
        transition: all 0.5s ease;
    }}
    [data-testid="stSidebar"] {{
        background: {sidebar};
        color: white;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, {accent}, #48cae4);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }}
    div.stButton > button:first-child:hover {{
        transform: scale(1.05);
        background: linear-gradient(90deg, #48cae4, {accent});
        box-shadow: 0 0 12px rgba(0,183,255,0.8);
    }}

    /* Tabs */
    .stTabs [role="tablist"] {{
        border-bottom: 3px solid {accent};
    }}
    .stTabs [role="tab"] {{
        background: rgba(255,255,255,0.7);
        border-radius: 12px 12px 0 0;
        margin-right: 5px;
        font-weight: 600;
        color: {text_color};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {accent}, #48cae4);
        color: white !important;
        box-shadow: 0px -2px 10px rgba(0, 183, 255, 0.4);
    }}

    /* Tables & Downloads */
    .dataframe {{
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0px 6px 16px rgba(0,0,0,0.08);
        background-color: white;
        margin-top: 10px;
    }}
    .stDownloadButton > button {{
        background: linear-gradient(90deg, {accent}, #48cae4);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }}
    .stDownloadButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 10px rgba(0,183,255,0.6);
    }}

    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# JS + CSS Fix for Invisible Text in File Uploader
# --------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stFileUploader"] div[role="button"] * {
    color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
}
section[data-testid="stFileUploader"] > div:first-child {
    background: rgba(0, 0, 0, 0.25) !important;
    border: 2px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: 12px !important;
    text-align: center !important;
}
</style>

<script>
setTimeout(function() {
    const uploaders = window.parent.document.querySelectorAll('section[data-testid="stFileUploader"]');
    uploaders.forEach(uploader => {
        const texts = uploader.querySelectorAll('p, span, label, div');
        texts.forEach(el => {
            el.style.color = '#ffffff';
            el.style.opacity = '1';
            el.style.visibility = 'visible';
            el.style.textShadow = '0 0 4px rgba(0,0,0,0.8)';
        });
    });
}, 1500);
</script>
""", unsafe_allow_html=True)
# ✅ Final Uploader Text Visibility Fix
st.markdown("""
<style>
[data-testid="stFileUploader"] * {
    color: #000000 !important;       /* मजकूर काळा दिसेल */
    font-weight: 600 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stFileUploader"] > div:first-child {
    background: rgba(255,255,255,0.85) !important;  /* हलका background */
    border: 2px dashed #00796b !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)
# ✅ Strong Fix for Theme Dropdown Visibility
st.markdown("""
<style>
/* Main dropdown box (closed state) */
div[data-baseweb="select"] > div {
    background-color: rgba(0, 0, 0, 0.4) !important; /* dark semi-transparent box */
    color: #ffffff !important;                       /* white text */
    border: 1.5px solid #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    text-shadow: 0 0 3px rgba(0,0,0,0.8);
}

/* Dropdown text and emoji */
div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* When dropdown is open (options list) */
ul[role="listbox"] {
    background-color: rgba(0, 0, 0, 0.85) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
ul[role="listbox"] li {
    color: #ffffff !important;
    font-weight: 600 !important;
}
ul[role="listbox"] li:hover {
    background-color: rgba(72, 202, 228, 0.7) !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
EDA_DIR = PROJECT_ROOT / "eda_plots"
FORECAST_DIR = PROJECT_ROOT / "forecast_plots"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for f in [DATA_DIR, OUTPUT_DIR, EDA_DIR, FORECAST_DIR]:
    f.mkdir(exist_ok=True)

CLEANED_FILE = OUTPUT_DIR / "cleaned_retail_inventory_dataset.csv"
FORECAST_FILE = DATA_DIR / "forecast_results.csv"
EVAL_FILE = DATA_DIR / "forecast_evaluation_summary.csv"

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def run_script(name, timeout=600):
    s = SCRIPT_DIR / name
    if not s.exists():
        return False, f"Script not found: {s}"
    try:
        proc = subprocess.run(
            [sys.executable, str(s)],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=timeout
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except Exception as e:
        return False, str(e)

def list_images(folder):
    return sorted([f for f in folder.glob("*.png") if f.is_file()])

# --------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------
st.sidebar.title("Control Panel")
uploaded = st.sidebar.file_uploader("Upload Raw Sales CSV", type=["csv"])
if uploaded:
    path = DATA_DIR / "retail_inventory_dataset.csv"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved to: {path}")

if st.sidebar.button("Run Milestone 1 (Preprocessing)"):
    st.info("Running Milestone 1: Cleaning and EDA...")
    ok, log = run_script("milestone1_data_preprocessing.py", timeout=240)
    st.code(log)
    st.success("Milestone 1 Completed" if ok else "Milestone 1 Failed")

if st.sidebar.button("Run Milestone 2 (Forecasting)"):
    st.info("Running Milestone 2: Forecasting...")
    ok, log = run_script("milestone2_forecasting1.py", timeout=900)
    st.code(log)
    st.success("Milestone 2 Completed" if ok else "Milestone 2 Failed")

if st.sidebar.button("Run Both (1 & 2)"):
    st.info("Running Milestones 1 and 2 sequentially...")
    ok1, log1 = run_script("milestone1_data_preprocessing.py", timeout=240)
    st.code(log1)
    if ok1:
        ok2, log2 = run_script("milestone2_forecasting1.py", timeout=900)
        st.code(log2)
        st.success("All Milestones Completed" if ok2 else "Milestone 2 Failed")
    else:
        st.error("Milestone 1 Failed")

# --------------------------------------------------------
# Header
# --------------------------------------------------------
emoji = random.choice(["📦", "📊", "📈", "🧮", "🏪"])
st.markdown(f"""
<div style="
background:linear-gradient(90deg,{accent},#48cae4);
padding:15px 25px;border-radius:10px;margin-bottom:25px;color:white;">
<h1 style="margin-bottom:0;">{emoji} SmartStock Retail Inventory Optimization Dashboard</h1>
<p style="font-size:17px;margin-top:4px;">
Milestone 4 – Streamlit Dashboard and Reporting | Developed by <b>[Your Name]</b> | <b>[Your Institute / Organization]</b>
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Tabs
# --------------------------------------------------------
tabs = st.tabs(["Data & EDA", "Forecasts", "Inventory Optimization", "Reports"])

# TAB 1
with tabs[0]:
    st.header("Data and Exploratory Data Analysis")
    df = safe_read_csv(CLEANED_FILE)
    if df is not None:
        st.dataframe(df.head(200), width='stretch')
        st.success(f"Loaded {len(df):,} rows of cleaned data.")
    else:
        st.warning("Cleaned dataset not found. Run Milestone 1 first.")
    st.subheader("EDA Visuals")
    imgs = list_images(EDA_DIR)
    if imgs:
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            with cols[i % 3]:
                st.image(str(img), caption=img.name, width='stretch')
    else:
        st.info("No EDA plots found. Run Milestone 1.")

# TAB 2
with tabs[1]:
    st.header("Forecast Results and Model Evaluation")
    forecast = safe_read_csv(FORECAST_FILE)
    eval_df = safe_read_csv(EVAL_FILE)
    if forecast is not None:
        st.dataframe(forecast.head(100), width='stretch')
        products = sorted(
            forecast["item_name"].dropna().unique()
            if "item_name" in forecast
            else forecast["product"].unique())
        if products:
            sel = st.selectbox("Select Product", products)
            safe = "".join(c if c.isalnum() else "_" for c in sel)
            plot = FORECAST_DIR / f"forecast_{safe}.png"
            if plot.exists():
                st.image(str(plot), caption=f"Forecast Plot for {sel}", width='stretch')
            else:
                st.warning("No forecast plot found.")
    else:
        st.warning("No forecast results found. Run Milestone 2 first.")
    if eval_df is not None:
        st.subheader("Forecast Evaluation Summary")
        st.dataframe(eval_df.head(200), width='stretch')

# TAB 3
with tabs[2]:
    st.header("Inventory Optimization (EOQ / ROP / Safety Stock / ABC)")
    forecast = safe_read_csv(FORECAST_FILE)
    if forecast is None:
        st.warning("Run Milestone 2 first to generate forecast data.")
    else:
        lead = st.slider("Lead Time (days)", 1, 90, 7)
        oc = st.number_input("Ordering Cost ($/order)", 0.0, 1000.0, 50.0)
        hc = st.number_input("Holding Cost ($/unit)", 0.0, 10.0, 2.0)
        svc = st.selectbox("Service Level", ["90%", "95%", "99%"])
        z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[svc]
        rows = []
        for prod, grp in forecast.groupby(forecast.columns[-1]):
            total = grp["forecast"].sum()
            avg_daily = grp["forecast"].mean() / 30
            std = grp["forecast"].std(ddof=0)
            eoq = np.sqrt((2 * total * oc) / max(hc, 1e-6))
            safety = z * (std / 30) * np.sqrt(lead)
            rop = (avg_daily * lead) + safety
            rows.append({
                "Product": prod,
                "AvgDailySales": round(avg_daily, 3),
                "EOQ": round(eoq, 2),
                "SafetyStock": round(safety, 2),
                "ReorderPoint": round(rop, 2),
                "TotalForecast": round(total, 2)
            })
        inv = pd.DataFrame(rows)
        if not inv.empty:
            inv["InventoryValue"] = inv["TotalForecast"] * hc
            inv = inv.sort_values("InventoryValue", ascending=False).reset_index(drop=True)
            inv["Cumulative%"] = inv["InventoryValue"].cumsum() / inv["InventoryValue"].sum() * 100
            inv["ABC"] = inv["Cumulative%"].apply(lambda x: "A" if x <= 20 else "B" if x <= 50 else "C")
            st.dataframe(inv, width='stretch')
            st.download_button("Download Inventory Report",
                               inv.to_csv(index=False).encode("utf-8"),
                               "inventory_plan.csv", "text/csv")

# TAB 4
with tabs[3]:
    st.header("Reports and Downloads")
    if CLEANED_FILE.exists():
        st.download_button("Cleaned Dataset (M1)", CLEANED_FILE.read_bytes(), CLEANED_FILE.name)
    if FORECAST_FILE.exists():
        st.download_button("Forecast Results (M2)", FORECAST_FILE.read_bytes(), FORECAST_FILE.name)
    if EVAL_FILE.exists():
        st.download_button("Forecast Evaluation Summary", EVAL_FILE.read_bytes(), EVAL_FILE.name)
    imgs = list_images(FORECAST_DIR)
    if imgs:
        sel = st.selectbox("Select Forecast Plot", [i.name for i in imgs])
        st.image(str(FORECAST_DIR / sel), caption=sel, width='stretch')
    else:
        st.info("No forecast plots found. Run Milestone 2.")

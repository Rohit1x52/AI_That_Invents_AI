import streamlit as st
import pandas as pd
import sqlite3
import json
import subprocess
import shlex
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

st.set_page_config(
    page_title="AI Inventor — Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    .stCodeBlock { background-color: #0e1117; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #4CAF50; }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #333;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "dkb.sqlite"

@st.cache_data(ttl=30)
def load_architectures(dkb_path: str) -> pd.DataFrame:
    if not Path(dkb_path).exists():
        return pd.DataFrame()
    
    con = sqlite3.connect(dkb_path)
    try:
        query = "SELECT id, name, blueprint_json, created_at, params, flops, summary FROM architectures"
        df = pd.read_sql_query(query, con)
    except Exception:
        try:
            df = pd.read_sql_query("SELECT * FROM architectures", con)
        except Exception:
            con.close()
            return pd.DataFrame()
    con.close()
    
    if "blueprint_json" in df.columns:
        def try_parse(b):
            try:
                return json.loads(b) if isinstance(b, str) else b
            except Exception:
                return {}
        df["blueprint_parsed"] = df["blueprint_json"].apply(try_parse)
        
    if "params" in df.columns:
        df["params_fmt"] = df["params"].apply(lambda x: f"{x/1e6:.2f}M" if x and x > 0 else "N/A")
    return df

@st.cache_data(ttl=30)
def load_metrics(dkb_path: str) -> pd.DataFrame:
    if not Path(dkb_path).exists():
        return pd.DataFrame()
    con = sqlite3.connect(dkb_path)
    try:
        m = pd.read_sql_query("SELECT * FROM metrics", con)
    except Exception:
        try:
            m = pd.read_sql_query("SELECT * FROM trials", con)
        except Exception:
            m = pd.DataFrame()
    con.close()
    return m

def run_command(cmd: str, workdir: Optional[str] = None, timeout: Optional[int] = None):
    with st.status(f"Executing: {cmd.split()[0]}...", expanded=True) as status:
        st.write(f"Command: `{cmd}`")
        output_container = st.empty()
        full_output = []
        
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=workdir)
        try:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    clean_line = line.rstrip()
                    full_output.append(clean_line)
                    output_container.code("\n".join(full_output[-15:]), language="bash")
            
            rc = proc.wait(timeout=timeout)
            if rc == 0:
                status.update(label="Process completed successfully!", state="complete", expanded=False)
            else:
                status.update(label="Process failed.", state="error")
                st.error(f"Return Code: {rc}")
                
        except subprocess.TimeoutExpired:
            proc.kill()
            status.update(label="Process timed out.", state="error")
            st.error("Command timed out and was killed.")
            return {"rc": 124}
            
    return {"rc": rc}

with st.sidebar:
    st.image("https://img.icons8.com/dusk/64/000000/neural-network.png", width=50)
    st.title("Control Center")
    dkb_path_input = st.text_input("Database Path", value=DB_PATH)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
    
    st.divider()
    st.info("Phase 2: Automated NAS & Optimization pipeline.")

df_arch = load_architectures(dkb_path_input)
df_metrics = load_metrics(dkb_path_input)

st.title("🧬 AI That Invents AI")
st.markdown("### Evolutionary Architecture Search Dashboard")

if not df_arch.empty:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Discovered Architectures", len(df_arch))
    with kpi2:
        best_acc = 0.0
        if not df_metrics.empty:
            for col in ["val_acc", "best_val_acc", "accuracy"]:
                if col in df_metrics.columns:
                    best_acc = df_metrics[col].max()
                    break
        st.metric("Best Accuracy", f"{best_acc:.2%}")
    with kpi3:
        avg_params = df_arch["params"].mean() if "params" in df_arch.columns else 0
        st.metric("Avg Parameter Count", f"{avg_params/1e6:.2f}M")
    with kpi4:
        st.metric("Database Status", "Connected" if Path(dkb_path_input).exists() else "Missing")
else:
    st.warning("Database is empty or not found. Please check the path in the sidebar.")

tab_explore, tab_campaign, tab_ops = st.tabs(["📊 Analytics & Discovery", "🧪 Launch Campaign", "⚙️ Operations"])

with tab_explore:
    col_viz, col_details = st.columns([2, 1])
    
    plot_df = df_metrics.copy()
    
    target_acc = next((c for c in ["val_acc", "best_val_acc", "accuracy"] if c in plot_df.columns), None)
    target_lat = next((c for c in ["latency_cpu_ms", "latency_ms", "flops"] if c in plot_df.columns), None)
    
    if "params" not in plot_df.columns and "arch_id" in plot_df.columns and not df_arch.empty:
        try:
            arch_map = df_arch.set_index("id")["params"].to_dict()
            plot_df["params"] = plot_df["arch_id"].map(arch_map)
        except KeyError:
            pass

    with col_viz:
        st.subheader("Pareto Frontier Explorer")
        if target_acc and target_lat:
            fig = px.scatter(
                plot_df, 
                x=target_lat, 
                y=target_acc, 
                size="params" if "params" in plot_df.columns else None,
                color=target_acc,
                color_continuous_scale="Viridis",
                hover_data=["arch_id", "trial_id"] if "arch_id" in plot_df.columns else None,
                labels={target_acc: "Accuracy", target_lat: "Cost (Latency/FLOPs)"},
                title="Accuracy vs. Efficiency Trade-off"
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for Pareto plot (Need accuracy and latency/flops columns).")

    with col_details:
        st.subheader("Leaderboard")
        if not df_metrics.empty and target_acc:
            leaderboard = plot_df.sort_values(target_acc, ascending=False).head(10)
            cols_to_show = [target_acc]
            if "arch_id" in leaderboard.columns:
                cols_to_show.insert(0, "arch_id")
            if "params" in leaderboard.columns:
                cols_to_show.append("params")
            st.dataframe(
                leaderboard[cols_to_show],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No metrics available.")

    st.divider()
    st.subheader("Architecture Library")
    
    if not df_arch.empty:
        event = st.dataframe(
            df_arch[["id", "name", "created_at", "params_fmt", "summary"]],
            selection_mode="single-row",
            on_select="rerun",
            use_container_width=True,
            height=300
        )
        
        if len(event.selection["rows"]):
            idx = event.selection["rows"][0]
            selected_row = df_arch.iloc[idx]
            
            with st.expander(f"🔎 Blueprint Details: {selected_row['name']}", expanded=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.json(selected_row["blueprint_parsed"])
                with c2:
                    st.markdown(f"**Created:** {selected_row['created_at']}")
                    st.markdown(f"**Params:** {selected_row['params']}")
                    st.markdown(f"**FLOPs:** {selected_row.get('flops', 'N/A')}")
                    st.info("To export this model, go to the Operations tab.")

with tab_campaign:
    st.header("Start New Discovery Campaign")
    st.caption("Evolve new architectures using Genetic Algorithms or Random Search.")
    
    with st.container(border=True):
        col_form, col_help = st.columns([2, 1])
        
        with col_form:
            with st.form("campaign_form"):
                uploaded_file = st.file_uploader("Seed Blueprint (JSON)", type=["json"])
                
                c1, c2 = st.columns(2)
                with c1:
                    n_candidates = st.number_input("Population Size", 5, 100, 10)
                    device = st.selectbox("Training Device", ["cpu", "cuda"])
                with c2:
                    workers = st.number_input("Parallel Workers", 1, 16, 1)
                    params_max = st.number_input("Max Params Constraint", 0, 100000000, 500000, step=100000)

                dry_run = st.checkbox("Dry Run (Simulate only)", value=False)
                
                submit_btn = st.form_submit_button("🚀 Launch Campaign", type="primary")

        with col_help:
            st.markdown("""
            **Guide:**
            1. **Seed:** Initial architecture to mutate.
            2. **Population:** How many variants to generate.
            3. **Workers:** Increase if you have multiple GPUs.
            4. **Constraints:** Soft limits for the search.
            """)

    if submit_btn:
        if not uploaded_file:
            st.error("Please upload a seed blueprint.")
        else:
            bp_path = Path("frontend_tmp_blueprint.json")
            bp_path.write_bytes(uploaded_file.getvalue())
            
            cmd = (
                f"python bin/run_campaign.py "
                f"--blueprint {bp_path} "
                f"--n {n_candidates} "
                f"--dkb {dkb_path_input} "
                f"--device {device} "
            )
            if params_max > 0:
                cmd += f" --params_max {params_max}"
            if workers > 1:
                cmd += f" --parallel {workers}"
            if dry_run:
                cmd += " --dry_run"
            
            run_command(cmd)

with tab_ops:
    c_retrain, c_export = st.columns(2)
    
    with c_retrain:
        st.subheader("🏋️ Retrain Champions")
        st.markdown("Train the best models fully for production.")
        
        with st.form("retrain_form"):
            r_epochs = st.number_input("Epochs", 1, 500, 50)
            r_bs = st.number_input("Batch Size", 16, 512, 128)
            r_dev = st.selectbox("Device", ["cuda", "cpu"])
            
            st.caption("Automatically selects best models from DKB if no file provided.")
            champ_file = st.file_uploader("Specific Champions JSON (Optional)", type=["json"])
            
            if st.form_submit_button("Start Retraining"):
                cmd = f"python bin/retrain_champions.py --dkb {dkb_path_input} --epochs {r_epochs} --batch {r_bs} --device {r_dev}"
                if champ_file:
                    tmp_p = Path("frontend_champs.json")
                    tmp_p.write_bytes(champ_file.getvalue())
                    cmd += f" --champions {tmp_p}"
                else:
                    cmd += " --champions champions.json"
                
                run_command(cmd)

    with c_export:
        st.subheader("📦 Export Model")
        st.markdown("Convert architecture to TorchScript/ONNX.")
        
        arch_opts = df_arch["id"].tolist() if not df_arch.empty else []
        sel_export_id = st.selectbox("Select Architecture ID", arch_opts)
        
        if st.button("Export Selected Model"):
            if not sel_export_id:
                st.error("No architecture selected.")
            else:
                trial_arg = ""
                if not df_metrics.empty and "arch_id" in df_metrics.columns:
                    trials = df_metrics[df_metrics["arch_id"] == sel_export_id]
                    if not trials.empty:
                        best_t = trials.sort_values("created_at", ascending=False).iloc[0]["trial_id"]
                        trial_arg = f" --trial {best_t}"
                
                cmd = f"python bin/export_champion.py --dkb {dkb_path_input} --arch {sel_export_id}{trial_arg}"
                run_command(cmd)

st.markdown("<div class='footer'>Made with ❤️ by Rohit</div>", unsafe_allow_html=True)
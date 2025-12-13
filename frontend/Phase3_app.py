import streamlit as st
import sqlite3
import json
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import numpy as np
import os

# ===============================
# 1. Configuration & Styling
# ===============================
st.set_page_config(
    page_title="Neural EvoLab | Phase 3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Cool" UI
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #0E1117; 
        border-radius: 5px;
        padding: 10px;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

try:
    DB_PATH = st.secrets.get("dkb_path", "dkb.sqlite")
except Exception:
    DB_PATH = "dkb.sqlite"

if not os.path.exists(DB_PATH) and os.path.exists(os.path.join("..", "dkb.sqlite")):
    DB_PATH = os.path.join("..", "dkb.sqlite")

# ===============================
# 2. Data Loader
# ===============================
@st.cache_data(ttl=60) # Auto-refresh every 60s
def load_data():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    def get_df(query):
        try:
            return pd.read_sql(query, conn)
        except Exception:
            return pd.DataFrame()

    data = {
        "archs": get_df("SELECT * FROM architectures"),
        "trials": get_df("SELECT * FROM trials"),
        "metrics": get_df("SELECT * FROM metrics"),
        # Try/Except for tables that might not exist in early runs
        "genealogy": get_df("SELECT * FROM genealogy"),
    }
    
    # Merge for a Master Table
    if not data["trials"].empty and not data["archs"].empty:
        # Rename trials.id to trial_id for clarity before merge
        trials_df = data["trials"].rename(columns={"id": "trial_id"})
        
        # Get latest metric per trial
        best_metrics = data["metrics"].sort_values("epoch", ascending=False).drop_duplicates("trial_id")
        
        master = trials_df.merge(data["archs"], left_on="arch_id", right_on="id", suffixes=("_trial", "_arch"))
        master = master.merge(best_metrics, on="trial_id", how="left", suffixes=("", "_metric"))
        
        # DNA parsing
        def parse_dna(x):
            try: return json.loads(x)
            except: return {}
        
        if "dna_json" in master.columns:
            master["dna_dict"] = master["dna_json"].apply(parse_dna)
            # Extract key DNA features for sorting
            master["depth"] = master["dna_dict"].apply(lambda d: d.get("total_depth", 0))
            master["width"] = master["dna_dict"].apply(lambda d: d.get("avg_width", 0))
        else:
            master["dna_dict"] = [{}] * len(master)
            master["depth"] = 0
            master["width"] = 0
        
        # Ensure required columns exist with defaults
        if "id_arch" not in master.columns and "id" in master.columns:
            master["id_arch"] = master["id"]
        if "latency_cpu_ms" not in master.columns:
            master["latency_cpu_ms"] = 0.0
        if "status" not in master.columns:
            master["status"] = "COMPLETED"
        
        data["master"] = master
    else:
        data["master"] = pd.DataFrame()

    return data

data = load_data()
df = data.get("master", pd.DataFrame())

# ===============================
# 3. Sidebar Navigation
# ===============================
st.sidebar.markdown("## 🧬 Neural EvoLab")
st.sidebar.markdown("---")

if df.empty:
    st.error("Database is empty or path is incorrect. Please run training first.")
    st.stop()

selected_view = st.sidebar.radio(
    "Navigation",
    ["🏆 Leaderboard", "🕸️ Lineage Tree", "📈 Pareto Frontier", "🔬 DNA Inspector"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Total Models: {len(df)}")
if "val_acc" in df.columns and not df["val_acc"].isna().all():
    st.sidebar.caption(f"Best Accuracy: {df['val_acc'].max():.2%}")
else:
    st.sidebar.caption("Best Accuracy: N/A")

# ===============================
# 4. View: Leaderboard
# ===============================
if selected_view == "🏆 Leaderboard":
    st.title("🏆 Architecture Hall of Fame")
    
    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    
    # Safe access to metrics with fallbacks
    if "val_acc" in df.columns and not df["val_acc"].isna().all():
        best_model = df.loc[df["val_acc"].idxmax()]
        c1.metric("Top Accuracy", f"{best_model['val_acc']:.2%}", best_model.get('name', 'Unknown'))
    else:
        c1.metric("Top Accuracy", "N/A", "No data")
        best_model = df.iloc[0] if len(df) > 0 else {}
    
    if "latency_cpu_ms" in df.columns and not df["latency_cpu_ms"].isna().all() and (df["latency_cpu_ms"] > 0).any():
        fast_model = df.loc[df[df["latency_cpu_ms"] > 0]["latency_cpu_ms"].idxmin()]
        c2.metric("Best Latency", f"{fast_model['latency_cpu_ms']:.1f} ms", fast_model.get('name', 'Unknown'))
    else:
        c2.metric("Best Latency", "N/A", "No data")
    
    if "params" in df.columns and not df["params"].isna().all():
        c3.metric("Avg Params", f"{df['params'].mean()/1e6:.1f} M")
    else:
        c3.metric("Avg Params", "N/A")
    
    if "status" in df.columns:
        c4.metric("Active Population", len(df[df['status']=='COMPLETED']))
    else:
        c4.metric("Active Population", len(df))

    st.markdown("### 🥇 Top Candidates")
    
    # Table Styling
    display_cols = ["id_arch", "name", "val_acc", "params", "flops", "latency_cpu_ms", "created_at_arch"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    leaderboard = df.sort_values("val_acc", ascending=False)[display_cols].copy()
    leaderboard["params"] = (leaderboard["params"] / 1e6).map("{:.2f}M".format)
    # leaderboard["val_acc"] = leaderboard["val_acc"].map("{:.2%}".format) # Keep numeric for ProgressColumn
    
    st.dataframe(
        leaderboard,
        use_container_width=True,
        column_config={
            "val_acc": st.column_config.ProgressColumn("Accuracy", format="%.2f", min_value=0, max_value=1),
            "created_at_arch": st.column_config.DatetimeColumn("Created")
        },
        hide_index=True
    )

# ===============================
# 5. View: Lineage Tree
# ===============================
elif selected_view == "🕸️ Lineage Tree":
    st.title("🕸️ Evolutionary Lineage")
    
    genealogy = data.get("genealogy", pd.DataFrame())
    
    if genealogy.empty:
        st.warning("No genealogy data found (Generation 0?). Run mutations to see the tree.")
    else:
        # Build Graph
        G = nx.DiGraph()
        
        # Add Nodes
        for _, row in df.iterrows():
            G.add_node(
                row["id_arch"], 
                label=row["name"], 
                acc=row["val_acc"],
                gen=row.get("created_at_arch", 0) # Fallback if no generation tracking
            )
            
        # Add Edges
        for _, row in genealogy.iterrows():
            if row["parent_arch_id"] in G.nodes and row["child_arch_id"] in G.nodes:
                G.add_edge(row["parent_arch_id"], row["child_arch_id"], type=row["mutation_type"])

        # Layout: Use simple hierarchical layout based on generations
        # (Assuming ID correlates roughly with time for Y axis, or we calculate topological generations)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback layout if graphviz missing
            pos = nx.spring_layout(G, seed=42)

        # Convert to Plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            acc = G.nodes[node].get('acc', 0)
            name = G.nodes[node].get('label', 'Unknown')
            node_text.append(f"<b>{name}</b><br>Acc: {acc:.2%}")
            node_color.append(acc)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=False,
                color=node_color,
                size=15,
                colorbar=dict(title="Accuracy")
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                     ))
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Nodes colored by Accuracy. Edges represent Parent->Child mutations.")

# ===============================
# 6. View: Pareto Frontier
# ===============================
elif selected_view == "📈 Pareto Frontier":
    st.title("📈 Efficiency vs Accuracy Trade-off")
    
    # Determine available columns for axes
    available_x = [c for c in ["params", "flops", "latency_cpu_ms"] if c in df.columns]
    available_color = [c for c in ["val_acc", "depth", "width"] if c in df.columns]
    
    if not available_x:
        st.warning("No cost metrics (params, flops, latency) available in dataset.")
    elif "val_acc" not in df.columns or df["val_acc"].isna().all():
        st.warning("No accuracy data available for Pareto analysis.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### Settings")
            x_axis = st.selectbox("X Axis (Cost)", available_x, index=min(len(available_x)-1, 0))
            c_axis = st.selectbox("Color By", available_color if available_color else ["val_acc"], index=0)
        
        with col1:
            # Filter valid data - handle NaN and zero values
            plot_df = df.copy()
            
            # Filter rows where x_axis > 0 and val_acc is not NaN
            mask = (plot_df[x_axis].notna()) & (plot_df[x_axis] > 0) & (plot_df["val_acc"].notna())
            plot_df = plot_df[mask].copy()
            
            if plot_df.empty:
                st.warning(f"No valid data points with {x_axis} > 0 and valid accuracy.")
            else:
                # Ensure size column has valid values
                if "params" in plot_df.columns and plot_df["params"].notna().any():
                    size_col = "params"
                else:
                    plot_df["_size"] = 10
                    size_col = "_size"
                
                fig = px.scatter(
                    plot_df,
                    x=x_axis,
                    y="val_acc",
                    color=c_axis if c_axis in plot_df.columns and plot_df[c_axis].notna().any() else None,
                    size=size_col,
                    hover_name="name" if "name" in plot_df.columns else None,
                    title=f"Accuracy vs {x_axis.replace('_', ' ').title()}",
                    template="plotly_dark",
                    color_continuous_scale="RdYlGn"
                )
                
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Analysis")
        st.info("Models in the **Top-Left** corner are the Pareto Frontier (High Accuracy, Low Cost).")

# ===============================
# 7. View: DNA Inspector
# ===============================
elif selected_view == "🔬 DNA Inspector":
    st.title("🔬 Deep Architecture Inspection")

    # Selector
    c1, c2 = st.columns([1, 2])
    with c1:
        arch_id = st.selectbox("Select Model", df["id_arch"].tolist(), format_func=lambda x: df[df["id_arch"]==x]["name"].values[0])
    
    model_row = df[df["id_arch"] == arch_id].iloc[0]
    
    # 1. DNA Radar Chart 
    dna = model_row.get("dna_dict", {})
    if dna:
        st.subheader("🧬 Genetic Signature")
        
        # Normalize metrics for radar chart
        # We need relative values (0-1) compared to population max
        radar_metrics = {
            "Depth": dna.get("total_depth", 0),
            "Width": dna.get("avg_width", 0),
            "Kernels": dna.get("kernel_diversity", 0),
            "RF Size": dna.get("receptive_field_est", 0)
        }
        
        # Normalize against population max
        r_vals = []
        r_theta = []
        for k, v in radar_metrics.items():
            max_val = df["dna_dict"].apply(lambda d: d.get(
                {"Depth": "total_depth", "Width": "avg_width", "Kernels": "kernel_diversity", "RF Size": "receptive_field_est"}[k]
            , 0)).max()
            norm_val = (v / max_val) if max_val > 0 else 0
            r_vals.append(norm_val)
            r_theta.append(k)
        
        # Close the loop
        r_vals.append(r_vals[0])
        r_theta.append(r_theta[0])
        
        fig = go.Figure(data=go.Scatterpolar(
            r=r_vals,
            theta=r_theta,
            fill='toself',
            name=model_row["name"]
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        
        col_a, col_b = st.columns([1, 1])
        col_a.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.json(dna)

    # 2. Critic Feedback
    st.subheader("🧠 AI Critic Report")
    
    # Fetch critic json
    critic_row = data["archs"][data["archs"]["id"] == arch_id].iloc[0]
    if "critic_json" in critic_row and critic_row["critic_json"]:
        critic = json.loads(critic_row["critic_json"])
        
        score = critic.get("scores", {}).get("overall", 0)
        verdict = critic.get("verdict", "Unknown")
        color = "green" if score > 7 else "orange" if score > 4 else "red"
        
        st.markdown(f"## Verdict: :{color}[{verdict.upper()}] (Score: {score}/10)")
        
        findings = critic.get("findings", [])
        for f in findings:
            st.warning(f"⚠️ {f}")
            
        suggestions = critic.get("suggestions", [])
        for s in suggestions:
            st.success(f"💡 Suggestion: {s.get('action')} ({s.get('reason')})")
    else:
        st.info("No critic evaluation available for this model.")

    # 3. Raw Blueprint
    with st.expander("See Raw Blueprint"):
        st.json(model_row.get("blueprint_json"))
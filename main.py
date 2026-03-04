# def main():
#     print("Hello from hackathon!")


# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# ─────────────────────────────────────────
# 1. PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SafeCity Montgomery | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #111827;
        --accent-red: #ef4444;
        --accent-orange: #f97316;
        --accent-green: #22c55e;
        --text-muted: #94a3b8;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg-primary); color: #f1f5f9; }
    [data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid #1e293b; }
    
    .metric-card {
        background: var(--bg-card); border: 1px solid #1e293b; border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .metric-val { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1; }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-top: 0.5rem; letter-spacing: 0.05em;}
    
    .dispatch-card {
        background: #1a2235; border-left: 4px solid var(--accent-red); padding: 1rem; margin-bottom: 0.8rem; border-radius: 4px;
    }
    .dispatch-card.medium { border-left-color: var(--accent-orange); }
    .dispatch-card.low { border-left-color: var(--accent-green); }
    
    .badge { font-size: 0.7rem; padding: 2px 6px; border-radius: 10px; margin-left: 5px; font-weight: bold; }
    .badge-siren { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .badge-velocity { background: rgba(249, 115, 22, 0.2); color: #fdba74; border: 1px solid #f97316; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 2. DATA LOADING & PROCESSING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/feature_matrix.csv")
    
    # Fill nulls just in case
    df = df.fillna(0)
    
    # Try to load model, fallback to proxy calculation if missing
    if os.path.exists("nuisance_predictor.pkl"):
        model = joblib.load("nuisance_predictor.pkl")
        features = model.feature_names_in_
        # Ensure all columns exist
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df['base_risk'] = model.predict_proba(df[features])[:, 1]
    else:
        # Fallback simulated risk based on your key features
        df['base_risk'] = ((df['nuisance_rate'] * 0.4) + (df['open_violation_rate'] * 0.4) + (df['total_chronic_locations'] / df['total_chronic_locations'].max() * 0.2)).clip(0, 1)

    # Calculate Risk Velocity (30d vs 90d avg)
    df['avg_90d'] = df['complaint_count_90d'] / 3.0
    df['risk_velocity'] = np.where(df['avg_90d'] > 0, df['complaint_count_30d'] / df['avg_90d'], 1.0)
    df['is_emerging'] = (df['risk_velocity'] > 1.5).astype(int)
    
    # Root Cause Diagnosis
    def get_root_cause(row):
        causes = {
            "Chronic Problem Properties": row.get('total_chronic_locations', 0),
            "Environmental Violations": row.get('env_violations', 0),
            "Drainage/Nuisance History": row.get('total_nuisance', 0)
        }
        return max(causes, key=causes.get) if sum(causes.values()) > 0 else "Routine Wear & Tear"
    
    df['root_cause'] = df.apply(get_root_cause, axis=1)
    return df

df = load_data()

# ─────────────────────────────────────────
# 3. SIDEBAR: OPERATIONAL CONTROLS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Command Controls")
    
    st.markdown("#### ⛈️ 1. Weather Stressor")
    weather_event = st.selectbox("Simulate Event", ["Clear (Baseline)", "Heavy Rain", "Flash Flood Watch"])
    w_mult = {"Clear (Baseline)": 1.0, "Heavy Rain": 1.3, "Flash Flood Watch": 1.6}[weather_event]
    
    st.markdown("#### 🚒 2. Dispatch Optimizer")
    crew_capacity = st.slider("Available Dispatch Crews", 5, 50, 15)
    
    st.markdown("#### 🚨 3. Infrastructure Flags")
    show_siren_gaps = st.checkbox("Prioritize Siren Blindspots", value=True)
    show_emerging   = st.checkbox("Prioritize Emerging Hotspots", value=True)

# ─────────────────────────────────────────
# 4. DYNAMIC LOGIC
# ─────────────────────────────────────────
# Apply Weather Multiplier
df['adjusted_score'] = (df['base_risk'] * w_mult).clip(0, 1)

# Categorize
df['risk_label'] = pd.cut(df['adjusted_score'], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])

# Identify Double Jeopardy
df['double_jeopardy'] = ((df['adjusted_score'] >= 0.7) & (df['siren_coverage_gap'] == 1)).astype(int)

# Sort by Action Priority (Risk Score + Modifiers if toggled)
df['priority_score'] = df['adjusted_score']
if show_siren_gaps: df.loc[df['siren_coverage_gap'] == 1, 'priority_score'] += 0.1
if show_emerging:   df.loc[df['is_emerging'] == 1, 'priority_score'] += 0.1

df = df.sort_values("priority_score", ascending=False)

# Get today's workload based on crew capacity
dispatch_list = df.head(crew_capacity)

# ─────────────────────────────────────────
# 5. HEADER & KPIS
# ─────────────────────────────────────────
st.markdown("<h2>SafeCity Montgomery 📍 Operational Dashboard</h2>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ef4444">{(df["risk_label"]=="High").sum()}</div><div class="metric-label">High Risk Zones</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#f97316">{df["double_jeopardy"].sum()}</div><div class="metric-label">Siren Blindspots (Double Jeopardy)</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#3b82f6">{df["is_emerging"].sum()}</div><div class="metric-label">Emerging Hotspots</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#22c55e">{crew_capacity}</div><div class="metric-label">Tasks Dispatched Today</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 6. MAIN MAP & DISPATCH FEED
# ─────────────────────────────────────────
col_map, col_feed = st.columns([2, 1])

with col_map:
    st.markdown("#### 🗺️ Tactical Risk Map")
    
    m = folium.Map(location=[df["cell_lat"].median(), df["cell_lon"].median()], zoom_start=11, tiles="CartoDB dark_matter")
    
    # Plot top 200 high/medium points to keep map performant
    map_data = df[df["adjusted_score"] > 0.4].head(200)
    
    for _, row in map_data.iterrows():
        color = "#ef4444" if row["risk_label"] == "High" else "#f97316"
        radius = 8 if row["double_jeopardy"] == 1 else 5
        
        # HTML Tooltip with Root Cause
        popup_html = f"""
        <div style="font-family:sans-serif; width:220px;">
            <b style="color:{color}; font-size:14px;">Zone {row['grid_cell']}</b><br>
            <hr style="margin:5px 0;">
            <b>Risk Score:</b> {row['adjusted_score']:.2f}<br>
            <b>Root Cause:</b> {row['root_cause']}<br>
            <b>Siren Coverage:</b> {"❌ NO SIREN" if row['siren_coverage_gap']==1 else "✅ Covered"}<br>
            <b>30d Velocity:</b> {row['risk_velocity']:.1f}x vs 90d avg
        </div>
        """
        
        folium.CircleMarker(
            location=[row["cell_lat"], row["cell_lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    st_folium(m, height=500, use_container_width=True)

with col_feed:
    st.markdown(f"#### 📋 Top {crew_capacity} Dispatch Orders")
    
    for _, row in dispatch_list.iterrows():
        lbl = str(row["risk_label"])
        css_cls = "high" if lbl == "High" else "medium"
        
        # Determine specific action based on root cause
        if row['root_cause'] == "Chronic Problem Properties":
            action = "🚔 Code Enforcement: Inspect Chronic Property"
        elif row['root_cause'] == "Environmental Violations":
            action = "🔍 Inspect Open Code Violations"
        else:
            action = "🧹 Sanitation: Clear Drainage / Overgrowth"

        # Build Badges
        badges = ""
        if row["siren_coverage_gap"] == 1:
            badges += '<span class="badge badge-siren">NO SIREN</span>'
        if row["is_emerging"] == 1:
            badges += '<span class="badge badge-velocity">EMERGING</span>'

        st.markdown(f"""
        <div class="dispatch-card {css_cls}">
            <b>Zone {row['grid_cell']}</b> {badges}<br>
            <div style="color:#94a3b8; font-size:0.8rem; margin: 4px 0;">Score: {row['adjusted_score']:.2f} | Priority Rank: {row['priority_score']:.2f}</div>
            <div style="color:#3b82f6; font-size:0.85rem; font-weight:bold;">{action}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# 7. ANALYTICS ROW
# ─────────────────────────────────────────
st.markdown("<hr style='border-color:#1e293b; margin-top:2rem'>", unsafe_allow_html=True)
st.markdown("#### 📊 Risk Diagnostics")
c_bot1, c_bot2, c_bot3 = st.columns(3)

with c_bot1:
    fig_root = px.pie(df[df['risk_label']=='High'], names='root_cause', title="Root Causes (High Risk Zones)",
                      hole=0.5, color_discrete_sequence=["#ef4444", "#f97316", "#eab308"])
    fig_root.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9", margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_root, use_container_width=True)

with c_bot2:
    fig_hist = px.histogram(df, x="adjusted_score", color="risk_label", title="Citywide Risk Distribution",
                            color_discrete_map={"High":"#ef4444", "Medium":"#f97316", "Low":"#22c55e"})
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9", margin=dict(t=30, b=10, l=10, r=10))
    fig_hist.update_xaxes(gridcolor="#1e293b")
    fig_hist.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig_hist, use_container_width=True)

with c_bot3:
    fig_scatter = px.scatter(df, x="complaint_count_30d", y="complaint_count_90d", color="is_emerging", 
                             title="Risk Velocity (30d vs 90d)", color_continuous_scale=["#3b82f6", "#ef4444"])
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9", margin=dict(t=30, b=10, l=10, r=10), coloraxis_showscale=False)
    fig_scatter.update_xaxes(gridcolor="#1e293b", title="Last 30 Days")
    fig_scatter.update_yaxes(gridcolor="#1e293b", title="Last 90 Days")
    st.plotly_chart(fig_scatter, use_container_width=True)
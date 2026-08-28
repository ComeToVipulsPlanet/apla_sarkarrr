import json
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.risk_model import predict_project_risk, train_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NAV-NIRMAAN",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS (FIXED TEXT VISIBILITY & HIGH CONTRAST)
# ============================================================

st.markdown(
    """
    <style>

    /* Force App Background and Main Text Color */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* General Text Visibility Override */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0f172a !important;
    }

    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Metric Cards Fix */
    [data-testid="stMetric"] {
        background: white !important;
        padding: 20px !important;
        border-radius: 14px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0px 3px 12px rgba(15, 23, 42, 0.08) !important;
    }

    [data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 900 !important;
    }

    /* Selectbox Visibility Fix */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }

    .dashboard-header {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a8a
        );
        padding: 28px;
        border-radius: 18px;
        color: white !important;
        margin-bottom: 25px;
        box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.18);
    }

    .dashboard-header h1 {
        color: white !important;
        margin-bottom: 5px;
    }

    .dashboard-header p {
        color: #dbeafe !important;
        margin: 0;
    }

    .section-card {
        background: white !important;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 3px 10px rgba(15, 23, 42, 0.06);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA & GEOJSON
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("projects.csv")


@st.cache_data
def get_geojson():
    url = "https://raw.githubusercontent.com/subhash-bichu/India-State-and-UT-GeoJSON/main/india_state.geojson"

    try:
        with urlopen(url) as response:
            return json.load(response)
    except Exception:
        return None


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ml_model():
    return train_model()


# ============================================================
# PROCESS DATA
# ============================================================

@st.cache_data
def process_data(data):

    data = data.copy()

    data["Budget_Used"] = (
        data["Spent"] / data["Cost"] * 100
    ).round(1)

    data["Financial_Gap"] = (
        data["Budget_Used"] - data["Progress"]
    ).round(1)

    def calculate_risk(row):

        risk = 20

        if row["Financial_Gap"] > 25:
            risk += 45

        elif row["Financial_Gap"] > 15:
            risk += 35

        elif row["Financial_Gap"] > 8:
            risk += 20

        elif row["Financial_Gap"] > 3:
            risk += 10

        if row["Status"] == "Delayed":
            risk += 25

        elif row["Status"] == "At Risk":
            risk += 15

        return min(risk, 100)

    data["Risk_Score"] = data.apply(
        calculate_risk,
        axis=1
    )

    def risk_level(score):

        if score >= 70:
            return "🔴 HIGH"

        elif score >= 40:
            return "🟡 MEDIUM"

        return "🟢 LOW"

    data["Risk"] = data["Risk_Score"].apply(
        risk_level
    )

    data["Potential_Overrun"] = (
        data["Cost"] * 0.15
    ).round(2)

    data["Estimated_Revised_Cost"] = (
        data["Cost"] + data["Potential_Overrun"]
    ).round(2)

    return data


# ============================================================
# ML PREDICTIONS
# ============================================================

def generate_ml_predictions(data, model):

    results = []

    for _, row in data.iterrows():

        risk_name, probability = predict_project_risk(
            model,
            row["Budget_Used"],
            row["Progress"],
            row["Financial_Gap"],
            0 if row["Status"] == "On Track" else 8,
            0 if row["Status"] == "On Track" else 10,
        )

        results.append(
            {
                "ML_Risk": risk_name,
                "ML_Confidence": round(
                    probability,
                    1
                )
            }
        )

    return pd.DataFrame(results)


# ============================================================
# INITIALIZE
# ============================================================

df = load_data()
india_geojson = get_geojson()

model, model_accuracy = load_ml_model()

df = process_data(df)

ml_results = generate_ml_predictions(
    df,
    model
)

df = pd.concat(
    [
        df.reset_index(drop=True),
        ml_results.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏛️ NAV-NIRMAAN")

st.sidebar.caption(
    "Neural Analytics & Vision for National Infrastructure Risk Management"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Command Center",
        "📁 Project Explorer",
        "💰 Cost Analytics",
        "⏰ Time & Risk Prediction",
        "🚨 Early Warning System",
        "📊 Benchmarking",
        "🤖 AI Assistant",
    ],
)

st.sidebar.divider()

st.sidebar.success(
    "Prototype Mode\n\n"
    "Synthetic demonstration dataset"
)

st.sidebar.caption(
    f"ML Model Accuracy: {model_accuracy * 100:.1f}%"
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🏛️ NAV-NIRMAAN</h1>

            <p>
                Neural Analytics & Vision for National Infrastructure Risk Management
                <br>
                Web-Based Integrated Project-Monitoring Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(df)

    delayed = len(
        df[df["Status"] == "Delayed"]
    )

    high_risk = len(
        df[df["Risk_Score"] >= 70]
    )

    budget_risk = df[
        df["Risk_Score"] >= 70
    ]["Cost"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📊 Projects Monitored",
        total
    )

    c2.metric(
        "🔴 Delayed Projects",
        delayed
    )

    c3.metric(
        "⚠️ High Risk Projects",
        high_risk
    )

    c4.metric(
        "💰 Budget at Risk",
        f"₹{budget_risk:,.0f} Cr"
    )

    st.divider()

    # ========================================================
    # STATE INTERACTIVE MAP SECTION
    # ========================================================

    st.subheader(
        "🗺️ State-Wise Infrastructure Map"
    )

    st.caption(
        "States ke naam ab map ke upar bilkul saaf visible hain."
    )

    state_coords = {

        "Maharashtra": {
            "lat": 19.7515,
            "lon": 75.7139
        },

        "Delhi": {
            "lat": 28.7041,
            "lon": 77.1025
        },

        "Gujarat": {
            "lat": 22.2587,
            "lon": 71.1924
        },

        "Karnataka": {
            "lat": 15.3173,
            "lon": 75.7139
        },

        "Tamil Nadu": {
            "lat": 11.1271,
            "lon": 78.6569
        },

        "Uttar Pradesh": {
            "lat": 26.8467,
            "lon": 80.9462
        },

        "West Bengal": {
            "lat": 22.9868,
            "lon": 87.8550
        },

        "Telangana": {
            "lat": 18.1124,
            "lon": 79.0193
        },

        "Kerala": {
            "lat": 10.8505,
            "lon": 76.2711
        },

        "Rajasthan": {
            "lat": 27.0238,
            "lon": 74.2179
        },

        "Madhya Pradesh": {
            "lat": 22.9734,
            "lon": 78.6569
        },

        "Bihar": {
            "lat": 25.0961,
            "lon": 85.3131
        },

        "Assam": {
            "lat": 26.2006,
            "lon": 92.9376
        },

        "Punjab": {
            "lat": 31.1471,
            "lon": 75.3412
        },

        "Haryana": {
            "lat": 29.0588,
            "lon": 76.0856
        },
    }

    state_summary = (
        df.groupby("State")
        .agg(
            Total_Projects=(
                "Project",
                "count"
            ),

            Pending_Works=(
                "Status",
                lambda x: (
                    x != "Completed"
                ).sum()
            ),

            Delayed_Projects=(
                "Status",
                lambda x: (
                    x == "Delayed"
                ).sum()
            ),

            High_Risk_Projects=(
                "Risk_Score",
                lambda x: (
                    x >= 70
                ).sum()
            ),

            Avg_Progress=(
                "Progress",
                "mean"
            ),
        )
        .reset_index()
    )

    state_summary["lat"] = (
        state_summary["State"].map(
            lambda s: state_coords.get(
                s,
                {}
            ).get(
                "lat",
                20.5937
            )
        )
    )

    state_summary["lon"] = (
        state_summary["State"].map(
            lambda s: state_coords.get(
                s,
                {}
            ).get(
                "lon",
                78.9629
            )
        )
    )

    if india_geojson:

        fig_map = px.choropleth_mapbox(
            state_summary,
            geojson=india_geojson,
            featureidkey="properties.NAME_1",
            locations="State",
            color="Pending_Works",
            color_continuous_scale="Reds",
            mapbox_style="open-street-map",
            zoom=3.8,
            center={
                "lat": 22.5937,
                "lon": 78.9629
            },
            opacity=0.5,
            labels={
                "Pending_Works":
                    "Pending Projects"
            },
            hover_data=[
                "Total_Projects",
                "Delayed_Projects",
                "High_Risk_Projects",
            ],
        )

        fig_map.add_trace(
            go.Scattermapbox(
                lat=state_summary["lat"],
                lon=state_summary["lon"],
                mode="text+markers",

                marker=dict(
                    size=10,
                    color="#b91c1c"
                ),

                text=state_summary["State"],

                textposition="top center",

                textfont=dict(
                    size=13,
                    color="#000000",
                    family="Arial, sans-serif",
                ),

                hoverinfo="none",
                showlegend=False,
            )
        )

        fig_map.update_layout(
            margin={
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0
            },
            height=520
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

    # ========================================================
    # STATE SEARCH
    # ========================================================

    state_list = [
        "All States"
    ] + sorted(
        df["State"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_map_state = st.selectbox(
        "📌 State select karein detail dekhne ke liye:",
        options=state_list,
        index=0,
    )

    if selected_map_state != "All States":

        state_df = df[
            df["State"] == selected_map_state
        ]

        pending_df = state_df[
            state_df["Status"] != "Completed"
        ]

        st.markdown(
            f"### 📍 Detailed Pending Works for **{selected_map_state}**"
        )

        ma_col1, ma_col2, ma_col3 = st.columns(3)

        ma_col1.metric(
            "Total State Projects",
            len(state_df)
        )

        ma_col2.metric(
            "Pending/Delayed Works",
            len(pending_df)
        )

        ma_col3.metric(
            "High Risk Projects",
            len(
                state_df[
                    state_df["Risk_Score"] >= 70
                ]
            )
        )

        st.dataframe(
            pending_df[
                [
                    "Project",
                    "Sector",
                    "Progress",
                    "Budget_Used",
                    "Risk_Score",
                    "Status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🚦 Portfolio Status"
        )

        status = (
            df["Status"]
            .value_counts()
            .reset_index()
        )

        status.columns = [
            "Status",
            "Projects"
        ]

        fig = px.pie(
            status,
            names="Status",
            values="Projects",
            hole=0.55,
            title="Project Status Distribution",
        )

        fig.update_layout(
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "⚠️ Risk Distribution"
        )

        risk = (
            df["Risk"]
            .value_counts()
            .reset_index()
        )

        risk.columns = [
            "Risk",
            "Projects"
        ]

        fig = px.bar(
            risk,
            x="Risk",
            y="Projects",
            text="Projects",
            title="Risk Level Distribution",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "🚨 Priority Projects"
    )

    priority = (
        df.sort_values(
            "Risk_Score",
            ascending=False
        )
        .head(7)
    )

    st.dataframe(
        priority[
            [
                "Project",
                "Sector",
                "State",
                "Progress",
                "Budget_Used",
                "Risk_Score",
                "Risk",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PROJECT EXPLORER
# ============================================================

elif page == "📁 Project Explorer":

    st.title(
        "📁 Project Explorer"
    )

    st.caption(
        "Search and inspect individual infrastructure projects."
    )

    search = st.text_input(
        "🔎 Search project"
    )

    sectors = [
        "All"
    ] + sorted(
        df["Sector"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_sector = st.selectbox(
        "🏗️ Sector",
        sectors
    )

    filtered = df.copy()

    if search:

        filtered = filtered[
            filtered["Project"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if selected_sector != "All":

        filtered = filtered[
            filtered["Sector"] == selected_sector
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    if len(filtered) > 0:

        selected = st.selectbox(
            "📌 Select Project",
            filtered["Project"].tolist()
        )

        p = filtered[
            filtered["Project"] == selected
        ].iloc[0]

        st.header(
            f"🏗️ {p['Project']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Physical Progress",
            f"{p['Progress']}%"
        )

        c2.metric(
            "Budget Used",
            f"{p['Budget_Used']}%"
        )

        c3.metric(
            "Risk Score",
            f"{p['Risk_Score']}/100"
        )

        c4.metric(
            "Status",
            p["Status"]
        )

        st.divider()

        st.subheader(
            "📋 Project Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Sector:** {p['Sector']}"
            )

            st.write(
                f"**State:** {p['State']}"
            )

            st.write(
                f"**Approved Cost:** ₹{p['Cost']:,} Cr"
            )

        with col2:

            st.write(
                f"**Expenditure:** ₹{p['Spent']:,} Cr"
            )

            st.write(
                f"**Original Target:** {p['Target']}"
            )

            st.write(
                f"**Expected Completion:** {p['Expected']}"
            )

        st.subheader(
            "📈 Physical Progress"
        )

        progress_value = min(
            max(
                float(p["Progress"]),
                0
            ),
            100
        )

        st.progress(
            progress_value / 100
        )

        st.write(
            f"{progress_value:.1f}% completed"
        )


# ============================================================
# COST ANALYTICS
# ============================================================

elif page == "💰 Cost Analytics":

    st.title(
        "💰 Cost Escalation Analytics"
    )

    st.info(
        "Financial pressure is identified when expenditure grows significantly faster than physical progress."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📊 Budget Utilization"
        )

        fig = px.scatter(
            df,
            x="Progress",
            y="Budget_Used",
            size="Cost",
            hover_name="Project",
            color="Status",
            labels={
                "Progress":
                    "Physical Progress (%)",
                "Budget_Used":
                    "Budget Used (%)",
            },
            title="Progress vs Budget Utilization",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🚨 Top Financial Risk"
        )

        top = (
            df.sort_values(
                "Financial_Gap",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top,
            x="Financial_Gap",
            y="Project",
            orientation="h",
            text="Financial_Gap",
            title="Highest Financial Gaps",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "💡 Cost Escalation Analysis"
    )

    st.dataframe(
        df[
            [
                "Project",
                "Cost",
                "Spent",
                "Budget_Used",
                "Progress",
                "Financial_Gap",
                "Potential_Overrun",
                "Estimated_Revised_Cost",
            ]
        ]
        .sort_values(
            "Financial_Gap",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TIME & RISK PREDICTION
# ============================================================

elif page == "⏰ Time & Risk Prediction":

    st.title(
        "⏰ Predictive Risk Analytics"
    )

    st.info(
        "The prototype combines financial pressure, physical progress and project status to generate an early risk assessment."
    )

    risk_data = (
        df.sort_values(
            "Risk_Score",
            ascending=False
        )
    )

    fig = px.bar(
        risk_data,
        x="Project",
        y="Risk_Score",
        color="Risk",
        text="Risk_Score",
        title="Project Risk Score",
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "🎯 Risk Breakdown"
    )

    selected = st.selectbox(
        "📁 Select Project",
        df["Project"].tolist()
    )

    p = df[
        df["Project"] == selected
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Financial Pressure",
        f"{p['Financial_Gap']:.1f}%"
    )

    c2.metric(
        "Risk Score",
        f"{p['Risk_Score']}/100"
    )

    c3.metric(
        "ML Risk",
        p["ML_Risk"]
    )

    c4.metric(
        "ML Confidence",
        f"{p['ML_Confidence']}%"
    )

    if p["Risk_Score"] >= 70:

        st.error(
            "🔴 HIGH RISK — Immediate monitoring recommended."
        )

    elif p["Risk_Score"] >= 40:

        st.warning(
            "🟡 MEDIUM RISK — Increased monitoring recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — Project currently appears stable."
        )


# ============================================================
# EARLY WARNING SYSTEM
# ============================================================

elif page == "🚨 Early Warning System":

    st.title(
        "🚨 Early Warning & Alert Center"
    )

    st.caption(
        "Projects requiring immediate attention."
    )

    high = df[
        df["Risk_Score"] >= 70
    ]

    medium = df[
        (df["Risk_Score"] >= 40)
        & (df["Risk_Score"] < 70)
    ]

    st.subheader(
        f"🔴 Critical Alerts ({len(high)})"
    )

    if len(high) == 0:

        st.success(
            "No critical alerts detected."
        )

    else:

        for _, p in high.iterrows():

            st.error(
                f"**{p['Project']}** — "
                f"Risk Score {p['Risk_Score']}/100 | "
                f"Budget Used {p['Budget_Used']}% | "
                f"Progress {p['Progress']}%"
            )

    st.divider()

    st.subheader(
        f"🟡 Warning Alerts ({len(medium)})"
    )

    if len(medium) == 0:

        st.success(
            "No medium-risk alerts detected."
        )

    else:

        for _, p in medium.iterrows():

            st.warning(
                f"**{p['Project']}** — "
                f"Risk Score {p['Risk_Score']}/100 | "
                f"Financial Gap {p['Financial_Gap']}%"
            )


# ============================================================
# BENCHMARKING
# ============================================================

elif page == "📊 Benchmarking":

    st.title(
        "📊 Benchmarking & Comparative Analytics"
    )

    sector_summary = (
        df.groupby("Sector")
        .agg(
            Projects=(
                "Project",
                "count"
            ),

            Avg_Risk=(
                "Risk_Score",
                "mean"
            ),

            Avg_Progress=(
                "Progress",
                "mean"
            ),

            Avg_Budget_Used=(
                "Budget_Used",
                "mean"
            ),
        )
        .reset_index()
    )

    sector_summary["Avg_Risk"] = (
        sector_summary["Avg_Risk"]
        .round(1)
    )

    sector_summary["Avg_Progress"] = (
        sector_summary["Avg_Progress"]
        .round(1)
    )

    sector_summary["Avg_Budget_Used"] = (
        sector_summary["Avg_Budget_Used"]
        .round(1)
    )

    st.subheader(
        "🏗️ Sector Performance"
    )

    st.dataframe(
        sector_summary,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        sector_summary,
        x="Sector",
        y="Avg_Risk",
        text="Avg_Risk",
        title="Average Risk by Sector",
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "🤖 AI Assistant":

    st.title(
        "🤖 Project Intelligence Assistant"
    )

    st.caption(
        "Ask questions about an individual infrastructure project."
    )

    st.divider()

    selected = st.selectbox(
        "📁 Select Project",
        df["Project"].tolist()
    )

    p = df[
        df["Project"] == selected
    ].iloc[0]

    st.subheader(
        "📊 Project Snapshot"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Physical Progress",
        f"{p['Progress']}%"
    )

    c2.metric(
        "Budget Used",
        f"{p['Budget_Used']}%"
    )

    c3.metric(
        "Risk Score",
        f"{p['Risk_Score']}/100"
    )

    c4.metric(
        "Status",
        p["Status"]
    )

    st.divider()

    question = st.text_area(
        "💬 Ask about this project",

        placeholder=(
            "Why is this project at risk?\n"
            "Why is this project delayed?\n"
            "What is the financial problem?\n"
            "What action should the officer take?\n"
            "Give me a project summary."
        ),
    )

    if st.button(
        "🔍 Analyze Project",
        use_container_width=True
    ):

        q = question.lower().strip()

        if not q:

            st.warning(
                "Please enter a question first."
            )

        elif (
            "summary" in q
            or "overview" in q
        ):

            st.info(
                f"""
### 📋 Project Summary

**Project:** {p['Project']}  
**Sector:** {p['Sector']}  
**State:** {p['State']}  
**Approved Cost:** ₹{p['Cost']:,} Cr  
**Expenditure:** ₹{p['Spent']:,} Cr  
**Physical Progress:** {p['Progress']}%  
**Budget Utilization:** {p['Budget_Used']}%  
**Financial Gap:** {p['Financial_Gap']}%  
**Risk Score:** {p['Risk_Score']}/100  
**Status:** {p['Status']}
"""
            )

        elif (
            "risk" in q
            or "danger" in q
            or "problem" in q
        ):

            if p["Risk_Score"] >= 70:

                st.error(
                    f"""
### 🔴 High Risk Assessment

**{p['Project']}** currently has a **HIGH risk score of {p['Risk_Score']}/100**.

Budget utilization is **{p['Budget_Used']}%** while physical progress is only **{p['Progress']}%**.

Financial-progress gap: **{p['Financial_Gap']}%**

### ⚠️ Key Risks

• Expenditure is ahead of physical progress.  
• Project implementation requires immediate attention.  
• Potential cost and time overrun exists.

### 🎯 Recommended Actions

1. Conduct immediate milestone review.
2. Identify delay-causing activities.
3. Review contractor performance.
4. Monitor remaining expenditure.
5. Establish corrective milestones.
"""
                )

            elif p["Risk_Score"] >= 40:

                st.warning(
                    f"""
### 🟡 Moderate Risk Assessment

**{p['Project']}** has a risk score of **{p['Risk_Score']}/100**.

Budget utilization is **{p['Budget_Used']}%** against physical progress of **{p['Progress']}%**.

### Recommendation

Increase monitoring frequency and investigate the financial-progress gap.
"""
                )

            else:

                st.success(
                    f"""
### 🟢 Low Risk Assessment

**{p['Project']}** currently has a risk score of **{p['Risk_Score']}/100**.

The project appears to be progressing within acceptable parameters.

### Recommendation

Continue regular monitoring and milestone tracking.
"""
                )

        elif (
            "delay" in q
            or "late" in q
            or "delayed" in q
        ):

            if p["Status"] == "Delayed":

                st.error(
                    f"""
### ⏰ Delay Alert

**{p['Project']}** is currently marked as **DELAYED**.

Physical progress: **{p['Progress']}%**  
Budget utilization: **{p['Budget_Used']}%**

### Possible Causes

• Physical progress is not keeping pace with expenditure.  
• Implementation milestones may be behind schedule.  
• Contractor or resource performance may require review.

### Recommended Action

Conduct a milestone-level review and identify the activities responsible for the delay.
"""
                )

            else:

                st.success(
                    f"""
### ✅ No Major Delay Flag

**{p['Project']}** is currently marked as **{p['Status']}**.

Continue monitoring milestones to prevent future schedule slippage.
"""
                )

        elif (
            "budget" in q
            or "cost" in q
            or "money" in q
            or "financial" in q
            or "expenditure" in q
            or "spent" in q
        ):

            st.info(
                f"""
### 💰 Financial Analysis

**Approved Cost:** ₹{p['Cost']:,} Cr  
**Expenditure:** ₹{p['Spent']:,} Cr  
**Budget Used:** {p['Budget_Used']}%  
**Physical Progress:** {p['Progress']}%  
**Financial Gap:** {p['Financial_Gap']}%  

### Assessment

A positive financial gap indicates that expenditure is ahead of physical progress.

### Estimated Revised Cost

**₹{p['Estimated_Revised_Cost']:,.2f} Cr**
"""
            )

        elif (
            "action" in q
            or "recommend" in q
            or "officer" in q
            or "should" in q
        ):

            if p["Risk_Score"] >= 70:

                st.error(
                    """
### 🚨 Immediate Action Plan

1. Conduct immediate project review.
2. Identify delay-causing activities.
3. Review contractor performance.
4. Verify expenditure against milestones.
5. Create corrective milestones.
6. Increase monitoring frequency.
"""
                )

            elif p["Risk_Score"] >= 40:

                st.warning(
                    """
### ⚠️ Recommended Action Plan

1. Increase monitoring frequency.
2. Review financial-progress gap.
3. Check upcoming milestones.
4. Investigate emerging delays.
5. Track corrective actions.
"""
                )

            else:

                st.success(
                    """
### ✅ Recommended Action Plan

1. Continue regular monitoring.
2. Track project milestones.
3. Monitor expenditure.
4. Maintain current implementation pace.
"""
                )

        else:

            st.info(
                """
### 🤖 Project Intelligence

Try asking:

• Why is this project at risk?  
• Why is this project delayed?  
• What is the financial problem?  
• What action should the officer take?  
• Give me a project summary.
"""
            )

    st.divider()

    st.caption(
        "NAV-NIRMAAN AI Assistant • Prototype Intelligence Engine"
    )
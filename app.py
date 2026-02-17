import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="AutoML Pro", page_icon="🚀", layout="wide")

st.title("🚀 AutoML Pro — Animated Dashboard")

# Sidebar
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Overview", "🤖 Model Training"])
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -----------------------------------
# HOME
# -----------------------------------
if page == "🏠 Home":
    st.markdown("### Upload → Analyze → Train → Download")
    c1, c2, c3 = st.columns(3)
    c1.metric("Auto Cleaning", "✔")
    c2.metric("Model Comparison", "✔")
    c3.metric("Animated Charts", "✔")


# -----------------------------------
# DATA OVERVIEW
# -----------------------------------
if file:
    df = pd.read_csv(file)

    # Auto clean
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    if page == "📊 Data Overview":

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        tabs = st.tabs(["📈 Distribution", "📊 Correlation Heatmap", "📉 Outliers"])

        # -------------------------
        # DISTRIBUTION (Animated)
        # -------------------------
        with tabs[0]:
            if len(numeric_cols) > 0:
                col_select = st.selectbox("Select column", numeric_cols)

                fig = px.histogram(
                    df,
                    x=col_select,
                    nbins=30,
                    title=f"Distribution of {col_select}",
                    template="plotly_dark",
                    animation_frame=None
                )

                fig.update_layout(transition_duration=800)
                st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # CORRELATION HEATMAP
        # -------------------------
        with tabs[1]:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap",
                    aspect="auto"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # OUTLIERS (Interactive Boxplot)
        # -------------------------
        with tabs[2]:
            if len(numeric_cols) > 0:
                col_select = st.selectbox("Outlier column", numeric_cols, key="outlier")

                fig = px.box(
                    df,
                    y=col_select,
                    points="all",
                    title=f"Outlier Detection - {col_select}",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------
    if page == "🤖 Model Training":

        st.subheader("Select Target Column")
        target = st.selectbox("Target", df.columns)

        if target:
            X = df.drop(target, axis=1)
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            if y.dtype == "object" or y.nunique() < 10:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor()
                }

            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                cv = cross_val_score(model, X_scaled, y, cv=5).mean()
                results[name] = cv

            result_df = pd.DataFrame({
                "Model": results.keys(),
                "CV Score": results.values()
            })

            # -------------------------
            # Animated Model Comparison
            # -------------------------
            fig = px.bar(
                result_df,
                x="Model",
                y="CV Score",
                color="Model",
                title="Model Comparison",
                template="plotly_dark"
            )

            fig.update_layout(transition_duration=800)
            st.plotly_chart(fig, use_container_width=True)

            best_model_name = result_df.sort_values("CV Score", ascending=False).iloc[0]["Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

else:
    if page != "🏠 Home":
        st.warning("Upload dataset from sidebar to start.")

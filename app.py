import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AutoML Pro — Production Edition")


# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Overview", "🤖 Model Training"]
)

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# ------------------------------
# HOME
# ------------------------------
if page == "🏠 Home":
    st.markdown("### Upload → Analyze → Train → Download")

    col1, col2, col3 = st.columns(3)
    col1.metric("Auto Cleaning", "✔")
    col2.metric("Model Comparison", "✔")
    col3.metric("PDF Reports", "✔")


# ------------------------------
# MAIN LOGIC
# ------------------------------
if file:

    df = pd.read_csv(file)

    # ------------------------------
    # SAFE MISSING VALUE HANDLING
    # ------------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    numeric_cols = df.select_dtypes(include=np.number).columns

    # ------------------------------
    # DATA OVERVIEW
    # ------------------------------
    if page == "📊 Data Overview":

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        tabs = st.tabs(["📈 Distribution", "📊 Correlation", "📉 Outliers"])

        # Distribution
        with tabs[0]:
            if len(numeric_cols) > 0:
                col_select = st.selectbox("Select Column", numeric_cols)

                fig = px.histogram(
                    df,
                    x=col_select,
                    nbins=30,
                    title=f"Distribution of {col_select}",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, width="stretch")

        # Correlation
        with tabs[1]:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )

                st.plotly_chart(fig, width="stretch")

        # Outliers
        with tabs[2]:
            if len(numeric_cols) > 0:
                col_select = st.selectbox("Outlier Column", numeric_cols, key="outlier")

                fig = px.box(
                    df,
                    y=col_select,
                    points="all",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, width="stretch")

    # ------------------------------
    # MODEL TRAINING
    # ------------------------------
    if page == "🤖 Model Training":

        st.subheader("Select Target Column")
        target = st.selectbox("Target", df.columns)

        if target:

            X = df.drop(columns=[target])
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)

            if X.shape[1] == 0:
                st.error("Not enough features after encoding.")
                st.stop()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            if y.dtype == "object" or y.nunique() < 10:
                problem_type = "Classification"
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier()
                }
            else:
                problem_type = "Regression"
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor()
                }

            st.info(f"{problem_type} Problem Detected")

            results = {}

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)

                    cv_score = cross_val_score(
                        model,
                        X_scaled,
                        y,
                        cv=min(5, len(y))
                    ).mean()

                    results[name] = cv_score
                except:
                    results[name] = 0

            result_df = pd.DataFrame({
                "Model": results.keys(),
                "CV Score": results.values()
            }).sort_values("CV Score", ascending=False)

            fig = px.bar(
                result_df,
                x="Model",
                y="CV Score",
                color="Model",
                title="Model Comparison",
                template="plotly_dark"
            )

            st.plotly_chart(fig, width="stretch")

            best_model_name = result_df.iloc[0]["Model"]
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            st.success(f"Best Model: {best_model_name}")

            # ------------------------------
            # DOWNLOAD MODEL
            # ------------------------------
            buffer = io.BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)

            st.download_button(
                "Download Model (.pkl)",
                buffer,
                "best_model.pkl"
            )

            # ------------------------------
            # PDF REPORT
            # ------------------------------
            def generate_pdf():
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                elements = []
                styles = getSampleStyleSheet()

                elements.append(Paragraph("AutoML Pro Report", styles["Title"]))
                elements.append(Spacer(1, 0.4 * inch))
                elements.append(Paragraph(f"Dataset Shape: {df.shape}", styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph(f"Problem Type: {problem_type}", styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph(f"Best Model: {best_model_name}", styles["Normal"]))
                elements.append(Spacer(1, 0.4 * inch))

                table_data = [result_df.columns.tolist()] + result_df.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                elements.append(table)

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            pdf = generate_pdf()

            st.download_button(
                "Download PDF Report",
                pdf,
                "automl_report.pdf",
                mime="application/pdf"
            )

else:
    if page != "🏠 Home":
        st.warning("Upload a dataset to begin.")

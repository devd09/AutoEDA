import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch


st.set_page_config(page_title="Advanced AutoML System", layout="wide")
st.title("🚀 Advanced AutoML + AutoEDA System")


# PDF GENERATION FUNCTION
def generate_pdf_report(df, result_df, best_model_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AutoML Report", styles["Title"]))
    elements.append(Spacer(1, 0.4 * inch))

    elements.append(Paragraph(f"Dataset Shape: {df.shape}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Best Model: {best_model_name}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Model Comparison:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [result_df.reset_index().columns.tolist()] + result_df.reset_index().values.tolist()
    table = Table(table_data)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return buffer


# FILE UPLOAD
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    # AUTO MISSING VALUE HANDLING
    st.subheader("🧹 Handling Missing Values")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    st.success("Missing values handled automatically!")

    # OUTLIER DETECTION
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        st.subheader("📉 Outlier Detection")
        col_select = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[col_select], ax=ax)
        st.pyplot(fig)

    # CORRELATION HEATMAP
    if len(numeric_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

    # TARGET SELECTION
    st.subheader("🎯 Select Target Column")
    target = st.selectbox("Target Column", df.columns)

    if target:
        X = df.drop(target, axis=1)
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        results = {}

        # CLASSIFICATION
        if y.dtype == "object" or y.nunique() < 10:

            st.subheader("🤖 Classification Problem Detected")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                cv = cross_val_score(model, X_scaled, y, cv=5).mean()
                results[name] = (acc, cv)

            result_df = pd.DataFrame(results, index=["Test Accuracy", "CV Score"]).T
            st.dataframe(result_df)

            best_model_name = result_df["CV Score"].idxmax()
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

        # REGRESSION
        else:

            st.subheader("🤖 Regression Problem Detected")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                r2 = model.score(X_test, y_test)
                cv = cross_val_score(model, X_scaled, y, cv=5).mean()
                results[name] = (r2, cv)

            result_df = pd.DataFrame(results, index=["Test R2", "CV Score"]).T
            st.dataframe(result_df)

            best_model_name = result_df["CV Score"].idxmax()
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

        st.success(f"🏆 Best Model: {best_model_name}")

        # FEATURE IMPORTANCE
        if hasattr(best_model, "feature_importances_"):
            st.subheader("📈 Feature Importance")

            importance = best_model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values("Importance", ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=feat_df.head(10))
            st.pyplot(fig)

        
        # DOWNLOAD MODEL
        st.subheader("💾 Download Trained Model")

        model_buffer = io.BytesIO()
        joblib.dump(best_model, model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download Model (.pkl)",
            data=model_buffer,
            file_name="best_model.pkl"
        )

        # DOWNLOAD PDF REPORT
        st.subheader(" Download PDF Report")

        pdf_buffer = generate_pdf_report(df, result_df, best_model_name)

        st.download_button(
            label="Download Full Report",
            data=pdf_buffer,
            file_name="automl_report.pdf",
            mime="application/pdf"
        )

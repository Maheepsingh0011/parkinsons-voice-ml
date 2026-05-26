import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Page config ──
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide"
)

# ── Title ──
st.title("🧠 Parkinson's Disease Detection")
st.caption("Biomedical Voice Measurement Analysis using Linear SVM")

# ── Load and train model ──
@st.cache_resource
def load_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test, X_test_scaled, y_test, X

model, scaler, X_test, X_test_scaled, y_test, X = load_model()

# ── Metrics row ──
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "Linear SVM")
col2.metric("Accuracy", f"{accuracy*100:.1f}%")
col3.metric("Dataset Size", "195 samples")
col4.metric("Features", "22 voice measurements")

st.divider()

# ── Two columns layout ──
left, right = st.columns(2)

# ── LEFT: Confusion Matrix ──
with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Healthy', "Predicted Parkinson's"],
        yticklabels=['Actual Healthy', "Actual Parkinson's"],
        ax=ax, annot_kws={'size': 14})
    ax.set_title("Confusion Matrix — Test Set", fontweight='bold')
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                target_names=['Healthy', "Parkinson's"],
                output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

# ── RIGHT: Live Prediction ──
with right:
    st.subheader("Live Patient Prediction")
    st.caption("Select a patient from the test set to classify:")

    patient_index = st.slider(
        "Patient sample number",
        0, len(X_test)-1, 0
    )

    sample = X_test.iloc[patient_index].tolist()
    actual = y_test.iloc[patient_index]

    if st.button("Run Prediction", type="primary"):
        sample_df = pd.DataFrame([sample], columns=X.columns)
        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)[0]
        confidence = model.predict_proba(sample_scaled)[0]

        st.divider()

        # Show result
        if prediction == 1:
            st.error(f"🔴 Parkinson's Detected — {max(confidence)*100:.1f}% confidence")
        else:
            st.success(f"🟢 Healthy — {max(confidence)*100:.1f}% confidence")

        # Actual label
        actual_label = "Parkinson's" if actual == 1 else "Healthy"
        st.info(f"Actual label: **{actual_label}**")

        # Confidence bar chart
        fig2, ax2 = plt.subplots(figsize=(5, 2))
        ax2.barh(['Healthy', "Parkinson's"], confidence,
                 color=['#1D9E75', '#D85A30'])
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Prediction Confidence')
        st.pyplot(fig2)

st.divider()

# ── Feature importance ──
st.subheader("Top 10 Most Important Voice Features")
feature_weights = abs(model.coef_[0])
feature_names = X.columns.tolist()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_weights
}).sort_values('Importance', ascending=False).head(10)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.barh(importance_df['Feature'][::-1],
         importance_df['Importance'][::-1],
         color='#7F77DD')
ax3.set_title('Top 10 Feature Importances — Linear SVM Weights')
ax3.set_xlabel('Absolute Weight')
st.pyplot(fig3)
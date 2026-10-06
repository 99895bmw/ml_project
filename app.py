import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, brier_score_loss)

# Page Configuration
st.set_page_config(page_title="Fairness-Aware Credit Default Prediction", layout="wide")
st.title("Fairness-Aware Credit Default Prediction")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Load directly from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    try:
        df = pd.read_excel(url, header=1)
        # Standardize target column name
        df = df.rename(columns={'default payment next month': 'default'})
        # Create a readable label for fairness analysis (1=Male, 2=Female)
        df['SEX_LABEL'] = df['SEX'].map({1: 'Male', 2: 'Female'})
        return df
    except Exception as e:
        st.error(f"Failed to load dataset from UCI: {e}")
        return None

df = load_data()

if df is not None:
    # --- PREPROCESSING ---
    features = [col for col in df.columns if col not in ['ID', 'default', 'SEX_LABEL']]
    X = df[features]
    y = df['default']
    protected_attribute = df['SEX_LABEL']
    
    # Train-test split (stratified to maintain default ratio)
    X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(
        X, y, protected_attribute, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Model Configuration")
    model_choice = st.sidebar.selectbox("Select Base Classifier", ["Random Forest", "Logistic Regression"])
    
    st.sidebar.header("Fairness & Thresholds")
    decision_threshold = st.sidebar.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, 
                                           help="Adjust to balance Precision vs Recall and observe fairness trade-offs.")

    # --- MODEL TRAINING ---
    @st.cache_resource
    def train_models(choice):
        if choice == "Random Forest":
            base_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
        else:
            base_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            
        # Train standard model
        base_model.fit(X_train_scaled, y_train)
        
        # Train calibrated model (Sigmoid/Platt Scaling)
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train_scaled, y_train)
        
        return base_model, calibrated_model

    base_model, calibrated_model = train_models(model_choice)
    
    # Generate Probabilities
    probs_base = base_model.predict_proba(X_test_scaled)[:, 1]
    probs_calib = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply Threshold
    preds_custom = (probs_calib >= decision_threshold).astype(int)

    # --- UI TABS ---
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Model Performance & Calibration", "Fairness Analysis"])

    # TAB 1: Dataset Overview
    with tab1:
        st.subheader("Default of Credit Card Clients Dataset")
        st.write("Target Variable Distribution (Imbalanced):")
        st.bar_chart(df['default'].value_counts())
        
        st.write("Demographic Distribution (Sex):")
        st.bar_chart(df['SEX_LABEL'].value_counts())
        
        st.write("Data Preview:")
        st.dataframe(df.head(10))

    # TAB 2: Model Performance & Calibration
    with tab2:
        col1, col2 = st.columns(2)
        
        def calculate_metrics(y_true, y_prob, y_pred):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall (TPR)": recall_score(y_true, y_pred),
                "F1 Score": f1_score(y_true, y_pred),
                "ROC-AUC": roc_auc_score(y_true, y_prob),
                "Brier Score (Loss)": brier_score_loss(y_true, y_prob)
            }
        
        metrics_base = calculate_metrics(y_test, probs_base, (probs_base >= 0.5).astype(int))
        metrics_calib = calculate_metrics(y_test, probs_calib, (probs_calib >= 0.5).astype(int))
        
        df_metrics = pd.DataFrame([metrics_base, metrics_calib], index=["Base Model", "Calibrated Model"])
        
        with col1:
            st.subheader("Global Metrics (Threshold = 0.5)")
            st.dataframe(df_metrics.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall (TPR)', 'F1 Score', 'ROC-AUC'], color='lightgreen')
                                           .highlight_min(subset=['Brier Score (Loss)'], color='lightgreen'))
            
            st.write("**Brier Score** measures the accuracy of probabilistic predictions (lower is better). Calibration improves this score.")

        with col2:
            st.subheader("Reliability Curves (Calibration)")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            fraction_pos_base, mean_pred_base = calibration_curve(y_test, probs_base, n_bins=10)
            fraction_pos_calib, mean_pred_calib = calibration_curve(y_test, probs_calib, n_bins=10)
            
            ax.plot(mean_pred_base, fraction_pos_base, "s-", label="Base Model")
            ax.plot(mean_pred_calib, fraction_pos_calib, "s-", label="Calibrated Model")
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            
            ax.set_ylabel("Fraction of Positives")
            ax.set_xlabel("Mean Predicted Value")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    # TAB 3: Fairness Analysis
    with tab3:
        st.subheader(f"Fairness Evaluation at Threshold = {decision_threshold}")
        st.write("Analyzing prediction errors between Male and Female demographic groups.")
        
        def calculate_fairness(y_true, y_pred, attr):
            results = []
            for group in attr.unique():
                mask = (attr == group)
                yt = y_true[mask]
                yp = y_pred[mask]
                
                cm = confusion_matrix(yt, yp, labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                selection_rate = (tp + fp) / len(yt)
                
                results.append({
                    "Group": group,
                    "Count": len(yt),
                    "True Positive Rate (Recall)": round(tpr, 3),
                    "False Positive Rate": round(fpr, 3),
                    "Selection/Flagging Rate": round(selection_rate, 3)
                })
            return pd.DataFrame(results)

        fairness_df = calculate_fairness(y_test, preds_custom, attr_test)
        st.dataframe(fairness_df, use_container_width=True)
        
        # Calculate Equality of Opportunity (Difference in TPR)
        tpr_male = fairness_df[fairness_df['Group'] == 'Male']['True Positive Rate (Recall)'].values[0]
        tpr_female = fairness_df[fairness_df['Group'] == 'Female']['True Positive Rate (Recall)'].values[0]
        tpr_diff = abs(tpr_male - tpr_female)
        
        # Calculate Equalized Odds (Difference in FPR)
        fpr_male = fairness_df[fairness_df['Group'] == 'Male']['False Positive Rate'].values[0]
        fpr_female = fairness_df[fairness_df['Group'] == 'Female']['False Positive Rate'].values[0]
        fpr_diff = abs(fpr_male - fpr_female)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.info(f"**Equality of Opportunity Gap (Δ TPR):** {tpr_diff:.3f}\n\n*A smaller gap means the model is equally good at catching actual defaults for both groups.*")
        with col_f2:
            st.warning(f"**Predictive Equality Gap (Δ FPR):** {fpr_diff:.3f}\n\n*A smaller gap means the model does not disproportionately falsely accuse one group of defaulting.*")

        st.divider()
        st.markdown("""
        **How to use this view for your project:**
        1. Notice how the base Random Forest often exhibits different error rates across genders due to historical biases embedded in the training data.
        2. Adjust the threshold slider on the left. See if raising or lowering the threshold to favor precision or recall alters the **Equality of Opportunity Gap**. 
        3. Fair-aware strategies often involve selecting group-specific thresholds (e.g., thresholding females at 0.52 and males at 0.48) so that the TPR gap reaches absolute zero.
        """)

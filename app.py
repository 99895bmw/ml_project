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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    try:
        df = pd.read_excel(url, header=1)
        df = df.rename(columns={'default payment next month': 'default'})
        df['SEX_LABEL'] = df['SEX'].map({1: 'Male', 2: 'Female'})
        return df
    except Exception as e:
        st.error(f"Failed to load dataset from UCI: {e}")
        return None

df = load_data()

if df is not None:
    # --- PREPROCESSING & TRAINING ---
    feature_cols = [col for col in df.columns if col not in ['ID', 'default', 'SEX_LABEL']]
    X = df[feature_cols]
    y = df['default']
    protected_attribute = df['SEX_LABEL']
    
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
    decision_threshold = st.sidebar.slider(
        "Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, 
        help="Adjust to balance Precision vs Recall and observe fairness trade-offs."
    )

    @st.cache_resource
    def train_models(choice):
        if choice == "Random Forest":
            base_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
        else:
            base_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            
        base_model.fit(X_train_scaled, y_train)
        
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train_scaled, y_train)
        
        return base_model, calibrated_model

    base_model, calibrated_model = train_models(model_choice)
    
    # Generate Probabilities
    probs_base = base_model.predict_proba(X_test_scaled)[:, 1]
    probs_calib = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    preds_custom = (probs_calib >= decision_threshold).astype(int)

    # --- UI TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Prediction", 
        "Dataset Overview", 
        "Model Performance & Calibration", 
        "Fairness Analysis"
    ])

    # ==========================================
    # TAB 1: LIVE USER PREDICTION
    # ==========================================
    with tab1:
        st.subheader("Predict Customer Credit Default Risk")
        st.write("Enter customer details below or select a sample profile to evaluate default risk and calibrated probability in real time.")
        
        # Quick Load Preset Buttons
        col_preset1, col_preset2, _ = st.columns([1, 1, 3])
        sample_profile = None
        if col_preset1.button("Load High Risk Sample"):
            sample_profile = {
                'LIMIT_BAL': 20000, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 28,
                'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 3, 'PAY_4': 4, 'PAY_5': 5, 'PAY_6': 6,
                'BILL_AMT1': 18000, 'BILL_AMT2': 19000, 'BILL_AMT3': 18500, 'BILL_AMT4': 19200, 'BILL_AMT5': 19500, 'BILL_AMT6': 20000,
                'PAY_AMT1': 0, 'PAY_AMT2': 1000, 'PAY_AMT3': 0, 'PAY_AMT4': 500, 'PAY_AMT5': 0, 'PAY_AMT6': 0
            }
        if col_preset2.button("Load Low Risk Sample"):
            sample_profile = {
                'LIMIT_BAL': 250000, 'SEX': 2, 'EDUCATION': 1, 'MARRIAGE': 2, 'AGE': 35,
                'PAY_0': -1, 'PAY_2': -1, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -1, 'PAY_6': -1,
                'BILL_AMT1': 5000, 'BILL_AMT2': 3000, 'BILL_AMT3': 4000, 'BILL_AMT4': 2000, 'BILL_AMT5': 1000, 'BILL_AMT6': 1500,
                'PAY_AMT1': 5000, 'PAY_AMT2': 3000, 'PAY_AMT3': 4000, 'PAY_AMT4': 2000, 'PAY_AMT5': 1000, 'PAY_AMT6': 1500
            }

        # User Input Form
        with st.form("prediction_form"):
            st.markdown("#### Customer Demographics & Credit Information")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            val_limit = sample_profile['LIMIT_BAL'] if sample_profile else 50000
            val_sex = sample_profile['SEX'] if sample_profile else 2
            val_edu = sample_profile['EDUCATION'] if sample_profile else 2
            val_marr = sample_profile['MARRIAGE'] if sample_profile else 2
            val_age = sample_profile['AGE'] if sample_profile else 30
            
            limit_bal = c1.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=val_limit, step=10000)
            sex = c2.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", index=0 if val_sex==1 else 1)
            education = c3.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x], index=val_edu-1)
            marriage = c4.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x], index=val_marr-1)
            age = c5.number_input("Age", min_value=18, max_value=100, value=val_age)

            st.markdown("#### Repayment Status (Past 6 Months)")
            st.caption("Status Scale: -1 = Paid in full, 0 = Revolving credit, 1 = Payment delay 1 month, 2 = Payment delay 2 months, etc.")
            
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            pay_opts = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            
            pay_0 = p1.selectbox("Sep (PAY_0)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_0']) if sample_profile else pay_opts.index(0))
            pay_2 = p2.selectbox("Aug (PAY_2)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_2']) if sample_profile else pay_opts.index(0))
            pay_3 = p3.selectbox("Jul (PAY_3)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_3']) if sample_profile else pay_opts.index(0))
            pay_4 = p4.selectbox("Jun (PAY_4)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_4']) if sample_profile else pay_opts.index(0))
            pay_5 = p5.selectbox("May (PAY_5)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_5']) if sample_profile else pay_opts.index(0))
            # Fixed: Changed Oct to Apr
            pay_6 = p6.selectbox("Apr (PAY_6)", options=pay_opts, index=pay_opts.index(sample_profile['PAY_6']) if sample_profile else pay_opts.index(0))

            with st.expander("Bill Amounts & Previous Payments (Optional Detail)", expanded=False):
                b1, b2, b3, b4, b5, b6 = st.columns(6)
                bill_amt1 = b1.number_input("Bill Sep", value=sample_profile['BILL_AMT1'] if sample_profile else 10000)
                bill_amt2 = b2.number_input("Bill Aug", value=sample_profile['BILL_AMT2'] if sample_profile else 10000)
                bill_amt3 = b3.number_input("Bill Jul", value=sample_profile['BILL_AMT3'] if sample_profile else 10000)
                bill_amt4 = b4.number_input("Bill Jun", value=sample_profile['BILL_AMT4'] if sample_profile else 10000)
                bill_amt5 = b5.number_input("Bill May", value=sample_profile['BILL_AMT5'] if sample_profile else 10000)
                # Fixed: Changed Bill Oct to Bill Apr
                bill_amt6 = b6.number_input("Bill Apr", value=sample_profile['BILL_AMT6'] if sample_profile else 10000)

                a1, a2, a3, a4, a5, a6 = st.columns(6)
                pay_amt1 = a1.number_input("Paid Sep", value=sample_profile['PAY_AMT1'] if sample_profile else 1000)
                pay_amt2 = a2.number_input("Paid Aug", value=sample_profile['PAY_AMT2'] if sample_profile else 1000)
                pay_amt3 = a3.number_input("Paid Jul", value=sample_profile['PAY_AMT3'] if sample_profile else 1000)
                pay_amt4 = a4.number_input("Paid Jun", value=sample_profile['PAY_AMT4'] if sample_profile else 1000)
                pay_amt5 = a5.number_input("Paid May", value=sample_profile['PAY_AMT5'] if sample_profile else 1000)
                # Fixed: Changed Paid Oct to Paid Apr
                pay_amt6 = a6.number_input("Paid Apr", value=sample_profile['PAY_AMT6'] if sample_profile else 1000)

            submit_button = st.form_submit_button("Predict Default Risk", use_container_width=True)

        if submit_button:
            # Construct DataFrame with exact feature order
            user_input = pd.DataFrame([{
                'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
                'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
                'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
                'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
            }])[feature_cols]

            # Scale Input
            user_input_scaled = scaler.transform(user_input)

            # Predict Probabilities
            raw_prob = base_model.predict_proba(user_input_scaled)[0, 1]
            calib_prob = calibrated_model.predict_proba(user_input_scaled)[0, 1]
            is_default = calib_prob >= decision_threshold

            st.divider()
            st.subheader("Prediction Results")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                if is_default:
                    st.error("### Risk Level: HIGH")
                    st.write("Decision: **Likely to Default**")
                else:
                    st.success("### Risk Level: LOW")
                    st.write("Decision: **Likely to Pay**")

            with res_col2:
                st.metric(
                    label=f"Calibrated Default Probability", 
                    value=f"{calib_prob * 100:.1f}%"
                )

            with res_col3:
                st.metric(
                    label="Uncalibrated Base Probability", 
                    value=f"{raw_prob * 100:.1f}%",
                    delta=f"{(calib_prob - raw_prob) * 100:.1f}% Calib Shift"
                )

            with res_col4:
                st.metric(
                    label="Active Threshold", 
                    value=f"{decision_threshold * 100:.0f}%"
                )

            # Risk Gauge Progress Bar
            st.write("Risk Gauge:")
            st.progress(float(calib_prob))

            st.info(f"""
            **Explanation:** At the active decision threshold of **{decision_threshold:.2f}**, a customer with a calibrated probability of **{calib_prob:.3f}** is classified as **{'High Risk (Default)' if is_default else 'Low Risk (Non-Default)'}**.
            *(You can change the decision threshold in the left sidebar to observe how strict or lenient the risk cutoff is.)*
            """)

    # ==========================================
    # TAB 2: DATASET OVERVIEW
    # ==========================================
    with tab2:
        st.subheader("Default of Credit Card Clients Dataset")
        st.write("Target Variable Distribution (Imbalanced):")
        st.bar_chart(df['default'].value_counts())
        
        st.write("Demographic Distribution (Sex):")
        st.bar_chart(df['SEX_LABEL'].value_counts())
        
        st.write("Data Preview:")
        st.dataframe(df.head(10))

    # ==========================================
    # TAB 3: MODEL PERFORMANCE & CALIBRATION
    # ==========================================
    with tab3:
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

    # ==========================================
    # TAB 4: FAIRNESS ANALYSIS
    # ==========================================
    with tab4:
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
        
        tpr_male = fairness_df[fairness_df['Group'] == 'Male']['True Positive Rate (Recall)'].values[0]
        tpr_female = fairness_df[fairness_df['Group'] == 'Female']['True Positive Rate (Recall)'].values[0]
        tpr_diff = abs(tpr_male - tpr_female)
        
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
        1. Notice how base models often exhibit different error rates across genders due to historical biases embedded in the dataset.
        2. Adjust the decision threshold slider in the sidebar to observe how changing strictness impacts the Equality of Opportunity Gap.
        """)

import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)

st.set_page_config(
    page_title="Spam Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.title("Configuration")
    
    # This code will load the models
    if not os.path.exists('model'):
        st.error("Model folder missing! Run `train_model.py` first.")
        st.stop()

    model_files = [f for f in os.listdir('model') if f.endswith('.pkl') and f != 'scaler.pkl']
    model_names = [f.replace('.pkl', '').replace('_', ' ').title() for f in model_files]
    
    st.subheader("1. Choose Model")
    selected_model_name = st.selectbox("Select Classifier", model_names)

    st.subheader("2. Input Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.markdown("---")
    st.info("**Tip:** Use `sample_test_data.csv` generated during training for a quick test.")

st.title("Email Spam Detection (Binary Classification)")
st.markdown(f"### Current Model: **{selected_model_name}**")

try:
    model_path = os.path.join('model', selected_model_name.replace(' ', '_').lower() + '.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    has_labels = 'target' in data.columns
    if has_labels:
        X_test = data.drop('target', axis=1)
        y_test = data['target']
    else:
        X_test = data

    st.subheader("Input Data Snapshot")
    col1, col2, col3 = st.columns([3, .5, .5])
    with col1:
        st.dataframe(data.head(10), width="stretch")
        st.caption(f"Showing first 10 rows. Total Shape: {data.shape}")
    with col2:
        st.markdown("**Dataset Stats**")
        st.metric("Total Instances (Rows)", data.shape[0])
        st.metric("Total Features (Cols)", X_test.shape[1])
    with col3:
        if has_labels:
            st.markdown("**Class Distribution**")
            dist_df = y_test.value_counts().reset_index()
            dist_df.columns = ['Class', 'Count']
            dist_df['Label'] = dist_df['Class'].map({1: 'Spam', 0: 'Not Spam'})
                
            fig = px.pie(dist_df, values='Count', names='Label', 
                        color='Label', color_discrete_map={'Spam':'red', 'Not Spam':'green'},
                        hole=0.4)
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Run Prediction")
    if selected_model_name in ["Logistic Regression", "Knn"]:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    if st.button("Analyze Emails", type="primary"):
        with st.spinner('Scanning emails...'):
            prediction = model.predict(X_input)
            prediction_prob = model.predict_proba(X_input)[:, 1]

            results_df = data.copy()
            results_df['Prediction'] = prediction
            results_df['Confidence'] = prediction_prob
            results_df['Label'] = results_df['Prediction'].map({1: 'SPAM', 0: 'HAM'})

            spam_count = sum(prediction)
            ham_count = len(prediction) - spam_count
                
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed", len(prediction))
            c2.metric("Detected Spam", spam_count, delta_color="inverse")
            c3.metric("Detected Ham", ham_count)

            st.markdown("### Detailed Prediction Results")
                
            def highlight_spam(val):
                color = '#ffcccc' if val == 'SPAM' else '#ccffcc'
                return f'background-color: {color}'

            st.dataframe(
                results_df.style.map(highlight_spam, subset=['Label']),
                width="stretch"
            )

            if has_labels:
                st.divider()
                st.subheader("Model Performance Metrics")
                    
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_test, prediction):.2%}")
                m2.metric("AUC Score", f"{roc_auc_score(y_test, prediction_prob):.3f}")
                m3.metric("Precision", f"{precision_score(y_test, prediction):.3f}")
                m4.metric("Recall", f"{recall_score(y_test, prediction):.3f}")
                m5.metric("F1 Score", f"{f1_score(y_test, prediction):.3f}")
                m6.metric("MCC", f"{matthews_corrcoef(y_test, prediction):.3f}")

                col_graph1, col_graph2 = st.columns(2)
                    
                with col_graph1:
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, prediction)
                        
                    z = cm
                    x = ['Predicted Ham', 'Predicted Spam']
                    y = ['Actual Ham', 'Actual Spam']
                        
                    annotations = []
                    for i in range(2):
                        for j in range(2):
                            annotations.append(dict(x=x[j], y=y[i], text=str(z[i][j]),
                                                 showarrow=False, font=dict(color='white' if z[i][j] > cm.max()/2 else 'black')))

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=z, x=x, y=y, colorscale='Blues', showscale=False
                    ))
                    fig_cm.update_layout(annotations=annotations, margin=dict(t=0, b=0, l=0, r=0), height=500)
                    st.plotly_chart(fig_cm, width="stretch")

else:
    st.markdown("""
    <div style='text-align: center; padding: 150px;'>
        <p>Upload a CSV file in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
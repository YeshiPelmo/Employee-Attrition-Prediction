import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model artifacts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl') 
best_accuracies = joblib.load('best_accuracies.pkl')    
best_model_name = max(best_accuracies, key=best_accuracies.get)
best_accuracy = best_accuracies[best_model_name]


# Page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard - Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .best-model-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👥 HR Analytics Dashboard - Employee Attrition Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="best-model-badge">This interactive tool will enable HR teams to predict employee attrition and take proactive measures.</div>', unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('IBM_HR_Analytics_Employee_attrition.csv')
    return df

df = load_data()


st.markdown("### 🔑 Key Metrics Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Employees", value=f"{df.shape[0]:,}")

with col2:
    attrition_rate = df["Attrition"].value_counts(normalize=True).get("Yes", 0)
    st.metric("Attrition Rate", value=f"{attrition_rate:.1%}")

with col3:
    st.metric("Best Model", value=best_model_name)

with col4:
    st.metric("Best Accuracy", value=0.8776, delta=f"{best_accuracy:.2%} over baseline", delta_color="normal")

# Create tabs
# tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔍 Your Analysis", "💰 Compensation", "🤖 SVM Sigmoid Prediction", "📋 Model Comparison"])

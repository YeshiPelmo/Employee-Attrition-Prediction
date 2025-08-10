import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model artifacts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl') 
score = joblib.load('best_accuracy.pkl')
best_model_name = "SVM (Sigmoid)"

#Load encoders dictionary
encoders = joblib.load('encoders.pkl')


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
    st.metric("🧑‍💼Total Employees", value=f"{df.shape[0]:,}")

with col2:
    attrition_rate = df["Attrition"].value_counts(normalize=True).get("Yes", 0)
    st.metric("📉Attrition Rate", value=f"{attrition_rate:.1%}")

with col3:
    st.metric("🏆Best Model", value=best_model_name)

with col4:
    st.metric("🎯Best Accuracy", value=f"{score:.2%}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🤖 SVM Sigmoid Prediction", "📈 Data Overview", "🔍 Analysis"])

with tab1:
    st.subheader("🤖 SVM Sigmoid Prediction")
    
    # Top 10 features
    top_features = [
    'MonthlyIncome','Age','TotalWorkingYears', 'HourlyRate', 'DistanceFromHome', 'YearsAtCompany',
    'YearsWithCurrManager', 'WorkLifeBalance',
    'YearsSinceLastPromotion',  'PercentSalaryHike'
    ]

    # Geting all feature names in the correct order (excluding target)
    all_features = [col for col in df.columns if col != "Attrition"]

    st.title("Employee Attrition Prediction")

    feature_inputs = {}

    #Split top features into two lists
    left_features = top_features[:5]
    right_features = top_features[5:]

    #Creating two columns for feature inputs
    col1, col2 = st.columns(2)

    # Widgets for top 10 features
    with col1:
        for col in left_features:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                options = df[col].unique().tolist()
                feature_inputs[col] = st.selectbox(f"{col}", options)
            else:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                feature_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val)

    with col2:
        for col in right_features:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                options = df[col].unique().tolist()
                feature_inputs[col] = st.selectbox(f"{col}", options)
            else:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                feature_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val)

    # Default values for other features
    for col in all_features:
        if col not in feature_inputs:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                feature_inputs[col] = df[col].mode()[0]  # Most common value
            else:
                feature_inputs[col] = int(df[col].mean())  # Mean value, rounded to int

    # Prediction button
    if st.button("Predict Attrition"):
        input_df = pd.DataFrame([[feature_inputs[col] for col in all_features]], columns=all_features)
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.success("Prediction: Employee is likely to leave (Attrition = Yes)")
        else:
            st.info("Prediction: Employee is likely to stay (Attrition = No)")


with tab2:
    st.subheader("📊 Dataset Summary Statistics")
    st.write(df.describe())
    st.subheader("📝 First 50 Rows of the Dataset")
    st.write(df.head(50))
    
    # Visualization 1: Key Features Distributions by Attrition
   
    fig1 = plt.figure(figsize=(16, 6))  
    key_features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears','JobSatisfaction', 'WorkLifeBalance', 'PerformanceRating']
    for i, feature in enumerate(key_features):
        plt.subplot(2, 4, i+1)
        for attrition in df['Attrition'].unique():
            data = df[df['Attrition'] == attrition][feature]
            plt.hist(data, alpha=0.7, label=attrition, bins=20)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{feature} Distribution by Attrition', fontsize=13)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # Preprocessing
    st.subheader("🔄 Data Preprocessing")
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    import joblib
    joblib.dump(encoders, 'encoders.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    #4. Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_encoded.loc[X_train.index])
    X_test_scaled = scaler.transform(X_encoded.loc[X_test.index])
   
    
    # For Streamlit output 
    st.write(f"Original classes: {label_encoder.classes_}")
    st.write(f"Encoded classes: {np.unique(y_encoded)}")
    st.write(f"Categorical columns encoded: {categorical_cols}")
    st.write(f"Training set size: {X_train.shape}")
    st.write(f"Test set size: {X_test.shape}")

    # --- KNN Validation Curve for Streamlit ---
    st.subheader("📉 KNN Validation Curve: Finding Optimal k")
    k_range = np.arange(1, 31, 2)
    st.write(f"Testing k values: {k_range}")

    train_scores, val_scores = validation_curve(
        KNeighborsClassifier(), X_train_scaled, y_train,
        param_name='n_neighbors', param_range=k_range,
        cv=5, scoring='accuracy', n_jobs=1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    optimal_k_idx = np.argmax(val_mean)
    optimal_k = k_range[optimal_k_idx]

    st.write(f"**Optimal k value:** {optimal_k}")
    st.write(f"**Best validation accuracy:** {val_mean[optimal_k_idx]:.4f}")

    fig = plt.figure(figsize=(6, 4))
    plt.plot(k_range, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(k_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(k_range, val_mean, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(k_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.axvline(x=optimal_k, color='green', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN Validation Curve: Finding Optimal k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Additional analysis: Different distance metrics for optimal k
    st.write(f"Testing Different Distance Metrics for k={optimal_k}...")

    distance_metrics = ['euclidean', 'manhattan', 'minkowski']
    distance_scores = {}

    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric, n_jobs=1)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=1)
        distance_scores[metric] = scores.mean()
        st.write(f"Distance metric '{metric}': CV Accuracy = {scores.mean():.4f}")

    best_metric = max(distance_scores, key=distance_scores.get)
    st.write(f"**Best distance metric:** {best_metric}")
            

with tab3:
    # Fit models and get predictions/accuracies for all columns
    test_accuracies = {}
    predictions = {}
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=1),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Polynomial)': SVC(kernel='poly', degree=3, random_state=42),
        'SVM (Sigmoid)': SVC(kernel='sigmoid', random_state=42),
        'SVM (RBF-Tuned)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'K-Nearest Neighbors (Default)': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        f'K-Nearest Neighbors (Optimal k={optimal_k})': KNeighborsClassifier(
            n_neighbors=optimal_k, metric=best_metric, n_jobs=1
        )
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        test_accuracies[name] = accuracy_score(y_test, y_pred)
    st.subheader("🔍 Analysis")
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.2, 1, 1])
    
    # Column 1
    with col1:
        st.markdown("### 🔝 Top 5 Performing Models")
        sorted_models = sorted(test_accuracies.items(), key=lambda x: x[1], reverse=True)
        for i, (name, accuracy) in enumerate(sorted_models[:5], 1):
            st.write(f"{i}. **{name}**: {accuracy * 100:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"### Confusion Matrix for Best Model ({best_model_name})")
        cm = confusion_matrix(y_test, predictions[best_model_name])
        fig_cm = plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.markdown("<br>", unsafe_allow_html=True)

    # Column 2
    with col2:
        st.subheader("📊 5-Fold Cross Validation Results")
        cv_scores = {}
        cv_folds = 5
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy', n_jobs=1)
            cv_scores[name] = scores
        fig = plt.figure(figsize=(10, 5))  # Consistent figure size
        model_names = list(cv_scores.keys())
        cv_means = [cv_scores[name].mean() for name in model_names]
        cv_stds = [cv_scores[name].std() for name in model_names]
        plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        plt.xlabel('Models', fontsize=16)
        plt.ylabel('Cross Validation Accuracy', fontsize=16)
        plt.title('5-Fold Cross Validation Results', fontsize=18)
        plt.xticks(range(len(model_names)), model_names, rotation=30, ha='right', fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.write("🧬Feature Importance Analysis...")
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred
            test_accuracies[name] = accuracy_score(y_test, y_pred)
            trained_models[name] = model #store trained model
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            top_n = 10
            top_features = feature_importance.head(top_n)
            fig_feat = plt.figure(figsize=(8, 5))
            plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
            plt.gca().invert_yaxis()
            plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_feat)
        st.markdown("<br>", unsafe_allow_html=True)

    # Column 3
    with col3:
        st.subheader("📊 Model Performance Metrics Matrix")
        classification_reports = {}
        performance_metrics = {}
        predictions = {}
        for name, model in models.items():
            # Fit and predict for each model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred
            report = classification_report(y_test, y_pred,
                                          target_names=label_encoder.classes_,
                                          output_dict=True)
            classification_reports[name] = report
            performance_metrics[name] = {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
        metrics_df = pd.DataFrame(performance_metrics).T
        st.write("Performance Metrics Comparison:")
        fig = plt.figure(figsize=(10, 5))  # Consistent figure size
        x = np.arange(len(metrics_df))
        width = 0.2
        plt.bar(x - 1.5*width, metrics_df['accuracy'], width, label='Accuracy', alpha=0.8)
        plt.bar(x - 0.5*width, metrics_df['precision'], width, label='Precision', alpha=0.8)
        plt.bar(x + 0.5*width, metrics_df['recall'], width, label='Recall', alpha=0.8)
        plt.bar(x + 1.5*width, metrics_df['f1_score'], width, label='F1-Score', alpha=0.8)
        plt.xlabel('Models', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        plt.title('Model vs Weighted Avg Of Various Metrics', fontsize=18)
        plt.xticks(x, metrics_df.index, rotation=30, ha='right', fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.write("🦾Finding optimal K using Elbow Method...")
        k_range = range(1, 20)
        inertias = []
        fit_times = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            start_time = time.time()
            kmeans.fit(X_train_scaled)
            end_time = time.time()
            inertias.append(kmeans.inertia_)
            fit_times.append(end_time - start_time)
        elbow_k = 2  
        elbow_score = inertias[elbow_k - 1]
        fig_kmeans, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(k_range, inertias, 'o-', color='blue', label='distortion score')
        ax1.set_xlabel('k')
        ax1.set_ylabel('distortion score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title("Distortion Score Elbow for KMeans Clustering")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=elbow_k, linestyle='--', color='black')
        ax1.text(elbow_k + 0.1, elbow_score + 50, f'elbow at $k$ = {elbow_k}, score = {elbow_score:.2f}',
            fontsize=10, style='italic', color='black')
        ax2 = ax1.twinx()
        ax2.plot(k_range, fit_times, 'o--', color='lightgreen', label='fit time')
        ax2.set_ylabel('fit time (seconds)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        plt.tight_layout()
        st.pyplot(fig_kmeans)
        st.markdown("<br>", unsafe_allow_html=True)
    
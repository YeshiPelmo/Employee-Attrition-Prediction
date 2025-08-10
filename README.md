# Employee-Attrition-Prediction
🚀 Deployed app is here: https://hr-analytics-employee-attrition.streamlit.app/
This interactivve web application predicts employee attrition using machine learning techniques based on the IMB HR Analytics Employee Attrition Dataset from Kaggle. As an HR professional, the goal of this project is to provide a dashboard that provides HR professionals with data-drivern insights to undertand the underlying causes to atttion and predict employee turnover patterns. This will aid in resourcemanagment through workforce planning, reduce cost through preventative measures and most importantly, assist in strategic decision making. 

Features:
Data Exploration: Summary statistics, target variable distribution, missing values checks and accompanying visualizations. 
Preprocessing: Categorical encoding, standarization and train/test split of 70:30 (stratified).
Modeling: Multiple classifiers: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM (various kernels), KNN (default and optimal k) Hyperparameter tuning for KNN and Cross-validation and performance metrics.
Deployment Artifacts: Best model, scaler, label encoder, encoders dictionary, and best accuracy saved as .pkl files.
Streamlit App: Dynamic input widgets for top 10 features, Data overview and Analysis tabs for user friendliness.

How to Run:
Install  streamlit and the requirements. txt.
Start the Streamlit app:streamlit run app.py.
Interact:
Enter employee data for prediction.
Explore analysis and visualizations in dedicated tabs.

Files:
app.py — Streamlit app.
Project2_Q1_YP-final.ipynb — Jupyter notebook with analysis and model training.
IBM_HR_Analytics_Employee_attrition.csv — Dataset.
encoders.pkl, scaler.pkl, label_encoder.pkl, best_model.pkl, best_accuracy.pkl — Saved artifacts.
requirements.txt — Python dependencies.

Requirements:
Python 3.8+.
See requirements.txt for all packages (pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, plotly).

Usage Notes:
All preprocessing steps and encoders are saved for reproducible predictions.
Visualizations are sized for clarity and compactness in the app.
Only top features are shown for user input and analysis.
No automatic updates—manual edits required for further changes.







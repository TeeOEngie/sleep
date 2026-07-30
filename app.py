import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(
    page_title="Sleep Quality Prediction",
    page_icon="zzz",
    layout="wide"
)

st.title("Sleep Quality Prediction System")

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('knn_model.pkl', 'rb') as f:
            models['KNN'] = pickle.load(f)
        with open('decision_tree_model.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f:
            models['SVM'] = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            models['K-Means'] = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f:
            models['Regression'] = pickle.load(f)
        with open('random_forest_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        return models, scaler, label_encoders, target_encoder, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

models, scaler, label_encoders, target_encoder, feature_columns = load_models()

page = st.sidebar.radio(
    "Menu",
    ["Home", "Prediction", "Models Info", "Developer"]
)

if page == "Home":
    st.header("Home")
    st.write("Welcome to Sleep Quality Prediction System")
    st.write("This system uses 6 machine learning models to predict sleep quality.")
    
    st.subheader("Models Used:")
    st.write("1. K-Nearest Neighbor (KNN)")
    st.write("2. Decision Tree")
    st.write("3. SVM (Support Vector Machine)")
    st.write("4. K-Means Clustering")
    st.write("5. Logistic Regression")
    st.write("6. Random Forest (Ensemble)")

elif page == "Prediction":
    st.header("Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=60, value=25)
        bedtime_screen = st.number_input("Bedtime Screen Time (minutes)", min_value=0, max_value=300, value=60)
        total_screen = st.number_input("Total Daily Screen Time (hours)", min_value=0.0, max_value=15.0, value=5.0)
        doomscroll_sessions = st.number_input("Doomscroll Sessions per Night", min_value=0, max_value=10, value=2)
        avg_doomscroll = st.number_input("Avg Doomscroll Session (minutes)", min_value=0, max_value=100, value=20)
        sleep_hours = st.number_input("Sleep Hours per Night", min_value=4.0, max_value=10.0, value=7.0)
        sleep_latency = st.number_input("Sleep Latency (minutes)", min_value=0, max_value=100, value=20)
        night_wakeups = st.number_input("Night Wakeups", min_value=0, max_value=10, value=2)
        caffeine = st.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=600, value=100)
        anxiety = st.slider("Anxiety Score (1-10)", 1, 10, 5)
        stress = st.slider("Stress Score (1-10)", 1, 10, 5)
    
    with col2:
        sleep_quality_score = st.slider("Self-rated Sleep Quality (1-10)", 1, 10, 5)
        daytime_fatigue = st.slider("Daytime Fatigue (1-10)", 1, 10, 5)
        news_apps = st.number_input("Number of News Apps", min_value=0, max_value=10, value=2)
        phone_checks = st.number_input("Phone Checks per Night", min_value=0, max_value=10, value=3)
        exercise = st.number_input("Exercise (minutes/day)", min_value=0, max_value=150, value=30)
        days_detox = st.number_input("Days Since Digital Detox", min_value=0, max_value=365, value=30)
        weekly_sleep_debt = st.number_input("Weekly Sleep Debt (hours)", min_value=0.0, max_value=30.0, value=5.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        occupation = st.selectbox("Occupation", ["Student", "Employed Full-time", "Employed Part-time", "Unemployed"])
        country = st.selectbox("Country/Region", ["United States", "India", "United Kingdom", "Canada", "Australia", "Germany", "Brazil", "Philippines", "Nigeria", "UAE"])
        primary_device = st.selectbox("Primary Device at Night", ["Smartphone", "Tablet", "Laptop", "TV"])
        bedtime_routine = st.selectbox("Bedtime Routine", ["No Fixed Routine", "Reading", "Meditation/Journaling", "Scrolling Social Media", "Watching Videos"])
    
    if st.button("Predict"):
        if models is not None:
            input_data = {
                'age': age,
                'bedtime_screen_time_minutes': bedtime_screen,
                'total_daily_screen_time_hours': total_screen,
                'doomscroll_sessions_per_night': doomscroll_sessions,
                'avg_doomscroll_session_minutes': avg_doomscroll,
                'sleep_hours_per_night': sleep_hours,
                'sleep_latency_minutes': sleep_latency,
                'number_of_night_wakeups': night_wakeups,
                'caffeine_intake_mg_per_day': caffeine,
                'anxiety_score': anxiety,
                'stress_score': stress,
                'sleep_quality_score': sleep_quality_score,
                'daytime_fatigue_score': daytime_fatigue,
                'number_of_news_apps_used': news_apps,
                'phone_checks_per_night': phone_checks,
                'exercise_minutes_per_day': exercise,
                'days_since_last_digital_detox': days_detox,
                'weekly_sleep_debt_hours': weekly_sleep_debt,
                'gender': gender,
                'occupation_status': occupation,
                'country_region': country,
                'primary_device_used_at_night': primary_device,
                'bedtime_routine_type': bedtime_routine
            }
            
            input_df = pd.DataFrame([input_data])
            
            for col in label_encoders.keys():
                if col in input_df.columns:
                    le = label_encoders[col]
                    input_df[col] = input_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
            
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_input = input_df[feature_columns].fillna(0)
            X_input_scaled = scaler.transform(X_input)
            
            st.subheader("Prediction Results")
            
            results = {}
            
            pred = models['KNN'].predict(X_input_scaled)[0]
            results['1. K-Nearest Neighbor'] = target_encoder.inverse_transform([pred])[0]
            
            pred = models['Decision Tree'].predict(X_input)[0]
            results['2. Decision Tree'] = target_encoder.inverse_transform([pred])[0]
            
            pred = models['SVM'].predict(X_input_scaled)[0]
            results['3. SVM'] = target_encoder.inverse_transform([pred])[0]
            
            cluster = models['K-Means'].predict(X_input_scaled)[0]
            results['4. K-Means'] = f"Cluster {cluster}"
            
            pred = models['Regression'].predict(X_input_scaled)[0]
            results['5. Regression'] = target_encoder.inverse_transform([pred])[0]
            
            pred = models['Random Forest'].predict(X_input)[0]
            results['6. Random Forest'] = target_encoder.inverse_transform([pred])[0]
            
            for model_name, prediction in results.items():
                st.write(f"**{model_name}**: {prediction}")
            
            predictions_list = [v for v in results.values() if v in ['Good', 'Fair', 'Poor']]
            if predictions_list:
                most_common = max(set(predictions_list), key=predictions_list.count)
                st.subheader("Summary")
                st.write(f"Most common prediction: **{most_common}**")

elif page == "Models Info":
    st.header("Machine Learning Models Information")
    
    st.subheader("1. K-Nearest Neighbor (KNN)")
    st.write("Classification algorithm that predicts based on the K nearest data points.")
    
    st.subheader("2. Decision Tree")
    st.write("Tree-based model that splits data based on feature values to make predictions.")
    
    st.subheader("3. SVM (Support Vector Machine)")
    st.write("Finds the optimal hyperplane to separate different classes of data.")
    
    st.subheader("4. K-Means Clustering")
    st.write("Unsupervised learning algorithm that groups data into K clusters.")
    
    st.subheader("5. Logistic Regression")
    st.write("Statistical model for binary and multi-class classification problems.")
    
    st.subheader("6. Random Forest (Ensemble)")
    st.write("Ensemble method combining multiple decision trees for better accuracy.")

elif page == "Developer":
    st.header("Developer Information")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image("profile.jpg", caption="Developer Photo", use_container_width=True)
        except:
            st.write("Photo not found. Please upload profile.jpg")
    
    with col2:
        st.subheader("Details")
        st.write("**Student ID:** 6XXXXXXXXX")
        st.write("**Name:** Your Name Here")
        st.write("**Class Group:** Group XX")
        st.write("**Email:** your.email@example.com")
    
    st.subheader("About This Project")
    st.write("This project uses machine learning to predict sleep quality based on lifestyle habits and doomscrolling behavior.")
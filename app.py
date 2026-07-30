import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Sleep Quality Prediction",
    page_icon="",
    layout="wide"
)

# หัวข้อเว็บ
st.title("😴 ระบบทำนายคุณภาพการนอนหลับ")
st.markdown("---")

# โหลดโมเดลและ preprocessors
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('knn_model.pkl', 'rb') as f:
            models['K-nearest neighbor'] = pickle.load(f)
        with open('decision_tree_model.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f:
            models['SVM'] = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            models['K-mean'] = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f:
            models['Regression'] = pickle.load(f)
        with open('random_forest_model.pkl', 'rb') as f:
            models['Ensemble (Random Forest)'] = pickle.load(f)
        
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
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None, None, None, None

models, scaler, label_encoders, target_encoder, feature_columns = load_models()

# แสดงข้อมูลผู้พัฒนา
st.sidebar.header("👨‍💻 ข้อมูลผู้พัฒนา")
st.sidebar.info("""
**รหัสนักศึกษา:** 6XXXXXXXXX

**ชื่อ-นามสกุล:** นาย/นางสาว XXXXXXXXXX

**หมู่เรียน:** หมู่เรียนที่ XX
""")

st.sidebar.markdown("---")
st.sidebar.header(" โมเดลที่ใช้")
st.sidebar.write("1. K-nearest neighbor")
st.sidebar.write("2. Decision Tree")
st.sidebar.write("3. SVM")
st.sidebar.write("4. K-mean")
st.sidebar.write("5. Regression")
st.sidebar.write("6. Ensemble (Random Forest)")

# ส่วนกรอกข้อมูล
st.header("📝 กรอกข้อมูลเพื่อทำนาย")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (Age)", min_value=15, max_value=60, value=25)
    bedtime_screen = st.number_input("เวลาใช้หน้าจอก่อนนอน (นาที)", min_value=0, max_value=300, value=60)
    total_screen = st.number_input("เวลาใช้หน้าจอรวมต่อวัน (ชั่วโมง)", min_value=0.0, max_value=15.0, value=5.0)
    doomscroll_sessions = st.number_input("จำนวนครั้งที่ Doomscroll ต่อคืน", min_value=0, max_value=10, value=2)
    avg_doomscroll = st.number_input("เวลาเฉลี่ยต่อ session Doomscroll (นาที)", min_value=0, max_value=100, value=20)
    sleep_hours = st.number_input("จำนวนชั่วโมงการนอนต่อคืน", min_value=4.0, max_value=10.0, value=7.0)
    sleep_latency = st.number_input("เวลาที่ใช้กว่าจะหลับ (นาที)", min_value=0, max_value=100, value=20)
    night_wakeups = st.number_input("จำนวนครั้งที่ตื่นกลางดึก", min_value=0, max_value=10, value=2)
    caffeine = st.number_input("ปริมาณคาเฟอีนต่อวัน (mg)", min_value=0, max_value=600, value=100)
    anxiety = st.slider("คะแนนความวิตกกังวล (1-10)", 1, 10, 5)
    stress = st.slider("คะแนนความเครียด (1-10)", 1, 10, 5)

with col2:
    sleep_quality_score = st.slider("คะแนนคุณภาพการนอน (1-10)", 1, 10, 5)
    daytime_fatigue = st.slider("คะแนนความเหนื่อยล้าตอนกลางวัน (1-10)", 1, 10, 5)
    news_apps = st.number_input("จำนวนแอปข่าวที่ใช้", min_value=0, max_value=10, value=2)
    phone_checks = st.number_input("จำนวนครั้งที่เช็คโทรศัพท์กลางดึก", min_value=0, max_value=10, value=3)
    exercise = st.number_input("เวลาออกกำลังกายต่อวัน (นาที)", min_value=0, max_value=150, value=30)
    days_detox = st.number_input("จำนวนวันตั้งแต่ digital detox", min_value=0, max_value=365, value=30)
    weekly_sleep_debt = st.number_input("หนี้การนอนต่อสัปดาห์ (ชั่วโมง)", min_value=0.0, max_value=30.0, value=5.0)
    gender = st.selectbox("เพศ", ["Male", "Female", "Prefer not to say"])
    occupation = st.selectbox("สถานะการทำงาน", ["Student", "Employed Full-time", "Employed Part-time", "Unemployed"])
    country = st.selectbox("ประเทศ/ภูมิภาค", ["United States", "India", "United Kingdom", "Canada", "Australia", "Germany", "Brazil", "Philippines", "Nigeria", "UAE"])
    primary_device = st.selectbox("อุปกรณ์หลักที่ใช้ก่อนนอน", ["Smartphone", "Tablet", "Laptop", "TV"])
    bedtime_routine = st.selectbox("กิจวัตรก่อนนอน", ["No Fixed Routine", "Reading", "Meditation/Journaling", "Scrolling Social Media", "Watching Videos"])

# ปุ่มทำนาย
if st.button("🎯 ทำนายผล", type="primary"):
    if models is not None:
        # เตรียมข้อมูล
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
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in input_df.columns:
                le = label_encoders[col]
                # Handle unseen labels
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        X_input = input_df[feature_columns].fillna(0)
        X_input_scaled = scaler.transform(X_input)
        
        # ทำนายผล
        st.markdown("---")
        st.header("📊 ผลการทำนาย")
        
        results = {}
        
        # 1. K-nearest neighbor
        pred = models['K-nearest neighbor'].predict(X_input_scaled)[0]
        results['1. K-nearest neighbor'] = target_encoder.inverse_transform([pred])[0]
        
        # 2. Decision Tree
        pred = models['Decision Tree'].predict(X_input)[0]
        results['2. Decision Tree'] = target_encoder.inverse_transform([pred])[0]
        
        # 3. SVM
        pred = models['SVM'].predict(X_input_scaled)[0]
        results['3. SVM'] = target_encoder.inverse_transform([pred])[0]
        
        # 4. K-mean (Clustering)
        cluster = models['K-mean'].predict(X_input_scaled)[0]
        results['4. K-mean'] = f"Cluster {cluster}"
        
        # 5. Regression
        pred = models['Regression'].predict(X_input_scaled)[0]
        results['5. Regression'] = target_encoder.inverse_transform([pred])[0]
        
        # 6. Ensemble (Random Forest)
        pred = models['Ensemble (Random Forest)'].predict(X_input)[0]
        results['6. Ensemble (Random Forest)'] = target_encoder.inverse_transform([pred])[0]
        
        # แสดงผล
        for model_name, prediction in results.items():
            if prediction in ['Good', 'Fair', 'Poor']:
                if prediction == 'Good':
                    st.success(f"**{model_name}**: {prediction} ✅")
                elif prediction == 'Fair':
                    st.warning(f"**{model_name}**: {prediction} ️")
                else:
                    st.error(f"**{model_name}**: {prediction} ❌")
            else:
                st.info(f"**{model_name}**: {prediction}")
        
        # สรุปผล
        st.markdown("---")
        st.subheader("📈 สรุปผล")
        predictions_list = [v for v in results.values() if v in ['Good', 'Fair', 'Poor']]
        if predictions_list:
            most_common = max(set(predictions_list), key=predictions_list.count)
            if most_common == 'Good':
                st.success(f"🎉 ผลการทำนายส่วนใหญ่: **{most_common}** - คุณภาพการนอนดี!")
            elif most_common == 'Fair':
                st.warning(f"😐 ผลการทำนายส่วนใหญ่: **{most_common}** - พอใช้ ควรปรับปรุง")
            else:
                st.error(f"😴 ผลการทำนายส่วนใหญ่: **{most_common}** - ควรปรับปรุงพฤติกรรมการนอน")
    else:
        st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดล")

st.markdown("---")
st.markdown("**หมายเหตุ:** ระบบนี้ใช้ Machine Learning Models 6 ประเภทในการทำนายคุณภาพการนอนหลับ")
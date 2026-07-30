import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบทำนายคุณภาพการนอนหลับ",
    page_icon="😴",
    layout="wide"
)

# หัวข้อหลัก
st.title("ระบบทำนายคุณภาพการนอนหลับ")
st.markdown("---")

# ฟังก์ชันโหลดโมเดล
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

# เมนูด้านข้าง
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio(
    "เลือกหน้า",
    ["หน้าหลัก", "ทำนายผล", "ข้อมูลโมเดล", "ผู้พัฒนา"]
)

# ==========================================
# หน้า 1: หน้าหลัก
# ==========================================
if page == "หน้าหลัก":
    st.header("ยินดีต้อนรับสู่ระบบทำนายคุณภาพการนอนหลับ")
    
    st.subheader("เกี่ยวกับโปรเจกต์นี้")
    st.write("""
    โปรเจกต์นี้พัฒนาขึ้นเพื่อศึกษาความสัมพันธ์ระหว่างพฤติกรรมการใช้หน้าจอ
    (โดยเฉพาะการ doomscroll) กับคุณภาพการนอนหลับ โดยใช้ Machine Learning 
    จำนวน 6 โมเดลในการทำนาย
    """)
    
    st.subheader("โมเดลที่ใช้ในระบบ")
    st.write("1. K-nearest neighbor (KNN)")
    st.write("2. Decision Tree")
    st.write("3. SVM (Support Vector Machine)")
    st.write("4. K-mean Clustering")
    st.write("5. Regression (Logistic Regression)")
    st.write("6. Ensemble (Random Forest)")
    
    st.subheader("วิธีใช้งาน")
    st.write("1. ไปที่เมนู 'ทำนายผล' ด้านซ้าย")
    st.write("2. กรอกข้อมูลเกี่ยวกับพฤติกรรมการนอนและการใช้หน้าจอ")
    st.write("3. กดปุ่ม 'ทำนายผล'")
    st.write("4. ดูผลลัพธ์จากโมเดลทั้ง 6 ตัว")

# ==========================================
# หน้า 2: ทำนายผล
# ==========================================
elif page == "ทำนายผล":
    st.header("กรอกข้อมูลเพื่อทำนายคุณภาพการนอนหลับ")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ข้อมูลส่วนตัว")
        age = st.number_input("อายุ", min_value=15, max_value=60, value=25)
        gender = st.selectbox("เพศ", ["Male", "Female", "Prefer not to say"])
        occupation = st.selectbox("สถานะการทำงาน", 
                                  ["Student", "Employed Full-time", "Employed Part-time", "Unemployed"])
        country = st.selectbox("ประเทศ/ภูมิภาค", 
                              ["United States", "India", "United Kingdom", "Canada", "Australia", 
                               "Germany", "Brazil", "Philippines", "Nigeria", "UAE"])
    
    with col2:
        st.subheader("พฤติกรรมการใช้หน้าจอ")
        bedtime_screen = st.number_input("เวลาใช้หน้าจอก่อนนอน (นาที)", min_value=0, max_value=300, value=60)
        total_screen = st.number_input("เวลาใช้หน้าจอรวมต่อวัน (ชั่วโมง)", min_value=0.0, max_value=15.0, value=5.0)
        doomscroll_sessions = st.number_input("จำนวนครั้งที่ doomscroll ต่อคืน", min_value=0, max_value=10, value=2)
        avg_doomscroll = st.number_input("เวลาเฉลี่ยต่อ session doomscroll (นาที)", min_value=0, max_value=100, value=20)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("ข้อมูลการนอน")
        sleep_hours = st.number_input("จำนวนชั่วโมงการนอนต่อคืน", min_value=4.0, max_value=10.0, value=7.0)
        sleep_latency = st.number_input("เวลาที่ใช้กว่าจะหลับ (นาที)", min_value=0, max_value=100, value=20)
        night_wakeups = st.number_input("จำนวนครั้งที่ตื่นกลางดึก", min_value=0, max_value=10, value=2)
        weekly_sleep_debt = st.number_input("หนี้การนอนต่อสัปดาห์ (ชั่วโมง)", min_value=0.0, max_value=30.0, value=5.0)
    
    with col4:
        st.subheader("ไลฟ์สไตล์")
        caffeine = st.number_input("ปริมาณคาเฟอีนต่อวัน (mg)", min_value=0, max_value=600, value=100)
        exercise = st.number_input("เวลาออกกำลังกายต่อวัน (นาที)", min_value=0, max_value=150, value=30)
        days_detox = st.number_input("จำนวนวันตั้งแต่ digital detox", min_value=0, max_value=365, value=30)
        phone_checks = st.number_input("จำนวนครั้งที่เช็คโทรศัพท์กลางดึก", min_value=0, max_value=10, value=3)
    
    st.markdown("---")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("สุขภาพจิต")
        anxiety = st.slider("คะแนนความวิตกกังวล (1-10)", 1, 10, 5)
        stress = st.slider("คะแนนความเครียด (1-10)", 1, 10, 5)
        sleep_quality_score = st.slider("คะแนนคุณภาพการนอน (1-10)", 1, 10, 5)
        daytime_fatigue = st.slider("คะแนนความเหนื่อยล้าตอนกลางวัน (1-10)", 1, 10, 5)
    
    with col6:
        st.subheader("การใช้เทคโนโลยี")
        news_apps = st.number_input("จำนวนแอปข่าวที่ใช้", min_value=0, max_value=10, value=2)
        primary_device = st.selectbox("อุปกรณ์หลักที่ใช้ก่อนนอน", 
                                     ["Smartphone", "Tablet", "Laptop", "TV"])
        bedtime_routine = st.selectbox("กิจวัตรก่อนนอน", 
                                      ["No Fixed Routine", "Reading", "Meditation/Journaling", 
                                       "Scrolling Social Media", "Watching Videos"])
    
    # ปุ่มทำนาย
    st.markdown("---")
    if st.button("ทำนายผล", type="primary"):
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
            
            # Encode ตัวแปรหมวดหมู่
            for col in label_encoders.keys():
                if col in input_df.columns:
                    le = label_encoders[col]
                    input_df[col] = input_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
            
            # ตรวจสอบคอลัมน์
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_input = input_df[feature_columns].fillna(0)
            X_input_scaled = scaler.transform(X_input)
            
            # ทำนายผล
            st.subheader("ผลลัพธ์การทำนาย")
            st.markdown("---")
            
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
            results['4. K-mean'] = f"Cluster {cluster} (Unsupervised Learning)"
            
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
                        st.success(f"{model_name}: {prediction} - คุณภาพการนอนดี")
                    elif prediction == 'Fair':
                        st.warning(f"{model_name}: {prediction} - คุณภาพการนอนปานกลาง")
                    else:
                        st.error(f"{model_name}: {prediction} - คุณภาพการนอนแย่")
                else:
                    st.info(f"{model_name}: {prediction}")
            
            # สรุปผล
            st.markdown("---")
            st.subheader("สรุปผลโดยรวม")
            predictions_list = [v for v in results.values() if v in ['Good', 'Fair', 'Poor']]
            if predictions_list:
                most_common = max(set(predictions_list), key=predictions_list.count)
                if most_common == 'Good':
                    st.success(f"ผลลัพธ์ส่วนใหญ่: Good - คุณภาพการนอนของคุณอยู่ในระดับดี")
                elif most_common == 'Fair':
                    st.warning(f"ผลลัพธ์ส่วนใหญ่: Fair - คุณภาพการนอนอยู่ในระดับปานกลาง ควรปรับปรุง")
                else:
                    st.error(f"ผลลัพธ์ส่วนใหญ่: Poor - คุณภาพการนอนอยู่ในระดับแย่ ควรปรับพฤติกรรม")
        else:
            st.error("โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดล")

# ==========================================
# หน้า 3: ข้อมูลโมเดล
# ==========================================
elif page == "ข้อมูลโมเดล":
    st.header("รายละเอียดโมเดล Machine Learning ทั้ง 6 ตัว")
    st.markdown("---")
    
    st.subheader("1. K-nearest neighbor (KNN)")
    st.write("""
    KNN เป็นอัลกอริทึมที่ใช้หลักการ "บอกจากเพื่อนบ้าน" โดยจะดูว่าข้อมูลใหม่
    อยู่ใกล้กับข้อมูลกลุ่มไหนมากที่สุด แล้วจัดให้อยู่ในกลุ่มนั้น
    """)
    
    st.subheader("2. Decision Tree")
    st.write("""
    Decision Tree ทำงานเหมือนการตัดสินใจแบบเป็นขั้นตอน (ถ้า...แล้ว...)
    โดยจะแบ่งข้อมูลออกเป็นกิ่งก้านสาขาตามเงื่อนไขต่างๆ
    """)
    
    st.subheader("3. SVM (Support Vector Machine)")
    st.write("""
    SVM จะหาเส้นหรือระนาบที่ดีที่สุดในการแยกประเภทข้อมูลออกจากกัน
    เหมาะกับข้อมูลที่ซับซ้อนและมีหลายมิติ
    """)
    
    st.subheader("4. K-mean Clustering")
    st.write("""
    K-mean เป็นอัลกอริทึมแบบ Unsupervised Learning ใช้สำหรับจัดกลุ่มข้อมูล
    โดยแบ่งข้อมูลออกเป็น K กลุ่มตามความคล้ายคลึงกัน
    """)
    
    st.subheader("5. Regression (Logistic Regression)")
    st.write("""
    Logistic Regression ใช้สำหรับทำนายผลลัพธ์ที่เป็นหมวดหมู่
    โดยคำนวณความน่าจะเป็นของผลลัพธ์แต่ละประเภท
    """)
    
    st.subheader("6. Ensemble (Random Forest)")
    st.write("""
    Random Forest เป็นโมเดลแบบ Ensemble ที่นำ Decision Tree หลายๆ ต้นมารวมกัน
    เพื่อลดความผิดพลาดและเพิ่มความแม่นยำในการทำนาย
    """)

# ==========================================
# หน้า 4: ผู้พัฒนา
# ==========================================
elif page == "ผู้พัฒนา":
    st.header("ข้อมูลผู้พัฒนา")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("รูปภาพ")
        try:
            st.image("profile.jpg", caption="รูปผู้พัฒนา", use_container_width=True)
        except:
            st.warning("ไม่พบไฟล์รูปภาพ กรุณาอัปโหลดไฟล์ profile.jpg")
            st.image("https://via.placeholder.com/300x300.png?text=Your+Photo", 
                    caption="รูปตัวอย่าง", use_container_width=True)
    
    with col2:
        st.subheader("รายละเอียด")
        st.write("**รหัสนักศึกษา:** 6XXXXXXXXX")
        st.write("**ชื่อ-นามสกุล:** นาย/นางสาว XXXXXXXXXX")
        st.write("**หมู่เรียน:** หมู่เรียนที่ XX")
        st.write("**อีเมล:** your.email@example.com")
    
    st.markdown("---")
    st.subheader("เกี่ยวกับโปรเจกต์")
    st.write("""
    โปรเจกต์นี้พัฒนาขึ้นเพื่อการศึกษาวิชา Machine Learning / Data Science
    โดยนำข้อมูลพฤติกรรมการนอนและการใช้หน้าจอมาสร้างโมเดลทำนายคุณภาพการนอนหลับ
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("ระบบทำนายคุณภาพการนอนหลับ")
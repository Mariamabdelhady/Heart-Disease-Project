

import streamlit as st
import pandas as pd
import joblib
import numpy as np





model = joblib.load("Heart_Disease_Project/models/final_model.pkl")

st.title(" Heart Disease Prediction App")
st.write("أدخل بيانات المريض للتنبؤ إذا كان معرض لمرض القلب أو لا.")


thalach = st.number_input("أقصى معدل نبض thalach", min_value=50, max_value=210, value=150)
thal = st.selectbox("thal (3 = normal, 6 = fixed defect, 7 = reversable defect)", [3, 6, 7])
cp = st.selectbox("نوع ألم الصدر cp (0-3)", [0, 1, 2, 3])
ca = st.number_input("عدد الأوعية الرئيسية (ca) 0-3", min_value=0, max_value=3, value=0)
oldpeak = st.number_input("oldpeak (انخفاض ST)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)


if st.button("🔍 تنبؤ"):
    input_data = pd.DataFrame({
        "thalach": [thalach],
        "thal": [thal],
        "cp": [cp],
        "ca": [ca],
        "oldpeak": [oldpeak]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" النتيجة: المريض معرض لمرض القلب بنسبة {prob:.2%}")
    else:
        st.success(f" النتيجة: المريض غير معرض لمرض القلب بنسبة {(1 - prob):.2%}")

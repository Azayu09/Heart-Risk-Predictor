import streamlit as st
import joblib
import pandas as pd

# ======================
# PAGE
# ======================
st.set_page_config(
    page_title="Heart Risk Detector",
    page_icon="❤️",
    layout="centered"
)

# ======================
# STYLE (tight + aligned)
# ======================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    max-width: 820px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.card {
    background-color: #1c1f26;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

.stButton > button {
    height: 55px;
    font-size: 18px;
    border-radius: 12px;
}

div[data-baseweb="select"] {
    margin-bottom: 12px;
}

.stSlider {
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


# ======================
# LOAD
# ======================
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# ======================
# HEADER
# ======================
st.markdown(
    "<h1 style='text-align:center;'>❤️ Heart Risk Detector</h1>",
    unsafe_allow_html=True
)
st.caption("ML powered heart risk prediction")


# ======================
# CARD START
# ======================
st.markdown('<div class="card">', unsafe_allow_html=True)

# PERFECT 2x GRID
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Vitals")
    age = st.slider("Age", 1, 120, 30)
    bp = st.slider("Resting BP", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    maxhr = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with right:
    st.subheader("Clinical Info")
    fbs = st.selectbox("Fasting BS > 120", [0, 1])
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

st.markdown("</div>", unsafe_allow_html=True)


# ======================
# BUTTON CENTERED
# ======================
colA, colB, colC = st.columns([1,2,1])

with colB:
    predict_btn = st.button("🚀 Predict Risk", use_container_width=True)


# ======================
# PREDICTION
# ======================
if predict_btn:

    input_dict = dict.fromkeys(columns, 0)

    input_dict["Age"] = age
    input_dict["RestingBP"] = bp
    input_dict["Cholesterol"] = chol
    input_dict["FastingBS"] = fbs
    input_dict["MaxHR"] = maxhr
    input_dict["Oldpeak"] = oldpeak

    if sex == "Male":
        input_dict["Sex_M"] = 1

    if cp != "ASY":
        input_dict[f"ChestPainType_{cp}"] = 1

    if ecg != "LVH":
        input_dict[f"RestingECG_{ecg}"] = 1

    if angina == "Yes":
        input_dict["ExerciseAngina_Y"] = 1

    if slope != "Down":
        input_dict[f"ST_Slope_{slope}"] = 1


    df = pd.DataFrame([input_dict])
    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.divider()

    st.metric("Risk Probability", f"{prob*100:.2f}%")
    st.progress(int(prob * 100))

    if pred == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")

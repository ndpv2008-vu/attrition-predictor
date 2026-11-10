import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥")
st.title("👥 Employee Attrition Predictor")
st.caption("Hybrid MI + LASSO feature selection with CatBoost")

@st.cache_resource
def load():
    return joblib.load("model.pkl")

bundle = load()
model, feats = bundle["model"], bundle["features"]
all_cols, depts = bundle["all_columns"], bundle["departments"]

c1, c2 = st.columns(2)
with c1:
    sat = st.slider("Satisfaction level", 0.0, 1.0, 0.5, 0.01)
    ev = st.slider("Last evaluation", 0.0, 1.0, 0.7, 0.01)
    proj = st.number_input("Number of projects", 1, 10, 4)
    hours = st.number_input("Average monthly hours", 50, 350, 200)
with c2:
    tenure = st.number_input("Years at company", 1, 20, 3)
    acc = st.selectbox("Work accident", [0, 1])
    promo = st.selectbox("Promoted in last 5 years", [0, 1])
    dept = st.selectbox("Department", depts)
    sal = st.selectbox("Salary", ["low", "medium", "high"])

if st.button("Predict", type="primary"):
    row = {c: 0 for c in all_cols}
    row.update({"satisfaction_level": sat, "last_evaluation": ev,
                "number_project": proj, "average_montly_hours": hours,
                "time_spend_company": tenure, "Work_accident": acc,
                "promotion_last_5years": promo})
    if f"sales_{dept}" in row: row[f"sales_{dept}"] = 1
    if f"salary_{sal}" in row: row[f"salary_{sal}"] = 1
    X = pd.DataFrame([row])[feats]
    p = float(model.predict_proba(X)[0, 1])
    st.metric("Probability of leaving", f"{p*100:.1f}%")
    if p >= 0.5: st.error("⚠️ High risk: this employee is likely to leave.")
    else: st.success("✅ Low risk: this employee is likely to stay.")

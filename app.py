import streamlit as st
import pandas as pd
import pickle
import pandas as pd
df = pd.read_csv("eda_data.csv")
import traceback

try:
    with open("model_file.p", "rb") as file:
        data = pickle.load(file)
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()
    
    
print(data.keys())
model = data["model"]

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

st.title("💼 Data Science Salary Prediction")

st.markdown("""
### 📖 About Project

This application predicts the estimated annual salary of Data Science professionals using a **Random Forest Regression** machine learning model.

The prediction is based on several factors, including:

- ⭐ Company Rating
- 🏢 Company Age
- 💻 Technical Skills (Python, AWS, Spark, Excel)
- 📍 Job Location
- 💼 Job Role
- 🏭 Industry
- 👥 Company Size
- 🎓 Seniority Level

The model was trained on historical Glassdoor job listings and uses feature engineering, exploratory data analysis (EDA), and hyperparameter tuning to improve prediction accuracy.

**Developed By:** Deepank Juneja  
**Course:** Bachelor of Computer Applications (BCA)  
**Project:** Data Science Salary Prediction using Machine Learning
""")

st.header("Enter Job Details")
rating = st.slider(
    "Company Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.1
)

st.write("Selected Rating:", rating)

age = st.number_input(
    "Company Age",
    min_value=0,
    max_value=200,
    value=10,
    step=1
)

st.write("Company Age:", age)

st.header("Enter skillset")

python_skill = st.checkbox("Python")
st.write("Python:", python_skill)

aws_skill = st.checkbox("AWS")
st.write("AWS:", aws_skill)

spark_skill = st.checkbox("Spark")
st.write("Spark:", spark_skill)

excel_skill = st.checkbox("Excel")
st.write("Excel:", excel_skill)

company_size = st.selectbox(
    "Company Size",
    [
        "1 to 50 employees",
        "51 to 200 employees",
        "201 to 500 employees",
        "501 to 1000 employees",
        "1001 to 5000 employees",
        "5001 to 10000 employees",
        "10000+ employees",
        "Unknown"
    ]
)

st.write("Company Size:", company_size)


industry = st.selectbox(
    "Industry",
    [
        "Aerospace & Defense",
        "Health Care Services & Hospitals",
        "Security Services",
        "Energy",
        "Advertising & Marketing",
        "Real Estate",
        "Banks & Credit Unions",
        "Consulting",
        "Internet",
        "Research & Development",
        "Biotech & Pharmaceuticals",
        "Insurance Carriers",
        "Telecommunications Services",
        "IT Services",
        "Computer Hardware & Software",
        "Consumer Products Manufacturing",
        "Industrial Manufacturing",
        "Financial Analytics & Research",
        "Education Training Services",
        "Transportation Equipment Manufacturing",
        "Construction",
        "Accounting",
        "Unknown"
    ]
)

st.write("Industry:", industry)

job_state = st.selectbox(
    "Job State",
    [
        "CA","TX","NY","WA","MA","VA","FL","NJ","IL",
        "CO","PA","GA","NC","OH","MD","MI","MN","OR",
        "AZ","UT","MO","TN","CT","WI","LA","NM","AL",
        "SC","KS","RI","DC","DE","ID","NE","IN","IA"
    ]
)

st.write("Job State:", job_state)


job_role = st.selectbox(
    "Job Role",
    [
        "data scientist",
        "data analyst",
        "data engineer",
        "machine learning engineer",
        "manager",
        "director",
        "other"
    ]
)

st.write("Job Role:", job_role)

seniority = st.selectbox(
    "Seniority",
    [
        "na",
        "junior",
        "senior",
        "lead",
        "principal"
    ]
)

st.write("Seniority:", seniority)

predict_button = st.button("💰 Predict Salary")


if predict_button:
    st.write("Preparing data for prediction...")
    input_dict = {}
    
    for feature in data["model_columns"]:
        input_dict[feature] = 0


    input_dict["Rating"] = rating
    input_dict["age"] = age
    input_dict["python_yn"] = int(python_skill)
    input_dict["aws"] = int(aws_skill)
    input_dict["spark"] = int(spark_skill)
    input_dict["excel"] = int(excel_skill)
    
    
    size_col = "Size_" + company_size
    if size_col in input_dict:
        input_dict[size_col] = 1
    industry_col = "Industry_" + industry
    if industry_col in input_dict:
        input_dict[industry_col] = 1
    state_col = "job_state_" + job_state
    if state_col in input_dict:
        input_dict[state_col] = 1
    role_col = "job_simp_" + job_role
    if role_col in input_dict:
        input_dict[role_col] = 1
    seniority_col = "seniority_" + seniority
    if seniority_col in input_dict:
        input_dict[seniority_col] = 1


    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ${prediction*1000:,.0f}")

st.subheader("📋 Similar Jobs from Dataset")

similar_jobs = df[
    (df["job_simp"] == job_role) &
    (df["job_state"] == job_state)
]

if python_skill:
    similar_jobs = similar_jobs[similar_jobs["python_yn"] == 1]

if aws_skill:
    similar_jobs = similar_jobs[similar_jobs["aws"] == 1]

if spark_skill:
    similar_jobs = similar_jobs[similar_jobs["spark"] == 1]

if excel_skill:
    similar_jobs = similar_jobs[similar_jobs["excel"] == 1]

if len(similar_jobs) > 0:

    display_df = similar_jobs[
        ["Company Name", "Job Title", "avg_salary"]
    ].copy()

    display_df["avg_salary"] = (
        "$" + (display_df["avg_salary"] * 1000).round().astype(int).astype(str)
    )

    st.dataframe(display_df.head(10), use_container_width=True)

    st.info(
        f"Found {len(similar_jobs)} similar jobs in the dataset."
    )

else:
    st.warning("No similar jobs were found in the dataset.")
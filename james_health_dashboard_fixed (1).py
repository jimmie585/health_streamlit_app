import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- 🎯 Project Overview ----
st.title("📊 Patient Health Data Dashboard")
st.markdown("""
### 🏥 Project Overview  
This interactive dashboard **analyzes patient health data** and applies **Machine Learning (KNN)** to **predict diabetes risks**.  
It helps healthcare professionals **identify trends, detect potential health risks, and make data-driven decisions**.

### 🔬 Problem Statement  
Healthcare professionals often struggle with **analyzing large datasets** to detect **trends and health risks** efficiently.  
This dashboard provides a **simple, visual, and data-driven solution** to explore patient metrics and predict health outcomes.
""")

# ---- 📂 Loading Dataset ----
st.markdown("## 📂 Loading Patient Data")
st.write("This dataset contains **patient health records**, including age, BMI, blood pressure, diabetes status, and medication adherence.")

# URL to dataset stored on GitHub
DATA_PATH = "https://raw.githubusercontent.com/jimmie585/health_streamlit_app/main/healthcare_dataset.csv"  

try:
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Dataset not found at: {DATA_PATH}")
    st.stop()  # Stop execution if dataset is missing

# ---- 🔍 Sidebar Filters ----
st.sidebar.header("🔍 Filter Patient Data")
st.sidebar.write("Use the filters below to **narrow down patient records** based on Age and BMI.")

# Age Filter (single input)
age_input = st.sidebar.number_input("Enter Age", min_value=0, max_value=120, value=30)

# BMI Filter (single input)
bmi_input = st.sidebar.number_input("Enter BMI", min_value=20.0, max_value=150.0, value=25.0)

# Apply Filters
filtered_df = df[(df["Age"] == age_input) & (df["BMI"] == bmi_input)]


# ---- 📋 Displaying Filtered Data ----
st.markdown("## 📋 Filtered Patient Data")
st.write("Here are the **filtered patient records** based on the selected Age and BMI range.")
st.write(filtered_df)

# ---- 📊 Statistical Insights ----
st.sidebar.header("📊 Statistical Insights")
st.sidebar.write("Below are key **statistical distributions** of patient health conditions.")

st.sidebar.subheader("👤 Gender Distribution")
st.sidebar.write(filtered_df["Gender"].value_counts())

st.sidebar.subheader("❤️ Blood Pressure Levels")
st.sidebar.write(filtered_df["Blood_Pressure"].value_counts())

st.sidebar.subheader("🩸 Diabetes Cases")
st.sidebar.write(filtered_df["Diabetes"].value_counts())

# ---- 📈 Data Visualizations ----
st.markdown("## 📈 Data Visualizations")
st.write("Visualizing **important health metrics** from patient records.")

st.subheader("🩸 Blood Pressure Distribution")
st.bar_chart(filtered_df["Blood_Pressure"].value_counts())

st.subheader("💊 Medication Adherence Levels")
st.bar_chart(filtered_df["Medication_Adherence"].value_counts())

# ---- 🧠 Machine Learning (KNN) for Diabetes Prediction ----
st.markdown("## 🧠 Machine Learning Model (KNN) for Diabetes Prediction")
st.write("""
The **K-Nearest Neighbors (KNN) algorithm** is used here to **predict whether a patient has diabetes** based on their **Age and BMI**.  
This is a simple machine learning model that **classifies patients into "Yes" (Diabetes) or "No" (No Diabetes)** categories.
""")

# Selecting Features for Prediction
features = ["Age", "BMI"]
X = df[features]
y = df["Diabetes"]  # Target Variable

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Data for Better Performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### ✅ Model Accuracy: {accuracy:.2f}")
st.write("This accuracy score tells us **how well the model predicts diabetes cases**.")

# ---- 📊 Confusion Matrix & Classification Report ----
st.markdown("### 🔄 Confusion Matrix")
st.write("The confusion matrix shows how well the model classified **patients with and without diabetes**.")
st.write(confusion_matrix(y_test, y_pred))

st.markdown("### 📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# ---- 💡 How to Use This Dashboard ----
st.markdown("## 💡 How to Use This Dashboard")
st.write("""
1️⃣ **Use the Sidebar Filters**: Adjust the **Age and BMI sliders** to filter patient records.  
2️⃣ **Check the Data Table**: View patient data matching the selected filters.  
3️⃣ **Analyze Statistics**: Explore **gender, blood pressure, and diabetes distributions** in the sidebar.  
4️⃣ **View Charts**: Understand patient health trends using the **bar charts**.  
5️⃣ **Check the Machine Learning Model**: The **KNN model predicts diabetes risks** based on Age and BMI.  
6️⃣ **Interpret the Accuracy & Confusion Matrix**: Higher accuracy means **better diabetes predictions**.
""")

# ---- 📞 Contact & Footer ----
st.markdown("---")
st.markdown("""
### 👨‍💻 Created by **James Ndungu**
📧 Email: [jamesndungu.dev@gmail.com](mailto:jamesndungu.dev@gmail.com)  
📞 Phone: **+254796593045**  
🔗 GitHub: [James' GitHub](https://github.com/jimmie585)  

✅ **If you find this dashboard useful, give it a ⭐ on GitHub!**
""")

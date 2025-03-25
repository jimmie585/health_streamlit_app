import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- Project Overview ----
st.title("📊 Patient Health Data Dashboard")
st.markdown("""
### Project Overview  
This dashboard provides an interactive analysis of patient health data using Machine Learning (KNN) to predict patient conditions.

### Problem Statement  
Healthcare professionals need tools to analyze patient data, detect trends, and assess risks efficiently.  
This dashboard simplifies patient health metric exploration.
""")

# ---- Loading Dataset ----
DATA_PATH = "https://raw.githubusercontent.com/jimmie585/health_streamlit_app/refs/heads/main/healthcare_dataset.csv"  # Update with actual file path

try:
    df = pd.read_csv(DATA_PATH)
    #st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Dataset not found at: {DATA_PATH}")
    st.stop()  # Stop execution if dataset is missing

# ---- Sidebar Filters ----
st.sidebar.header("🔍 Filter Data")

# Age Filter
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# BMI Filter
min_bmi, max_bmi = float(df["BMI"].min()), float(df["BMI"].max())
bmi_filter = st.sidebar.slider("Select BMI Range", min_bmi, max_bmi, (min_bmi, max_bmi))

# Apply Filters
filtered_df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1]) & 
                 (df["BMI"] >= bmi_filter[0]) & (df["BMI"] <= bmi_filter[1])]

# ---- Display Data ----
st.header("📋 Filtered Patient Data")
st.write(filtered_df)

# ---- Statistics ----
st.sidebar.header("📊 Statistics")

st.sidebar.subheader("👤 Gender Distribution")
st.sidebar.write(filtered_df["Gender"].value_counts())

st.sidebar.subheader("❤️ Blood Pressure Distribution")
st.sidebar.write(filtered_df["Blood_Pressure"].value_counts())

st.sidebar.subheader("🩸 Diabetes Distribution")
st.sidebar.write(filtered_df["Diabetes"].value_counts())

# ---- Visualizations ----
st.header("📈 Data Visualizations")

st.subheader("🩸 Blood Pressure Distribution")
st.bar_chart(filtered_df["Blood_Pressure"].value_counts())

st.subheader("💊 Medication Adherence Distribution")
st.bar_chart(filtered_df["Medication_Adherence"].value_counts())

# ---- Machine Learning Model (KNN) ----
st.header("🧠 KNN Model for Prediction")

features = ["Age", "BMI"]
X = df[features]
y = df["Diabetes"]  # Target Variable

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"**Model Accuracy:** {accuracy:.2f}")

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---- Footer ----
st.markdown("---")
st.markdown("""
**👨‍💻 Created by James Ndungu**  
📧 Email: [jamesndungu.dev@gmail.com](mailto:jamesndungu.dev@gmail.com)  
📞 Phone: +254796593045  
🔗 GitHub: [James' GitHub](https://github.com/jimmie585)
""")

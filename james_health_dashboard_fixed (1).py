import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- App Title and Description ----
st.title("🩺 Patient Health Analytics Dashboard")
st.markdown("""
Welcome to the **Patient Health Dashboard**.  
This tool helps you **explore healthcare data**, **predict diabetes risk**, and **understand clustering patterns** using Machine Learning models (KNN & K-Means).  
Use the sidebar to input Age and BMI to get personalized predictions, or scroll down to view the full data and visuals.

---
""")
# ---- Problem Statement and Solution ----
st.markdown("## 🩺 Problem Statement")
st.markdown("""
In modern healthcare, early detection and monitoring of chronic conditions like **diabetes** is critical.  
However, healthcare professionals often lack quick tools to analyze large amounts of patient data, visualize patterns, and predict health risks effectively.

This app addresses the following problems:
- ❌ Difficulty identifying patients at risk of diabetes using basic indicators like Age and BMI.
- ❌ Lack of quick, visual insights into health metrics like blood pressure, medication adherence, and diabetes distribution.
- ❌ Manual analysis of health records is time-consuming and error-prone.

---
""")

st.markdown("## ✅ Solution")
st.markdown("""
This dashboard offers a **data-driven solution** using **machine learning** to:
- ✅ **Predict Diabetes Status**: Users can enter Age and BMI to check diabetes risk using a trained **K-Nearest Neighbors (KNN)** model.
- ✅ **Cluster Patients**: The app uses **K-Means Clustering** to group similar patient profiles for better population health insights.
- ✅ **Interactive Data Filtering**: Users can filter patients by Age and BMI and view health trends instantly.
- ✅ **Visual Analytics**: Clear bar charts show blood pressure levels, medication adherence, and diabetes distribution.

By combining prediction, clustering, and visualization in one tool, healthcare data analysis becomes easier, faster, and more accessible — even without coding knowledge.

---
""")


# ---- Sidebar: Prediction Input ----
st.sidebar.header("📊 Make a Prediction")
st.sidebar.markdown("Enter a patient's **Age** and **BMI** to predict their diabetes status and determine which health cluster they belong to.")

# Sidebar inputs for prediction
st.sidebar.header("Diabetes Prediction Input")
age_input = st.sidebar.number_input("Enter Age", min_value=0, max_value=120, value=30)
bmi_input = st.sidebar.number_input("Enter BMI", min_value=10.0, max_value=150.0, value=22.5)

input_data = np.array([[age_input, bmi_input]])



if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[age_input, bmi_input]])
    diabetes_pred = knn.predict(input_data)[0]
    cluster_pred = kmeans.predict(input_data)[0]

    st.subheader("🔎 Prediction Result")
    st.markdown(f"- **Diabetes Status**: `{diabetes_pred}`")
    st.markdown(f"- **Assigned Cluster**: `Cluster {cluster_pred}`")
    st.info("These results are based on machine learning models trained using real patient data.")

# ---- Sidebar Stats ----
st.sidebar.markdown("---")
st.sidebar.header("📈 Filtered Stats Summary")
st.sidebar.markdown("This section shows summary stats based on your filter selections in the main section.")

# ---- Main: Filter Data Table ----
st.header("📄 Filter Patient Data")
st.markdown("Use the sliders below to filter patients based on **Age** and **BMI**. The table and charts will update accordingly.")
# Convert Age column to numeric and drop missing values
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df = df.dropna(subset=["Age"])

# Define min and max age for slider
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

# Create slider with safe values
age_filter = st.slider("Select Age Range", min_age, max_age, (20, 80))
bmi_filter = st.slider("Select BMI Range", float(df["BMI"].min()), float(df["BMI"].max()), (20.0, 50.0))

filtered_df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1]) &
                 (df["BMI"] >= bmi_filter[0]) & (df["BMI"] <= bmi_filter[1])]

st.dataframe(filtered_df, use_container_width=True)

# ---- Sidebar Stats Continued ----
st.sidebar.write("**Gender Distribution**")
st.sidebar.write(filtered_df["Gender"].value_counts())

st.sidebar.write("**Blood Pressure Levels**")
st.sidebar.write(filtered_df["Blood_Pressure"].value_counts())

st.sidebar.write("**Diabetes Cases**")
st.sidebar.write(filtered_df["Diabetes"].value_counts())

st.sidebar.write("**Medication Adherence**")
st.sidebar.write(filtered_df["Medication_Adherence"].value_counts())
# ---- Statistical Insights ----
st.markdown("## 📊 Statistical Insights")
st.markdown("This section summarizes key statistics from the currently filtered patient dataset to help identify health trends quickly.")

col1, col2 = st.columns(2)

with col1:
    st.metric("🧓 Average Age", f"{filtered_df['Age'].mean():.1f} years")
    st.metric("📉 Minimum BMI", f"{filtered_df['BMI'].min():.1f}")
    st.metric("📈 Maximum BMI", f"{filtered_df['BMI'].max():.1f}")

with col2:
    diabetes_rate = filtered_df['Diabetes'].value_counts(normalize=True).get("Yes", 0) * 100
    st.metric("🩺 Diabetes Rate", f"{diabetes_rate:.1f}%")
    adherence_rate = filtered_df['Medication_Adherence'].value_counts(normalize=True).get("Adherent", 0) * 100
    st.metric("💊 Medication Adherence", f"{adherence_rate:.1f}%")

# Additional frequency tables
st.markdown("### 🧠 Gender Distribution")
st.write(filtered_df["Gender"].value_counts())

st.markdown("### 💉 Blood Pressure Levels")
st.write(filtered_df["Blood_Pressure"].value_counts())


# ---- Visualizations ----
st.header("📊 Visualizations")
st.markdown("Below are visualizations to help you understand how different health indicators are distributed across the filtered data.")

st.subheader("📌 Blood Pressure Distribution")
st.bar_chart(filtered_df["Blood_Pressure"].value_counts())

st.subheader("📌 Diabetes Status Distribution")
st.bar_chart(filtered_df["Diabetes"].value_counts())

st.subheader("📌 Medication Adherence Levels")
st.bar_chart(filtered_df["Medication_Adherence"].value_counts())

# ---- 🧠 Machine Learning (KNN) for Diabetes Prediction ----
st.markdown("## 🧠 Machine Learning Model (KNN) for Diabetes Prediction")
st.write("""
The **K-Nearest Neighbors (KNN) algorithm** is used here to **predict whether a patient has diabetes** based on their **Age and BMI**.  
This is a simple machine learning model that **classifies patients into "Yes" (Diabetes) or "No" (No Diabetes)** categories.
""")

# Features and target
X = df[['Age', 'BMI']]
y = df['Diabetes']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_res, y_train_res)

# Evaluate
y_pred = knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

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

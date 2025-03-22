#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[2]:


#loading the data
df=pd.read_csv("C:\\Users\\ADMIN\\Downloads\\healthcare_dataset.csv")
df


# In[3]:


##handling missing values
df.isnull().sum()


# In[ ]:





# In[4]:


#feature and targeted variable
X=df[['Age','BMI']] # independant variables
y=df['Diabetes'] # dependant variable
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create KNN model with K=6
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN model: {accuracy * 100:.2f}%')
# Visualizing the results
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['BMI'], c=df['Diabetes'].apply(lambda x: 1 if x == 'Yes' else 0), cmap='coolwarm')
plt.title('KNN Classification')
plt.xlabel('BMI')
plt.ylabel('Age')
plt.scatter(X_test['Age'], X_test['BMI'], marker='x', color='black', label='Test Points')
plt.legend(['No', 'Yes','Test Points'])
plt.show()


# In[5]:


##finding no of k
import math
math.sqrt(len(X_test))


# In[6]:


knn.predict(X_test)


# In[7]:


knn.predict([[45,80],[90,78]])


# # K-MEANS CLUSTERING MODEL

# In[8]:


# Loading my dataset
df = pd.read_csv('C:\\Users\\ADMIN\\Downloads\\healthcare_dataset.csv')
from sklearn.preprocessing import LabelEncoder

# Labeling encode categorical variables
label_encoder = LabelEncoder()
df['Blood_Encoded'] = label_encoder.fit_transform(df['Blood_Pressure'])
df['Diabetes_Encoded'] = label_encoder.fit_transform(df['Diabetes'])
df['Smoking_Status'] = label_encoder.fit_transform(df['Smoking_Status'])
#  encoding data for clustering
X = df[['Blood_Encoded', 'Diabetes_Encoded', 'Smoking_Status']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
# Elbow Method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Ploting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


# In[9]:


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_


# Print the clustered data
print("Clustered Data:")
print(df)


# In[10]:


# Analyzing clusters
for cluster in range(5):
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster].value_counts())


# In[11]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reducing dimensions to 2D using Principal component analysis PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)

# Ploting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['Cluster'], cmap='viridis', s=100)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering (k=4) on Categorical Data')
plt.colorbar(label='Cluster')
plt.show()


# In[12]:


# Visualize the distribution of 'Blood_Pressure' in each cluster
plt.figure(figsize=(10, 6))
pd.crosstab(df['Cluster'], df['Blood_Pressure']).plot(kind='bar', stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Distribution of Blood Pressure in Each Cluster')
plt.show()

# Visualize the distribution of Diabetes in each cluster
plt.figure(figsize=(10, 6))
pd.crosstab(df['Cluster'], df['Diabetes']).plot(kind='bar', stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Distribution of Size in Each Cluster')
plt.show()

# Visualize the distribution of 'Category' in each cluster
plt.figure(figsize=(10, 6))
pd.crosstab(df['Cluster'], df['Smoking_Status']).plot(kind='bar', stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Distribution of Category in Each Cluster')
plt.show()


# # OBSERVATIONS
# ## cluster 0 
# * Patients have both Normal and Prehypertension
# * all patients have Diabetes
# * Most of the patients smoke
# ## cluster one
# * Most of the patients have Hypertension blood pressure and a small percentage have normal
# * Almost all patients have No Diabetes
# * All patients smoke
# ## cluster two
# * The patients have Normal Blood Pressure
# * Almost all the patients have No Diabetes
# * The petient do Not smoke
# ## cluster three
# * All the patients have Prehypertension Blood Pressure
# * All Patients have No Diabetes
# * Almost all Patients do Not smoke
# 
# ## cluster four
# * All patients have Prehypertension 
# * Almost all patients have No Diabetes
# * All patients do Not smoke

# In[13]:


print(knn)
print(kmeans)


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\healthcare_dataset.csv")

# Feature selection
X = df[['Age', 'BMI']]  # Independent variables
y = df['Diabetes']  # Dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save the models
import pickle
with open("knn_model.pkl", "wb") as knn_file:
    pickle.dump(knn, knn_file)

with open("kmeans_model.pkl", "wb") as kmeans_file:
    pickle.dump(kmeans, kmeans_file)

print("Models saved successfully!")


# In[15]:


import pickle

with open("knn_model.pkl", "rb") as file:
    knn = pickle.load(file)
print(knn)


# In[17]:


import streamlit as st
import pandas as pd

# Sample data
data = {
    "Patient_ID": ["P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016"],
    "Age": [32, 78, 38, 41, 20, 39, 70, 19, 47, 55, 19, 81, 77, 38, 50],
    "Gender": ["Female", "Male", "Male", "Female", "Male", "Male", "Male", "Male", "Male", "Female", "Female", "Female", "Male", "Male", "Male"],
    "BMI": [21.7, 40, 24.2, 39.5, 27.3, 19.2, 25.9, 32.1, 33.1, 29.9, 28.1, 30.4, 31.2, 20.2, 26.4],
    "Blood_Pressure": ["Hypertension", "Hypertension", "Normal", "Prehypertension", "Prehypertension", "Hypertension", "Normal", "Normal", "Normal", "Normal", "Prehypertension", "Normal", "Normal", "Hypertension", "Hypertension"],
    "Cholesterol_Level": ["Normal", "Low", "High", "High", "Low", "Low", "High", "Normal", "Normal", "Normal", "Normal", "High", "High", "High", "Normal"],
    "Diabetes": ["No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No"],
    "Hospital_Visits_Per_Year": [None] * 15,
    "Medication_Adherence": ["Good", "Good", "Good", "Good", "Good", "Moderate", "Moderate", "Good", "Good", "Good", "Poor", "Good", "Moderate", "Poor", "Moderate"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Streamlit App Title
st.title("Patient Health Data Dashboard")

# Sidebar for user input
st.sidebar.header("Filter Data")

# Filter by Age
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Filter by BMI
min_bmi = float(df["BMI"].min())
max_bmi = float(df["BMI"].max())
bmi_filter = st.sidebar.slider("Select BMI Range", min_bmi, max_bmi, (min_bmi, max_bmi))

# Apply filters
filtered_df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1]) & 
                (df["BMI"] >= bmi_filter[0]) & (df["BMI"] <= bmi_filter[1])]

# Display the filtered DataFrame
st.header("Filtered Patient Data")
st.write(filtered_df)

# Display statistics for other variables
st.sidebar.header("Statistics")

# Gender Distribution
st.sidebar.subheader("Gender Distribution")
gender_counts = filtered_df["Gender"].value_counts()
st.sidebar.write(gender_counts)

# Blood Pressure Distribution
st.sidebar.subheader("Blood Pressure Distribution")
blood_pressure_counts = filtered_df["Blood_Pressure"].value_counts()
st.sidebar.write(blood_pressure_counts)

# Diabetes Distribution
st.sidebar.subheader("Diabetes Distribution")
diabetes_counts = filtered_df["Diabetes"].value_counts()
st.sidebar.write(diabetes_counts)

# Medication Adherence Distribution
st.sidebar.subheader("Medication Adherence Distribution")
medication_counts = filtered_df["Medication_Adherence"].value_counts()
st.sidebar.write(medication_counts)

# Display a bar chart for Blood Pressure distribution
st.header("Blood Pressure Distribution")
st.bar_chart(blood_pressure_counts)

# Display a bar chart for Medication Adherence distribution
st.header("Medication Adherence Distribution")
st.bar_chart(medication_counts)


# In[ ]:





# In[ ]:





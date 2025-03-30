# health_streamlit_app
Streamlit Heart Disease Risk Analysis

## Overview

This project is a Streamlit-based dashboard designed to analyze and assess the risk of heart disease using data visualization and machine learning models. The tool helps users understand various risk factors and provides predictive insights based on user inputs.

## Features

User-Friendly Interface: Built with Streamlit for easy interaction.use this link to view the app https://healthappapp-5f9rlj5qgverakvlz4szij.streamlit.app/

Data Visualization: Interactive charts and graphs for exploratory data analysis (EDA).

Machine Learning Model: Utilizes classification models such as K-Nearest Neighbors (KNN) and Random Forest to predict heart disease risk.

Dynamic Inputs: Allows users to input health parameters like age, cholesterol levels, blood pressure, and more.

Real-Time Predictions: Provides instant risk assessment based on input data.

## Installation

To run the application, follow these steps:

## Clone the repository:

git clone https://github.com/jimmie585/streamlit-health-project.git
cd streamlit-health-project

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run health_analysis_final.py

Dependencies

Python 3.x

Streamlit

Pandas

Scikit-learn

Matplotlib & Seaborn

## Usage

Open the application in your browser after running streamlit run app.py.

Enter relevant health parameters in the input fields.

View real-time risk predictions and visualizations.

## Dataset

The model is trained on a publicly available heart disease dataset containing features such as:

Age

Sex

Blood Pressure

Cholesterol Level

Heart Rate

Other medical factors

## Future Improvements

Integration of more advanced ML models.

Deployment on cloud platforms for wider accessibility.

Addition of more health parameters for better accuracy.

#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

# Load dataset directly from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/jimmie585/health_streamlit_app/main/healthcare_dataset.csv")


# Display first few rows to confirm it works
print(df.head())


# In[ ]:





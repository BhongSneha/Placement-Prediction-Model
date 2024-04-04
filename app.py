import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
lg = pickle.load(open('placement.pkl', 'rb'))

# Define the Streamlit app
st.title("Job Placement Prediction Model")
img = Image.open('job-placement.jpg')
st.image(img, width=650)
# st.title("Job Placement Prediction Model")

# Prompt users to enter their details
st.markdown("<p style='font-size:20px;font-weight:bold;'>Please Enter Your Details</p>", unsafe_allow_html=True)

# Input fields for each feature separately
gender_options = ["Yes", "No"]
gender_M = st.radio("Select Gender (Male: Yes, Female: No)", options=gender_options)
ssc_percentage = st.number_input("Enter SSC Percentage", min_value=0.0, max_value=100.0, step=0.01)
hsc_percentage = st.number_input("Enter HSC Percentage", min_value=0.0, max_value=100.0, step=0.01)
Fe_cgpa = st.number_input("Enter FE CGPA", min_value=0.0, max_value=10.0, step=0.01)
Se_cgpa = st.number_input("Enter SE CGPA", min_value=0.0, max_value=10.0, step=0.01)
Te_cgpa = st.number_input("Enter TE CGPA", min_value=0.0, max_value=10.0, step=0.01)
Be_cgpa = st.number_input("Enter BE CGPA", min_value=0.0, max_value=10.0, step=0.01)
agg_percentage = st.number_input("Enter Degree Percentage", min_value=0.0, max_value=10.0, step=0.01)

# Convert gender to numeric
gender_numeric = 1 if gender_M == "Yes" else 0

work_experience_options = ["Yes", "No"]
work_experience_Yes = st.radio("Work Experience", options=work_experience_options)
work_experience_numeric = 1 if work_experience_Yes == "Yes" else 0

No_of_Internships = st.number_input("Enter Number of Internships", min_value=0, max_value=100, step=1)

# Button to trigger prediction
if st.button("Predict Placement"):
    # Make prediction
    input_data = np.array([[gender_numeric, ssc_percentage, hsc_percentage, Fe_cgpa, Se_cgpa, Te_cgpa, Be_cgpa, agg_percentage, work_experience_numeric, No_of_Internships]])
    pred = lg.predict(input_data)

    # Display prediction result with highlight
    if pred[0] == 1:
        st.markdown("**This person is placed for the job**", unsafe_allow_html=True)
    else:
        st.markdown("**This person is not placed for the job**", unsafe_allow_html=True)

# Save the trained model
pickle.dump(lg, open('placement.pkl', 'wb'))


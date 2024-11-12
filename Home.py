import streamlit as st
import pandas as pd
import numpy as np
from dashboard import dashboard  # Import the dashboard function

# Custom CSS for overall styling and backgrounds
st.markdown("""
    <style>
    .main {
        background: linear-gradient(rgba(27, 29, 30, 0.7), rgba(173, 216, 230, 0.7)), url('images/background.jpeg');
        background-size: cover;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar > div:first-child {
        padding-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Home Page
def home():
    st.title("🌍 Youth Employment Empowerment System ")
    st.header("Overview")
    st.image("image0.jpeg", use_column_width=True)  # Add an overview image
    st.write("""
    This project aims to address the unemployment problem in Rwanda by leveraging quality education.
    The application provides insights into current unemployment trends and offers personalized recommendations
    for further education or skill acquisition.
    """)
    with st.expander("See Features"):
        st.write("""
        ### Features:
        - **Dashboard**: Shows current trends in unemployment rates in Rwanda.
        - **Prediction Page**: Predicts future unemployment trends based on historical data.
        - **Skill & Course Recommendation System**: Provides personalized recommendations for further education or skill acquisition.
        """)

# Prediction Page
def prediction():
    st.title("Prediction Page")
    st.header("Predict Future Unemployment Trends")
    st.write("Machine learning model to predict future unemployment trends based on historical data.")
    st.write("Future unemployment trends will be displayed here.")
    # Placeholder for prediction model
    st.write("Future unemployment trends will be displayed here.")

# Skill & Course Recommendation System
def recommendation():
    st.title("🎓 Skill & Course Recommendation System")
    st.header("Personalized Recommendations")
    st.write("Please answer the following questions to get tailored recommendations.")

    # Collecting user responses
    name = st.text_input("1. What is your name?")
    gender = st.selectbox("2. What is your gender?", ["Male", "Female", "Other"])
    ug_course = st.text_input("3. What was your course in UG?")
    ug_specialization = st.text_input("4. What is your UG specialization? (e.g., Major Subject)")
    interests = st.text_input("5. What are your interests?")
    skills = st.multiselect(
        "6. What are your skills? (Select multiple if necessary)", 
        ["Programming", "Data Analysis", "Project Management", "Design", "Communication", "Marketing", "Research"]
    )
    cgpa = st.text_input("7. What was the average CGPA or Percentage obtained in under graduation?")
    certification = st.selectbox("8. Did you do any certification courses additionally?", ["Yes", "No"])
    
    # Conditional question based on certification answer
    if certification == "Yes":
        certificate_course = st.text_input("9. If yes, please specify your certificate course title.")

    working_status = st.selectbox("10. Are you working?", ["Yes", "No"])
    
    # Conditional question based on working status
    if working_status == "Yes":
        first_job_title = st.text_input("11. If yes, what was/is your first Job title in your field? If not applicable, write NA.")
    else:
        first_job_title = "NA"

    masters_status = st.selectbox(
        "12. Have you done a master’s degree after undergraduation?",
        ["Yes", "No"]
    )
    
    # Conditional question based on master's degree status
    if masters_status == "Yes":
        masters_field = st.text_input("If yes, mention your field of master’s. (e.g., Masters in Mathematics)")
    else:
        masters_field = "NA"

    # Submit button to trigger recommendations
    if st.button("Get Recommendations"):
        st.write(f"## Recommendations for {name}")
        st.write(f"### Based on your background in {ug_course} with a specialization in {ug_specialization}, here are some suggestions:")

        # Example logic for recommendations (customize as needed)
        if "Data Analysis" in skills:
            st.write("- Consider Data Science and Machine Learning courses to deepen your analysis skills.")
        if "Programming" in skills and masters_status == "No":
            st.write("- Pursuing a master's in Computer Science could open up advanced roles in software development.")
        if certification == "Yes" and masters_status == "Yes":
            st.write(f"- Building on your certification in {certificate_course} and master's in {masters_field}, you might explore leadership roles or specialized certifications.")


# Navigation
st.sidebar.image("Logo.png", use_column_width=True)  # Add the logo at the top of the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Prediction", "Recommendation"])

if page == "Home":
    home()
elif page == "Dashboard":
    dashboard()
elif page == "Prediction":
    prediction()
elif page == "Recommendation":
    recommendation()
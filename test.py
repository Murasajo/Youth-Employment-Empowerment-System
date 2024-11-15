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
        - *Dashboard*: Shows current trends in unemployment rates in Rwanda.
        - *Prediction Page*: Predicts future unemployment trends based on historical data.
        - *Skill & Course Recommendation System*: Provides personalized recommendations for further education or skill acquisition.
        """)

# Prediction Page
def prediction():
    st.title("Prediction Page")
    st.header("Predict Future Unemployment Trends")
    st.write("Machine learning model to predict future unemployment trends based on historical data.")
    st.write("Future unemployment trends will be displayed here.")

# Skill & Course Recommendation System
def recommendation():
    st.title("🎓 Skill & Course Recommendation System")
    st.header("Personalized Recommendations")
    # (User input fields are retained here as in the original code)
    # ...

# Documentation Page
def documentation():
    st.title("📄 Project Documentation")
    st.header("Abstract")
    st.write("""
    This project aims to tackle youth unemployment in Rwanda using data-driven solutions. 
    The platform provides insights into current trends, predictive analytics, and personalized recommendations for skill enhancement.
    """)

    st.header("Introduction")
    st.write("""
    Youth unemployment poses significant challenges, affecting economic growth and social stability. 
    This app leverages data analysis to address the issue, helping stakeholders make informed decisions.
    """)

    st.header("Methodology")
    st.write("""
    - *Dats Sets Used*: NISR Labour Force Survey 2022/2023,Youth Thematic Report EICV4/EICV5, World Bank, Macrotrends.net.
    - *Development Tools*: Python with Streamlit framework, using libraries like Pandas, NumPy, and Scikit-learn.
    - *Features*: Visualization, predictive modeling, and a recommendation system tailored to individual user inputs.
    """)

    st.header("Implementation")
    st.write("""
    The app is divided into four main sections:
    - *Home*: Overview of the project.
    - *Dashboard*: Data visualizations of unemployment trends.
    - *Prediction*: Machine learning predictions for future trends.
    - *Recommendation*: A tailored system suggesting skills and courses.
    """)

    st.header("Conclusion")
    st.write("""
    This project demonstrates how data-driven solutions can address critical societal challenges like youth unemployment.
    It provides stakeholders with actionable insights and empowers users to take steps toward skill enhancement and career growth.
    """)

# Navigation
st.sidebar.image("Logo.png", use_column_width=True)  # Add the logo at the top of the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Prediction", "Recommendation", "Documentation"])

if page == "Home":
    home()
elif page == "Dashboard":
    dashboard()
elif page == "Prediction":
    prediction()
elif page == "Recommendation":
    recommendation()
elif page == "Documentation":
   documentation()
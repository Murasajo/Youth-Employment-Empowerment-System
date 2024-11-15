# Import necessary packages
import streamlit as st
from dashboard import dashboard
from employment_unemployment import employment_unemployment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from prophet import Prophet
#from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import pickle
import json
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string




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
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 10px;
        text-align: center;
        width: 100%;
        min-width: 150px;
    }
    .metric-box h3 {
        margin: 0;
        font-size: 20px;
        color: #333;
    }
    .metric-box p {
        margin: 5px 0 0;
        font-size: 16px;
        color: #666;
    }
    .plotly-chart {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Home Page
def home():
    st.title("🌍 Youth Employment Empowerment System(YEES) ")
    st.header("Overview")
    st.image("Image0.jpeg", use_column_width=True)  # Add an overview image
    st.write("""
    This project aims to address the unemployment problem in Rwanda by leveraging quality education.
    The application provides insights into current unemployment trends and offers personalized recommendations
    for further education or skill acquisition.
    """)
    with st.expander("See Features"):
        st.write("""
        ### Features:
        - **Home**: The landing page where you can navigate to other sections and learn more about the app.
        - **Dashboard**: Displays current trends in youth employment and unemployment rates in Rwanda, with visual analytics.
        - **Youth Employment/Unemployment**: Provides detailed insights into youth employment and unemployment trends, and factors affecting them.
        - **Prediction**: Predicts future unemployment trends and outcomes based on historical data using machine learning models.
        - **Recommendation**: Offers personalized recommendations for skills and courses to improve employability, based on the user’s profile and goals.
        - **Documentation**: Contains detailed documentation on how to use the app, including explanations of models, data, and features.
        """)


# Prediction Page
def prediction():
    st.title("Prediction Page")
    st.header("Predict Future Unemployment Trends")
    st.write("This page utilizes machine learning models to forecast future unemployment trends based on historical data and offers insights into the key factors affecting unemployment.")
    
    # Load the dataset and models
    @st.cache_data
    def load_dataset():
        with open("dataset.pkl", "rb") as file:
            return pickle.load(file)
    
    @st.cache_resource
    def load_prophet_model():
        with open("prophet_model.pkl", "rb") as file:
            return pickle.load(file)
    
    @st.cache_resource
    def load_xgb_model():
        with open("xgb_model.pkl", "rb") as file:
            return pickle.load(file)
    
    # Load data and models
    df = load_dataset()
    prophet_model = load_prophet_model()
    xgb_model = load_xgb_model()

    # Data Preparation and Forecasting with Prophet
    st.subheader("Forecasting Unemployment with Prophet")
    st.write("Using the Prophet model to forecast the youth unemployment rate over the next five years based on historical data.")

    # Prepare data for Prophet model
    df_prophet = df[['Year', 'Youth_unemployment_rate']].rename(columns={'Year': 'ds', 'Youth_unemployment_rate': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

    # Make future predictions for the next 5 years
    future = prophet_model.make_future_dataframe(periods=5, freq='Y')
    forecast = prophet_model.predict(future)

    # Plot forecast with a compact layout
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Observed Data', marker=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))
    fig_forecast.update_layout(
        title="Youth Unemployment Rate Forecast (Next 5 Years)",
        xaxis_title="Year",
        yaxis_title="Youth Unemployment Rate",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Solution Approach with XGBoost Model
    st.subheader("Solution Approach Using XGBoost Model")
    st.write("An XGBoost model is trained on historical data to predict youth unemployment rates and analyze feature importance for interpretability.")

    # Data Preprocessing
    X = df.drop(['Year', 'Youth_unemployment_rate'], axis=1)
    y = df['Youth_unemployment_rate']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Predictions and Evaluation
    y_pred = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Model Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Feature Importance and SHAP Values in One Row
    st.subheader("Feature Importance and SHAP Analysis")
    st.write("The feature importance and SHAP value plots provide insight into factors influencing the model's predictions.")

    # Use columns to place the feature importance and SHAP summary plots side by side
    col1, col2 = st.columns(2)

    # Feature Importance Plot
    with col1:
        st.write("### Feature Importance")
        feature_importance = xgb_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        sorted_features = X.columns[sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        
        fig_importance = go.Figure(go.Bar(x=sorted_importance, y=sorted_features, orientation='h', marker=dict(color='skyblue')))
        fig_importance.update_layout(
            xaxis_title="Importance",
            yaxis_title="Features",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    # SHAP Values Plot
    with col2:
        st.write("### SHAP Values for Feature Contribution")
        explainer = shap.Explainer(xgb_model, X_test)
        shap_values = explainer(X_test)

        # SHAP summary plot with transparency
        fig, ax = plt.subplots(figsize=(8, 3))

        # Set transparent background for the figure
        fig.patch.set_alpha(0.0)  # Makes the figure background transparent
        ax.patch.set_alpha(0.0)   # Makes the axis background transparent

        # Generate SHAP plot
        shap.summary_plot(
            shap_values, 
            features=X_test, 
            feature_names=X.columns, 
            plot_type="bar", 
            show=False
        )

        # Display the plot in Streamlit
        st.pyplot(fig)

    # Predicted vs Actual Values Plot
    st.subheader("Predicted vs Actual Youth Unemployment Rates")
    st.write("This plot compares the model's predictions with actual unemployment rates to visualize prediction accuracy.")
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines+markers', name='Actual'))
    fig_comparison.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines+markers', name='Predicted'))
    fig_comparison.update_layout(
        title="Actual vs Predicted Youth Unemployment Rates",
        xaxis_title="Index",
        yaxis_title="Youth Unemployment Rate",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Model and Data Loading Section
    st.subheader("Loading and Saving Models")
    st.write("Both the models and datasets are loaded from pickle files to ensure consistency in prediction and interpretation for future use.")

# Skill & Course Recommendation System
def recommendation():
    st.title("🎓 Skill & Course Recommendation System")
    st.header("Personalized Recommendations")
    st.write("Please answer the following questions to get tailored recommendations.")

    # Initialize session state variables
    def init_session_state():
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'user_responses' not in st.session_state:
            st.session_state.user_responses = {}
        if 'recommendation_made' not in st.session_state:
            st.session_state.recommendation_made = False

    # Load career data
    @st.cache_data
    def load_career_data():
        try:
            with open('career.json', 'r') as file:
                return json.loads(file.read())
        except FileNotFoundError:
            st.error("Career data file not found. Please ensure 'career_data.json' exists in the current directory.")
            return []
        except json.JSONDecodeError:
            st.error("Error reading career data file. Please ensure it contains valid JSON.")
            return []

    # Define questions
    questions = [
        "What is your name?",
        "What is your gender?",
        "What was your course in UG?",
        "What is your UG specialization? Major Subject (Eg; Mathematics)",
        "What are your interests?",
        "What are your skills ? (Select multiple if necessary)",
        "What was the average CGPA or Percentage obtained in under graduation?",
        "Did you do any certification courses additionally?",
        "If yes, please specify your certificate course title.",
        "Are you working?",
        "If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA.",
        "Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)"
    ]

    def get_career_recommendations(user_responses, career_data):
        if not career_data:
            return []
            
        # Create feature text for user combining interests and skills
        user_features = f"{user_responses.get('What are your interests?', '')} {user_responses.get('What are your skills ? (Select multiple if necessary)', '')}"
        
        # Create feature texts for all careers in the dataset
        career_features = [f"{entry['What are your interests?']} {entry['What are your skills ? (Select multiple if necessary)']}" 
                        for entry in career_data]
        
        # Add user features to the list
        all_features = career_features + [user_features]
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_features)
            
            # Calculate similarity between user and all careers
            user_vector = tfidf_matrix[-1]
            career_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(user_vector, career_vectors)[0]
            
            # Get top 3 similar careers
            top_indices = similarities.argsort()[-3:][::-1]
            
            recommendations = []
            for idx in top_indices:
                career = career_data[idx]
                similarity_score = similarities[idx]
                if similarity_score > 0:  # Only include if there's some similarity
                    recommendations.append({
                        'profile': career,
                        'similarity': similarity_score
                    })
            
            return recommendations
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []

    def render_question(question_index):
        if 0 <= question_index < len(questions):
            current_question = questions[question_index]
            
            # Display current question and get user input
            if current_question == "What are your skills ? (Select multiple if necessary)":
                return st.text_input(
                    current_question,
                    key=f"q_{question_index}",
                    help="Enter skills separated by semicolons (;)"
                )
            elif current_question == "What was the average CGPA or Percentage obtained in under graduation?":
                return st.number_input(
                    current_question,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key=f"q_{question_index}"
                )
            elif current_question == "Did you do any certification courses additionally?":
                return st.selectbox(
                    current_question,
                    ["Yes", "No"],
                    key=f"q_{question_index}"
                )
            else:
                return st.text_input(current_question, key=f"q_{question_index}")
        return None

    def main():
        st.title("Career Recommendation System")
        
        # Initialize session state
        init_session_state()
        
        # Load career data
        career_data = load_career_data()
        
        if not career_data:
            st.warning("Unable to load career data. Please check if the data file exists and is properly formatted.")
            return
        
        if not st.session_state.recommendation_made:
            # Ensure current_question is within bounds
            if st.session_state.current_question >= len(questions):
                st.session_state.current_question = 0
            
            st.write(f"Question {st.session_state.current_question + 1} of {len(questions)}")
            
            # Render current question
            user_input = render_question(st.session_state.current_question)
            
            # Handle navigation
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Previous") and st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                    st.rerun()
            
            with col2:
                if st.button("Next"):
                    if user_input is not None:
                        st.session_state.user_responses[questions[st.session_state.current_question]] = user_input
                        
                        if st.session_state.current_question < len(questions) - 1:
                            st.session_state.current_question += 1
                        else:
                            st.session_state.recommendation_made = True
                        st.rerun()
        
        else:
            # Display recommendations
            st.write("### Based on your responses, here are your career recommendations:")
            
            recommendations = get_career_recommendations(st.session_state.user_responses, career_data)
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    profile = rec['profile']
                    similarity = rec['similarity']
                    
                    st.write(f"\n#### Recommendation {i} (Similarity: {similarity:.2%})")
                    st.write(f"**Profile Match:**")
                    st.write(f"- Education: {profile['What was your course in UG?']} in {profile['What is your UG specialization? Major Subject (Eg; Mathematics)']}")
                    st.write(f"- Skills: {profile['What are your skills ? (Select multiple if necessary)']}")
                    st.write(f"- Recommended Role: {profile['If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA.']}")
                    
                    if profile['Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)']:
                        st.write(f"- Further Education: {profile['Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)']}")
            else:
                st.warning("No recommendations could be generated based on your responses.")
            
            if st.button("Start Over"):
                st.session_state.current_question = 0
                st.session_state.user_responses = {}
                st.session_state.recommendation_made = False
                st.rerun()

    if __name__ == "__main__":
        main()

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
    - **Data Sets Used**: 
    - NISR Labour Force Survey 2022/2023
    - Youth Thematic Report EICV4/EICV5
    - World Bank
    - Macrotrends.net
    - **Development Tools**: 
    - Python with the Streamlit framework
    - Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Keras-TensorFlow, Plotly, Seaborn, and GeoPandas.
    - **Models Used**: 
    - Machine Learning: Random Forest Classifier, XGBoost Regressor
    - Deep Learning: Neural Networks using Keras-TensorFlow
    - Time Series Analysis: Prophet for forecasting
    - NLP: TF-IDF and cosine similarity for recommendation systems
    - **Features**: Visualization, predictive modeling, and a recommendation system tailored to individual user inputs.
    """)

    st.header("Implementation")
    st.write("""
    The app is divided into four main sections:
    - **Home**: Overview of the project.
    - **Dashboard**: Data visualizations of unemployment trends.
    - **Prediction**: Machine learning predictions for future trends.
    - **Recommendation**: A tailored system suggesting skills and courses.
    """)

    st.header("Conclusion")
    st.write("""
    This project demonstrates how data-driven solutions can address critical societal challenges like youth unemployment.
    It provides stakeholders with actionable insights and empowers users to take steps toward skill enhancement and career growth.
    The integration of advanced machine learning models, deep learning techniques, and visualization libraries makes this app a robust tool for tackling unemployment.
    """)



# Navigation
st.sidebar.image("Logo.png", use_column_width=True) 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Youth Employment/Unemployment Analysis", "Prediction", "Recommendation", "Documentation"])

if page == "Home":
    home()
elif page == "Dashboard":
    dashboard()
elif page == "Youth Employment/Unemployment Analysis":
    employment_unemployment()
elif page == "Prediction":
    prediction()
elif page == "Recommendation":
    recommendation()

elif page == "Documentation":
    documentation()


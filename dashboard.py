import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set up the layout
st.set_page_config(page_title="Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 10px;
        text-align: center;
    }
    .metric-box h3 {
        margin: 0;
        font-size: 24px;
        color: #333;
    }
    .metric-box p {
        margin: 5px 0 0;
        font-size: 18px;
        color: #666;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content h3 {
        margin: 0;
        font-size: 18px;
        color: #333;
    }
    .sidebar .sidebar-content p {
        margin: 5px 0 0;
        font-size: 14px;
        color: #666;
    }
    .sidebar .sidebar-content img {
        border-radius: 10px;
    }
    .plotly-chart {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def dashboard():
    st.title("Dashboard: Home")

    # Sidebar
    st.sidebar.write("Developed and Maintained by: Samir")
    st.sidebar.write("+255 567 839 840")
    st.sidebar.markdown("### Please filter")

    # Filters
    regions = ["East", "Midwest", "Northeast", "Central"]
    locations = ["Urban", "Rural"]
    construction_types = ["Frame", "Fire Resist", "Masonry", "Metal Clad"]

    selected_regions = st.sidebar.multiselect("Select Region", regions)
    selected_locations = st.sidebar.multiselect("Select Location", locations)
    selected_construction = st.sidebar.multiselect("Select Construction", construction_types)

    # Main Menu
    st.sidebar.markdown("### Main Menu")
    menu_option = st.sidebar.radio("", ["Home", "Progress"])

    # Top Metrics
    st.markdown("### My Excel Workbook")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown('<div class="metric-box"><h3>Total Investment</h3><p>2,482,205,481 TZS</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><h3>Most Frequent</h3><p>847,300 TZS</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><h3>Average</h3><p>4,964,411 TZS</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><h3>Central Earnings</h3><p>2,593,682 TZS</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-box"><h3>Ratings</h3><p>3.51K</p></div>', unsafe_allow_html=True)

    # Investment by State (Line Chart)
    state_data = pd.DataFrame({
        "State": ["Arusha", "Dar es Salaam", "Dodoma", "Iringa", "Kigoma", "Kilimanjaro", "Mwanza"],
        "Investment": np.random.randint(50, 250, size=7)
    })
    fig_line = px.line(state_data, x="State", y="Investment", title="Investment by State")
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # Investment by Business Type (Bar Chart)
    business_data = pd.DataFrame({
        "Business Type": ["Apartment", "Farming", "Office Bldg", "Hospitality", "Retail", "Manufacturing", "Organization", "Construction", "Service", "Other", "Recreation", "Medical", "Education"],
        "Investment": np.random.randint(20, 150, size=13)
    })
    fig_bar = px.bar(business_data, x="Investment", y="Business Type", orientation="h", title="Investment by Business Type")
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # Regions by Ratings (Pie Chart)
    ratings_data = pd.DataFrame({
        "Regions": ["Dodoma", "Kigoma", "Dar es Salaam", "Mwanza", "Arusha", "Kilimanjaro", "Iringa"],
        "Ratings": [50, 18.7, 17.1, 4, 4, 3, 3.2]
    })
    fig_pie = px.pie(ratings_data, values="Ratings", names="Regions", title="Regions by Ratings")
    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    # Footer
    st.write("Manage app")
    st.text("25°C | Clear | 6:12 PM")
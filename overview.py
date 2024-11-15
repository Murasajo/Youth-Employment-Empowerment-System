import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pickle

# Set up the layout
st.set_page_config(page_title="Labor Force Overview 2023", layout="wide")

# Custom CSS for styling to ensure uniform box and text sizes
st.markdown("""
    <style>
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

# Sidebar for selection
st.sidebar.header("Labor Force Overview")
st.sidebar.write("Select the category to display:")
options = ["Participation Rate", "Employment/Unemployment"]
selected_option = st.sidebar.selectbox(
    "Choose the view you'd like to explore:",
    options,
    help="Select a view to change the data perspective."
)


# Load the dataset
@st.cache_data
def load_data():
    # Replace hardcoded data with loading logic
    data = pd.DataFrame({
        "Metric": ["Total Population", "Total Labor Force", "Total Out of Labor Force"],
        "Value": [8100430.0, 4847069.4, 3253360.6]
    })

    # Load each dataset from a .pkl file
    with open("overall_labor_force.pkl", "rb") as f:
        overall_labor_force = pickle.load(f)
    
    with open("male_participation.pkl", "rb") as f:
        male_participation = pickle.load(f)
    
    with open("female_participation.pkl", "rb") as f:
        female_participation = pickle.load(f)
    
    with open("urban_participation.pkl", "rb") as f:
        urban_participation = pickle.load(f)
    
    with open("rural_participation.pkl", "rb") as f:
        rural_participation = pickle.load(f)
    
    with open("education_attainment.pkl", "rb") as f:
        education_attainment = pickle.load(f)

    with open("urban_participation_rate.pkl", "rb") as f:
        urban_participation_rate = pickle.load(f)

    with open("rural_participation_rate.pkl", "rb") as f:
        rural_participation_rate = pickle.load(f)

    return (data, overall_labor_force, male_participation, female_participation,
            urban_participation, rural_participation, education_attainment, urban_participation_rate, rural_participation_rate)

# Main Dashboard Layout
# Load the data
data, overall_labor_force, male_participation, female_participation, urban_participation, rural_participation, education_attainment, urban_participation_rate, rural_participation_rate = load_data()

# Display Overview based on selected option
if selected_option == "Participation Rate":
    st.title("Labor Force Participation Rate Overview(LFS) - 2023")

    # Top Row Metrics
    st.markdown("### Overall Labor force Participation Rate")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-box"><h3>Total Population</h3><p>{data.loc[0, "Value"]}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h3>Total Labor Force</h3><p>{data.loc[1, "Value"]}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h3>Total Out of Labor Force</h3><p>{data.loc[2, "Value"]}</p></div>', unsafe_allow_html=True)

    # Charts for Participation Rate
    st.markdown("### Labor Force Participation Rate Charts")
    row1_col1, row1_col2 = st.columns(2)
    row2_col1 = st.columns([1])[0]

    # Placeholder charts (replace with your chart logic)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=overall_labor_force['Age Group'],
        y=overall_labor_force['Labour force'],
        name='Labour Force',
        marker_color='#4c72b0',
        hovertemplate='Labour Force: %{y}<extra></extra>'
    ))
    fig1.add_trace(go.Bar(
        x=overall_labor_force['Age Group'],
        y=overall_labor_force['out of labour force'],
        name='Out of Labour Force',
        marker_color='#55a868',
        hovertemplate='Out of Labour Force: %{y}<extra></extra>',
        offsetgroup=0
    ))
    fig1.add_trace(go.Scatter(
        x=overall_labor_force['Age Group'],
        y=overall_labor_force['Labour force particiaption rate'],
        mode='lines+markers',
        name='Labour Force Participation Rate',
        marker=dict(color='#ff7f0e', size=8),
        line=dict(width=2),
        yaxis='y2',
        hovertemplate='Participation Rate: %{y}%<extra></extra>'
    ))
    fig1.update_layout(
        title="Total Population, Labor Force Composition, and Participation Rate by Age Group",
        xaxis=dict(title='Age Group'),
        yaxis=dict(title='Population Count'),
        yaxis2=dict(title='Labour Force Participation Rate (%)', overlaying='y', side='right'),
        barmode='stack',
        hovermode="x unified",
        legend=dict(x=1, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        width=800,
        height=600
    )
    row1_col1.plotly_chart(fig1, use_container_width=True)

    # Plot for male and female participation rate by age group
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=male_participation['Age Group'],
        y=male_participation['Labour force particiaption rate'],
        mode='lines+markers',
        name='Male',
        marker=dict(color='#1f77b4', size=8),
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Age Group: %{x}<br>Participation Rate: %{y}%<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=female_participation['Age Group'],
        y=female_participation['Labour force particiaption rate'],
        mode='lines+markers',
        name='Female',
        marker=dict(color='#ff7f0e', size=8),
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='Age Group: %{x}<br>Female Participation Rate: %{y}%<extra></extra>'
    ))
    fig2.update_layout(
        title="Male and Female Labour Force Participation Rate by Age Group",
        xaxis=dict(title='Age Group'),
        yaxis=dict(title='Labour Force Participation Rate (%)'),
        legend=dict(title='Legend', x=0.8, y=1),
        width=700,
        height=500
    )
    row1_col2.plotly_chart(fig2, use_container_width=True)

    
    
    # Participation Rate by Location
    # Create subplots for Urban and Rural participation rates
    
    # Create subplots for Urban and Rural participation rates with a main title
    fig3 = make_subplots(
        rows=1, cols=2, 
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Urban Labor Force Participation Rate", "Rural Labor Force Participation Rate")
    )

    # Urban participation pie chart
    fig3.add_trace(go.Pie(
        labels=urban_participation['Age Group'],
        values=urban_participation['Labour force particiaption rate'],
        hole=0.3,  # Donut-style for better aesthetics
        textinfo='percent',
        hoverinfo='label+percent+value',
        marker=dict(colors=px.colors.qualitative.Set3)
    ), row=1, col=1)

    # Rural participation pie chart
    fig3.add_trace(go.Pie(
        labels=rural_participation['Age Group'],
        values=rural_participation['Labour force particiaption rate'],
        hole=0.3,  # Donut-style
        textinfo='percent',
        hoverinfo='label+percent+value',
        marker=dict(colors=px.colors.qualitative.Set3)
    ), row=1, col=2)

    # Update layout for the main title and aesthetics
    fig3.update_layout(
        title={
            'text': "Urban vs Rural Labor Force Participation Rates by Age Group",
            'x': 0.5,  # Center the title
            'y': 0.95,  # Position above the subplot titles
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=400,
        width=700
    )

    # Display the figure in the full-width column of the second row
    row2_col1.plotly_chart(fig3, use_container_width=True)



elif selected_option == "Employment/Unemployment":
    st.title("Employment and Unemployment Overview - 2023")

    # Employment/Unemployment Metrics
    st.markdown("### Overall labor force Employed and Unemployed")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-box"><h3>Total Population</h3><p>8100430.0</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><h3>Labour Force</h3><p>4847069.4</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><h3>Out of Labor Force</h3><p>3253360.6</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><h3>Total Employed</h3><p>3972193.1</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-box"><h3>Total Unemployed</h3><p>874876.4</p></div>', unsafe_allow_html=True)

    # Charts for Employment/Unemployment
    #st.markdown("### Employment and Unemployment Charts")
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Placeholder charts (replace with your chart logic)
    overall_labor_force['Unemployment to population ratio'] = 100 - overall_labor_force['Employment to population ratio']

    # Create a Plotly figure
    fig1 = go.Figure()

    # Employment ratio bars
    fig1.add_trace(go.Bar(
        x=overall_labor_force['Age Group'],
        y=overall_labor_force['Employment to population ratio'],
        name='Employment Ratio',
        marker_color='#1f77b4',
        text=overall_labor_force['Employment to population ratio'],
        textposition='outside'
    ))

    # Unemployment ratio bars
    fig1.add_trace(go.Bar(
        x=overall_labor_force['Age Group'],
        y=overall_labor_force['Unemployment to population ratio'],
        name='Unemployment Ratio',
        marker_color='#ff7f0e',
        text=overall_labor_force['Unemployment to population ratio'],
        textposition='outside'
    ))

    # Customize layout
    fig1.update_layout(
        title="Employment and Unemployment Ratios by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Ratio",
        barmode='group',
        legend_title="Legend",
        template="plotly_white"
    )

    # Display the interactive plot in Streamlit
    # st.plotly_chart(fig1, use_container_width=True)
    row1_col1.plotly_chart(fig1, use_container_width=True)

    # Calculate Unemployment ratios
    male_participation['Unemployment to population ratio'] = 100 - male_participation['Employment to population ratio']
    female_participation['Unemployment to population ratio'] = 100 - female_participation['Employment to population ratio']

    # Create a Plotly figure
    fig2 = go.Figure()

    # Male Employment and Unemployment Ratios
    fig2.add_trace(go.Bar(
        y=male_participation['Age Group'],
        x=male_participation['Employment to population ratio'],
        name='Male Employment',
        orientation='h',
        marker_color='rgba(31, 119, 180, 0.6)',  # Semi-transparent blue
        text=[f'{x:.1f}' for x in male_participation['Employment to population ratio']],  # One decimal place
        textposition='outside'
    ))
    fig2.add_trace(go.Bar(
        y=male_participation['Age Group'],
        x=male_participation['Unemployment to population ratio'],
        name='Male Unemployment',
        orientation='h',
        marker_color='rgba(255, 127, 14, 0.6)',  # Semi-transparent orange
        text=[f'{x:.1f}' for x in male_participation['Unemployment to population ratio']],  # One decimal place
        textposition='outside'
    ))

    # Female Employment and Unemployment Ratios
    fig2.add_trace(go.Bar(
        y=female_participation['Age Group'],
        x=female_participation['Employment to population ratio'],
        name='Female Employment',
        orientation='h',
        marker_color='rgba(44, 160, 44, 0.6)',  # Semi-transparent green
        text=[f'{x:.1f}' for x in female_participation['Employment to population ratio']],  # One decimal place
        textposition='outside'
    ))
    fig2.add_trace(go.Bar(
        y=female_participation['Age Group'],
        x=female_participation['Unemployment to population ratio'],
        name='Female Unemployment',
        orientation='h',
        marker_color='rgba(214, 39, 40, 0.6)',  # Semi-transparent red
        text=[f'{x:.1f}' for x in female_participation['Unemployment to population ratio']],  # One decimal place
        textposition='outside'
    ))

    # Customize layout
    fig2.update_layout(
        title="Employment and Unemployment Ratios by Age Group and Gender",
        xaxis_title="Ratio (%)",
        #yaxis_title="Age Group",
        barmode='group',
        legend_title="Legend",
        template="plotly_white",
        height=600
    )
    row1_col2.plotly_chart(fig2, use_container_width=True)

    # Create a Plotly figure
    fig3 = go.Figure()

    # Employment bars
    fig3.add_trace(go.Bar(
        x=education_attainment['Education level'],
        y=education_attainment['Employment to population ratio'],
        name='Employed',
        marker_color='rgba(31, 119, 180, 0.7)',  # Blue with transparency
        text=education_attainment['Employment to population ratio'],
        texttemplate='%{text:.1f}',  # Format text to 1 decimal place
        textposition='inside'
    ))

    # Unemployment bars stacked on top
    fig3.add_trace(go.Bar(
        x=education_attainment['Education level'],
        y=education_attainment['Unemployment to population ratio'],
        name='Unemployed',
        marker_color='rgba(255, 127, 14, 0.7)',  # Orange with transparency
        text=education_attainment['Unemployment to population ratio'],
        texttemplate='%{text:.1f}',  # Format text to 1 decimal place
        textposition='inside'
    ))

    # Customize layout
    fig3.update_layout(
        title="Employment and Unemployment Ratios by Education Level",
        xaxis_title="Education Level",
        yaxis_title="Ratio",
        barmode='stack',
        legend_title="Status",
        template="plotly_white",
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
    )
    row2_col1.plotly_chart(fig3, use_container_width=True)

    # By Location Type
    # Data for urban and rural employment and unemployment ratios
    urban_tot_employment_ratio = urban_participation_rate['Employment to population ratio'][0]
    urban_tot_unemployment_ratio = 100 - urban_tot_employment_ratio

    rural_tot_employment_ratio = rural_participation_rate['Employment to population ratio'][0]
    rural_tot_unemployment_ratio = 100 - rural_tot_employment_ratio

    categories = ['Employment Ratio', 'Unemployment Ratio']
    urban_values = [urban_tot_employment_ratio, urban_tot_unemployment_ratio]
    rural_values = [rural_tot_employment_ratio, rural_tot_unemployment_ratio]

    # Create figure and axes with larger size
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10), dpi=100)  # Increase figure size

    # Set transparent background for the figure
    fig4.patch.set_alpha(0)  # Make the figure background transparent
    ax1.patch.set_alpha(0)   # Make the axes background transparent
    ax2.patch.set_alpha(0)   # Make the axes background transparent

    # Color scheme
    colors = ['#2ecc71', '#e74c3c']  # Green for employment, Red for unemployment

    # Urban Pie Chart
    urban_wedges, urban_texts, urban_autotexts = ax1.pie(
        urban_values,
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.85,
        explode=(0.05, 0),  # Slightly explode the employment slice
        startangle=90,
        shadow=True,
        wedgeprops=dict(width=0.5)  # Donut chart effect
    )

    # Add center circle for donut effect
    centre_circle_urban = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(centre_circle_urban)
    ax1.text(0, 0, 'Urban\nTotal', ha='center', va='center', fontweight='bold', fontsize=12)

    # Rural Pie Chart
    rural_wedges, rural_texts, rural_autotexts = ax2.pie(
        rural_values,
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.85,
        explode=(0.05, 0),  # Slightly explode the employment slice
        startangle=90,
        shadow=True,
        wedgeprops=dict(width=0.5)  # Donut chart effect
    )

    # Add center circle for donut effect
    centre_circle_rural = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle_rural)
    ax2.text(0, 0, 'Rural\nTotal', ha='center', va='center', fontweight='bold', fontsize=12)

    # Enhance percentage labels
    for autotext in urban_autotexts + rural_autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    # Enhance category labels
    for text in urban_texts + rural_texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    # Main title
    plt.suptitle('Employment and Unemployment Ratios: Urban vs Rural', 
                fontsize=18, fontweight='bold', y=1.05)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=colors[0], markersize=15, label='Employment'),
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=colors[1], markersize=15, label='Unemployment')
    ]
    fig4.legend(handles=legend_elements, loc='lower center', 
                bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    # Adjust layout manually for better positioning
    plt.subplots_adjust(top=0.85, bottom=0.2)  # Adjust top and bottom spacing

    # Layout adjustment
    plt.tight_layout()

    # Show the plot in Streamlit's row2_col2 layout with transparent background
    row2_col2.pyplot(fig4)

# Footer
st.write("Developed and Maintained by FutureForge Pioneers")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pickle
import folium
import json
import geopandas as gpd
from branca.colormap import LinearColormap
from datetime import datetime


def employment_unemployment():

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            color: white;
        }
        .css-1d391kg {
            background-color: pink;
        }
        .stMetric {
            background-color: #3700ff;
            padding: 20px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # data generation
    def generate_data():
        with open("merged_data.pkl", "rb") as f:
            merged_data = pickle.load(f)
        
        with open("employment_pop_ratio.pkl", "rb") as f:
            employment_pop_ratio = pickle.load(f)
        
        with open("employment_by_occupation_category.pkl", "rb") as f:
            employment_by_occupation_category = pickle.load(f)
        
        with open("melted_df.pkl", "rb") as f:
            melted_df = pickle.load(f)

        with open("unemployment_data.pkl", "rb") as f:
            df = pickle.load(f)

        return (merged_data, employment_pop_ratio, employment_by_occupation_category, melted_df, df)

    # Create sidebar
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.radio("Select Analysis", ["Employment", "Unemployment"])

    # Generate sample data
    merged_data, employment_pop_ratio, employment_by_occupation_category, melted_df, df = generate_data()

    # Main dashboard layout
    st.title(f"{page} Status")
    st.markdown(
        f'<h3 style="font-size: 15px; color: black;">({page} Status among the Rwandan youth aged 0-25 years)</h3>',
        unsafe_allow_html=True
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    if page == "Employment":
        # Employment metrics
        with col1:
            st.markdown("""
                <div class="custom-metric">
                    <div>Total Youth Population</div>
                    <div class="custom-metric-value">4,241,657</div>
                    <div class="custom-metric-change">-0.068%</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="custom-metric">
                    <div>Total Employed Youth Population</div>
                    <div class="custom-metric-value">2,080,003</div>
                    <div class="custom-metric-change">16.09%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="custom-metric">
                    <div>Total Employed Youth aged 16-24_yrs</div>
                    <div class="custom-metric-value">909,629</div>
                    <div class="custom-metric-change">22.34%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="custom-metric">
                    <div>Employement to pop_ratio age 16-24_yrs(%)</div>
                    <div class="custom-metric-value">38.3%</div>
                    <div class="custom-metric-change">8.4%</div>
                </div>
            """, unsafe_allow_html=True)

        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall youth population status in Rwanda
            # Add spacing and adjust placement
            st.markdown(
                """
                <style>
                .spacer { 
                    height: 100px; 
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Spacer above the chart

            # Add title above the chart
            st.markdown(
                """
                <h3 style="text-align: center; font-size: 16px; margin-top: -50px;">
                Youth Population Rate (16-25 years) by District in Rwanda
                </h3>
                """, 
                unsafe_allow_html=True
            )

            # Initialize Folium map centered on Rwanda
            rwanda_map = folium.Map(location=[-1.9403, 29.8739], zoom_start=8)

            # Create a colormap for the Tot_rate_16-25year field
            colormap = LinearColormap(
                colors=['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
                vmin=merged_data['Tot_rate_16-25year'].min(),
                vmax=merged_data['Tot_rate_16-25year'].max()
            )

            # Add choropleth layer to visualize Tot_rate_16-25year
            folium.Choropleth(
                geo_data=merged_data.__geo_interface__,
                name='Youth Population Aged 16-25year by Distr',
                data=merged_data,
                columns=['NAME_2', 'Tot_rate_16-25year'],
                key_on='feature.properties.NAME_2',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Youth Population Rate by District (%)',
                highlight=True
            ).add_to(rwanda_map)

            # Add tooltips with district information
            folium.GeoJson(
                merged_data,
                style_function=lambda x: {'fillColor': '#ffffff', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.1},
                control=False,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['NAME_2', 'Tot_rate_16-25year', 'Total Population'],
                    aliases=['District:', 'Youth Pop Rate (%):', 'Total Population:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            ).add_to(rwanda_map)

            # Resize the map for Streamlit
            map_html = rwanda_map._repr_html_()  # Get the HTML representation of the map
            st.components.v1.html(map_html, height=700)

        with col2:
            # Employment to population ratio, ages 15-24,
            # Create the interactive plot
            fig1 = go.Figure()

            # Add Female employment line
            fig1.add_trace(go.Scatter(
                x=employment_pop_ratio["Year"], y=employment_pop_ratio["Female"],
                mode="lines+markers",
                marker=dict(color="salmon", size=8),
                line=dict(color="salmon", width=2),
                name="Female"
            ))

            # Add Male employment line
            fig1.add_trace(go.Scatter(
                x=employment_pop_ratio["Year"], y=employment_pop_ratio["Male"],
                mode="lines+markers",
                marker=dict(color="skyblue", size=8),
                line=dict(color="skyblue", width=2),
                name="Male"
            ))

            # Set title and labels
            fig1.update_layout(
                title=dict(text="Youth Employment Status on Population Ratio Over Time", font=dict(size=15, color="black")),
                xaxis_title="Year",
                yaxis_title="Employment Ratio (%)",
                font=dict(color="white"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5, zeroline=False),
                legend_title_text="Gender"
            )

            # Customize the legend and add grid
            fig1.update_xaxes(tickfont=dict(size=12))
            fig1.update_yaxes(tickfont=dict(size=12))
            fig1.update_layout(legend=dict(font=dict(size=12), title_font=dict(size=13)))

            st.plotly_chart(fig1, use_container_width=True)

        # Distribution of usually employed youth
        # Create an interactive bar plot
        fig2 = px.bar(
            melted_df, 
            x='Count', 
            y='Occupation group of main usual job (ISCO 1 digit)', 
            color='Category', 
            orientation='h',
            title='Distribution of Occupations by Category'
        )

        # Customize layout appearance
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=20,
            xaxis_title="",
            yaxis_title="",
            legend_title="Category"
        )
        st.plotly_chart(fig2, use_container_width=True)


    else:  # Unemployment dashboard
        # Unemployment metrics
        with col1:
            st.metric("Total Youth Population(16-34)", "4,241,657", "-0.068%")
        with col2:
            st.metric("Total Unemployed Youth Pop.", "549,759", "-3.44%")
        with col3:
            st.metric("Total Unemployed Youth aged (16-24)", "280,990", "-0.36%")
        with col4:
            st.metric("Unemployment Rate 16-24 (%)", "23.6%", "-14.18%")


        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # By education level
            # Initialize figure
            fig4 = go.Figure()

            # Add traces for each education level
            fig4.add_trace(go.Bar(
                x=df['Year'],
                y=df['Unemployment_Basic (%)'],
                name='Basic',
                marker_color='#ff9999'
            ))

            fig4.add_trace(go.Bar(
                x=df['Year'],
                y=df['Unemployment_Intermediate (%)'],
                name='Intermediate',
                marker_color='#66b3ff',
                offsetgroup=0
            ))

            fig4.add_trace(go.Bar(
                x=df['Year'],
                y=df['Unemployment_Advanced (%)'],
                name='Advanced',
                marker_color='#99ff99',
                offsetgroup=0
            ))

            # Update layout to stack bars
            fig4.update_layout(
                barmode='stack',
                title={
                    'text': "Youth Unemployment Rate<br>by Education Level",  # Add a line break using <br>
                    'x': 0.5,  # Center align the title
                    'xanchor': 'center',
                    'yanchor': 'top',
                },
                xaxis_title="Year",
                yaxis_title="Unemployment Rate (%)",
                legend_title="Education Level",
                template="plotly_white"
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            #  Economic Indicators
            fig5 = px.scatter(
                df,
                x='GDP_Growth_Rate(%)',  # X-axis
                y='Youth_unemployment_rate',  # Y-axis
                size='Population_growth_rate_per(%)',  # Bubble size
                color='Economic_growth_rate(%)', 
                hover_name='Year',  # Label for hover information
                title='Relationship between GDP, Economic growth, Population growth and Youth Unemployment Rate',
                labels={
                    'GDP_Growth_Rate': 'GDP Growth Rate',
                    'Youth_unemployment_rate': 'Youth Unemployment Rate (%)',
                    'Economic_growth_rate': 'Economic Growth Rate (%)',
                    'Population_growth_rate_per(%)': 'Population Growth Rate(%)'
                }
            )

            # Customize layout for better readability
            fig5.update_layout(
                xaxis_title='GDP Growth Rate',
                yaxis_title='Youth Unemployment Rate (%)',
                title_x=0.5,
                legend_title=""
            )
            st.plotly_chart(fig5, use_container_width=True)

        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Youth Labor Force Participation
            # Pie chart showing the disparity between Male and female Literacy rate over time
            fig6 = go.Figure()

            # Add line for Female Youth Literacy Rate
            fig6.add_trace(
                go.Scatter(
                    x=df['Year'],
                    y=df['Female_youth_literacy_rate%'],
                    mode='lines+markers',
                    name='Female',
                    line=dict(color='red'),
                    marker=dict(size=6)
                )
            )

            # Add line for Male Youth Literacy Rate
            fig6.add_trace(
                go.Scatter(
                    x=df['Year'],
                    y=df['Male_youth_literacy_rate(%)'],
                    mode='lines+markers',
                    name='Male',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                )
            )

            # Update layout for better readability
            fig6.update_layout(
                title="Disparity Between Male and Female Youth Literacy Rates Over Time",
                xaxis_title="Year",
                yaxis_title="Youth Literacy Rate (%)",
                legend_title="Gender",
                hovermode="x",
                template="plotly_white"
            )


            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            # Literacy and Skill Development
            fig7 = px.scatter(
                df,
                x="Total_youth_literacy_rate%",
                y="Youth_unemployment_rate",
                title="Relationship Between Total Youth Literacy Rate and Youth Unemployment Rate",
                labels={
                    "Total_youth_literacy_rate%": "Total Youth Literacy Rate (%)",
                    "Youth_unemployment_rate": "Youth Unemployment Rate (%)"
                },
                template="plotly_white"
            )

            # Update marker settings for better visibility
            fig7.update_traces(marker=dict(size=8, color='teal', line=dict(width=1, color='DarkSlateGrey')))

            # Customize layout
            fig7.update_layout(
                xaxis_title="Total Youth Literacy Rate (%)",
                yaxis_title="Youth Unemployment Rate (%)",
                hovermode="closest"
            )

            st.plotly_chart(fig7, use_container_width=True)


        # Gender disparities in unemployment rates provide insights into the unique challenges faced by male and female youth.
        fig8 = sp.make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "scatter"}]],
                        subplot_titles=('Gender Distribution (2023)', 'Unemployment Rate Trends'))

        # Add pie chart for the most recent year
        pie_data = df.iloc[-1]  # Get the most recent year's data
        fig8.add_trace(
            go.Pie(
                labels=['Male', 'Female'],
                values=[
                    pie_data['Male_unemployment_rate_NL(%)'],
                    pie_data['Female_unemployement_rate_NL(%)']
                ],
                hole=0.3,
                marker_colors=['#0088FE', '#FF8042']
            ),
            row=1, col=1
        )

        # Add line chart for trends
        fig8.add_trace(
            go.Scatter(
                x=df['Year'],
                y=df['Male_unemployment_rate_NL(%)'],
                name='Male',
                line=dict(color='#0088FE', width=2)
            ),
            row=1, col=2
        )

        fig8.add_trace(
            go.Scatter(
                x=df['Year'],
                y=df['Female_unemployement_rate_NL(%)'],
                name='Female',
                line=dict(color='#FF8042', width=2)
            ),
            row=1, col=2
        )

        # Update layout
        fig8.update_layout(
            title_text='Unemployment Rates by Gender',
            showlegend=True,
            height=400,
            width=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.1
            )
        )

        # Update axes labels
        fig8.update_yaxes(title_text='Unemployment Rate (%)', row=1, col=2)
        fig8.update_xaxes(title_text='Year', row=1, col=2)

        st.plotly_chart(fig8, use_container_width=True)



        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            #  NEET (Not in Education, Employment, or Training)
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=df['Year'], y=df['NEET_Female (%)'], mode='lines+markers', name='Female NEET', line=dict(color='pink')))
            fig9.add_trace(go.Scatter(x=df['Year'], y=df['NEET_Male (%)'], mode='lines+markers', name='Male NEET', line=dict(color='blue')))
            fig9.add_trace(go.Scatter(x=df['Year'], y=df['NEET_Total (%)'], mode='lines+markers', name='Total NEET', line=dict(color='purple')))

            fig9.update_layout(title="NEET Rates Over Time by Gender", xaxis_title="Year", yaxis_title="NEET Rate (%)", template="plotly_white")


            st.plotly_chart(fig9, use_container_width=True)

        with col2:
            # Youth Labor Force Participation
            fig10 = make_subplots(rows=1, cols=2, subplot_titles=(
                "Participation Rate vs Unemployment Rate",
                "Employment-to-Population Ratio vs Unemployment Rate"))

            # Scatter plot 1: Participation Rate vs Youth Unemployment Rate
            fig10.add_trace(
                go.Scatter(
                    x=df['Participation_Rate_For_Ages(15-24)%'],
                    y=df['Youth_unemployment_rate'],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name="Participation Rate (15-24)"
                ),
                row=1, col=1
            )

            # Scatter plot 2: Employment-to-Population Ratio vs Youth Unemployment Rate
            fig10.add_trace(
                go.Scatter(
                    x=df['Total_employement_to_population_ratio(%)'],
                    y=df['Youth_unemployment_rate'],
                    mode='markers',
                    marker=dict(color='green', size=8),
                    name="Employment-to-Population Ratio"
                ),
                row=1, col=2
            )

            # Update layout for better visualization
            fig10.update_layout(
                title="Relationship Between Youth Unemployment Rate and Key Indicators",
                xaxis_title="Participation Rate (15-24) (%)",
                xaxis2_title="Employment-to-Population Ratio (%)",
                yaxis_title="Youth Unemployment Rate (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig10, use_container_width=True)

    

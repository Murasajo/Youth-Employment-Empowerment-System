import streamlit as st

def recommendation_page():
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

# Run the function to display the recommendation page
recommendation_page()

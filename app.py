import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page layout and configuration
st.set_page_config(
    page_title="Course Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ `model.pkl` file not found. Please place it in the same directory as `app.py`.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# Header Section
st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate student outcome using KNN Regression based on course workload and study hours.</div>', unsafe_allow_html=True)

st.divider()

# Layout: 2 Columns for Input and Output
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("📋 Input Parameters")
    
    with st.form("prediction_form"):
        # Input 1: Number of Courses
        number_courses = st.number_input(
            label="Number of Courses",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Select total number of enrolled courses."
        )
        
        # Input 2: Study Time
        time_study = st.slider(
            label="Study Time (Hours)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.25,
            help="Total daily or weekly study hours."
        )
        
        submit_btn = st.form_submit_button("Predict Outcome", use_container_width=True)

with col_results:
    st.subheader("🎯 Prediction Result")
    
    if submit_btn and model is not None:
        # Construct DataFrame matching model's feature names
        input_data = pd.DataFrame([[number_courses, time_study]], 
                                  columns=['number_courses', 'time_study'])
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            st.success("Prediction generated successfully!")
            
            st.metric(
                label="Predicted Value / Score",
                value=f"{prediction:.2f}"
            )
            
            # Additional input summary breakdown
            with st.expander("🔍 View Input Summary", expanded=True):
                st.dataframe(input_data, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            
    elif submit_btn and model is None:
        st.warning("Model is not loaded. Please verify your `model.pkl` file.")
    else:
        st.info("👈 Adjust the input parameters on the left and click **Predict Outcome** to get results.")

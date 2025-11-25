import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and feature columns
@st.cache_resource
def load_model():
    """Load the saved model and feature columns"""
    try:
        # Try loading both models and pick the one that exists
        try:
            with open('model/logistic_regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_name = "Logistic Regression"
        except:
            with open('model/random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_name = "Random Forest"
        
        with open('model/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        return model, feature_cols, model_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def prepare_input(gender, race, parental_edu, lunch, test_prep, feature_cols):
    """
    Prepare user input for prediction
    - Creates a dataframe with user selections
    - One-hot encodes it
    - Matches the training feature columns
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [parental_edu],
        'lunch': [lunch],
        'test preparation course': [test_prep]
    })
    
    # One-hot encode (same as training)
    input_encoded = pd.get_dummies(input_data, columns=[
        'gender', 
        'race/ethnicity', 
        'parental level of education', 
        'lunch', 
        'test preparation course'
    ])
    
    # Ensure all training columns are present (fill missing with 0)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_cols]
    
    return input_encoded

def main():
    # Page configuration
    st.set_page_config(
        page_title="College Readiness Predictor",
        page_icon="🎓",
        layout="wide"
    )
    
    # Title and description
    st.title("🎓 College Readiness Prediction System")
    st.markdown("""
    This ML system predicts whether a student is **college-ready** based on their demographic 
    and educational background. College readiness is defined as achieving an average score ≥ 75 
    across Math, Reading, and Writing assessments.
    """)
    
    # Loading model
    model, feature_cols, model_name = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run main.py first to train the model.")
        st.stop()
    
    st.success(f"✅ Model loaded: **{model_name}**")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Model Type:** {model_name}
        
        **Features Used:**
        - Gender
        - Race/Ethnicity
        - Parental Education Level
        - Lunch Type (Economic Status)
        - Test Preparation Course
        
        **Target:** College Ready (Avg Score ≥ 75)
        
        **Total Features:** {len(feature_cols)}
        """)
    
    # Main content
    st.header("Enter Student Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        
        gender = st.selectbox(
            "Gender",
            options=['female', 'male'],
            help="Student's gender"
        )
        
        race = st.selectbox(
            "Race/Ethnicity",
            options=['group A', 'group B', 'group C', 'group D', 'group E'],
            help="Demographic group"
        )
        
        parental_edu = st.selectbox(
            "Parental Education Level",
            options=[
                'some high school',
                'high school',
                'some college',
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ],
            help="Highest education level of parents"
        )
    
    with col2:
        st.subheader("Academic Background")
        
        lunch = st.selectbox(
            "Lunch Type",
            options=['standard', 'free/reduced'],
            help="Indicates socioeconomic status"
        )
        
        test_prep = st.selectbox(
            "Test Preparation Course",
            options=['none', 'completed'],
            help="Whether student completed test prep course"
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict College Readiness", type="primary", use_container_width=True):
        # Prepare input
        input_data = prepare_input(gender, race, parental_edu, lunch, test_prep, feature_cols)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.success("### ✅ COLLEGE READY")
                st.markdown(f"""
                The model predicts this student **is likely to be college-ready** 
                (average score ≥ 75).
                
                **Confidence:** {prediction_proba[1]:.1%}
                """)
                
                # Progress bar
                st.progress(prediction_proba[1])
                
                st.info("""
                **💡 Recommendation:**
                - Continue supporting the student's academic progress
                - Encourage advanced coursework
                - Provide college application guidance
                """)
            else:
                st.warning("### ⚠️ NOT COLLEGE READY")
                st.markdown(f"""
                The model predicts this student **may not be college-ready** 
                (average score < 75).
                
                **Confidence:** {prediction_proba[0]:.1%}
                """)
                
                # Progress bar
                st.progress(prediction_proba[0])
                
                st.info("""
                **💡 Recommendation:**
                - Provide additional academic support
                - Enroll in test preparation programs
                - Consider tutoring services
                - Schedule parent-teacher conferences
                - Create personalized learning plan
                """)
        
        # Detailed probabilities
        st.markdown("---")
        st.subheader("Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Outcome': ['Not College Ready', 'College Ready'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        
        st.bar_chart(prob_df.set_index('Outcome'))
    
    # Batch prediction section
    st.markdown("---")
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file with multiple students to get predictions for all at once.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should have columns: gender, race/ethnicity, parental level of education, lunch, test preparation course"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Prediction"):
                # One-hot encode batch data
                batch_encoded = pd.get_dummies(batch_data, columns=[
                    'gender', 
                    'race/ethnicity', 
                    'parental level of education', 
                    'lunch', 
                    'test preparation course'
                ])
                
                # Ensure all columns match
                for col in feature_cols:
                    if col not in batch_encoded.columns:
                        batch_encoded[col] = 0
                
                batch_encoded = batch_encoded[feature_cols]
                
                # Predict
                predictions = model.predict(batch_encoded)
                probabilities = model.predict_proba(batch_encoded)
                
                # Add results to original data
                batch_data['Prediction'] = ['College Ready' if p == 1 else 'Not Ready' for p in predictions]
                batch_data['Confidence'] = [max(prob) for prob in probabilities]
                
                st.success(f"✅ Predictions complete for {len(batch_data)} students!")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Students", len(batch_data))
                
                with col2:
                    college_ready = (predictions == 1).sum()
                    st.metric("College Ready", f"{college_ready} ({college_ready/len(batch_data)*100:.1f}%)")
                
                with col3:
                    not_ready = (predictions == 0).sum()
                    st.metric("Not Ready", f"{not_ready} ({not_ready/len(batch_data)*100:.1f}%)")
                
                # Show results
                st.dataframe(batch_data)
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="college_readiness_predictions.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
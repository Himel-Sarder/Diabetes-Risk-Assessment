import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import shap
import joblib


# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* ... (keep your existing CSS styles) ... */
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, names=columns)
    
    # Preprocessing
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df[features] = df[features].replace(0, np.nan)
    
    for feature in features:
        df.loc[(df['Outcome'] == 0) & (df[feature].isnull()), feature] = df[df['Outcome'] == 0][feature].median()
        df.loc[(df['Outcome'] == 1) & (df[feature].isnull()), feature] = df[df['Outcome'] == 1][feature].median()
    
    return df

# Backgroud
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.postimg.cc/7YPFZGgV/1702679947586734.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Train and cache model
@st.cache_resource
def train_model(df):
    # Feature engineering
    df['Insulin_Status'] = pd.cut(df['Insulin'], 
                                 bins=[0, 16, 166, float('inf')],
                                 labels=['Low', 'Normal', 'High'])
    
    df['Glucose_Status'] = pd.cut(df['Glucose'],
                                 bins=[0, 70, 99, 126, float('inf')],
                                 labels=['Low', 'Normal', 'Overweight', 'High'])
    
    df = pd.get_dummies(df, columns=['Insulin_Status', 'Glucose_Status'], drop_first=True)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scale numerical features
    num_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    robust_scaler = RobustScaler()
    X[num_features] = robust_scaler.fit_transform(X[num_features])
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    return model, robust_scaler, X.columns

# Prediction function
def predict_diabetes(input_data, model, scaler, columns):
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Feature engineering
    input_df['Insulin_Status'] = pd.cut(input_df['Insulin'], 
                                       bins=[0, 16, 166, float('inf')],
                                       labels=['Low', 'Normal', 'High'])
    
    input_df['Glucose_Status'] = pd.cut(input_df['Glucose'],
                                      bins=[0, 70, 99, 126, float('inf')],
                                      labels=['Low', 'Normal', 'Overweight', 'High'])
    
    input_df = pd.get_dummies(input_df, columns=['Insulin_Status', 'Glucose_Status'], drop_first=True)
    
    # Ensure all columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[columns]
    
    # Scale numerical features
    num_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df[num_features] = scaler.transform(input_df[num_features])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    return prediction[0], probability, input_df

# Main app
def main():
    # Load data and train model
    df = load_data()
    model, scaler, model_columns = train_model(df)
    
    # Header
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Create columns layout
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title"> Patient Information</h3>', unsafe_allow_html=True)
        
        # Input form
        with st.form("patient_form"):
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
            blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
            insulin = st.slider("Insulin Level (μU/mL)", 0, 846, 80)
            bmi = st.slider("BMI", 0.0, 67.1, 25.0, step=0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, step=0.01)
            age = st.slider("Age", 21, 81, 30)
            
            submitted = st.form_submit_button("Assess Diabetes Risk")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk factors info
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">⚠️ Diabetes Risk Factors</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        - **High glucose levels** (>126 mg/dL fasting)
        - **BMI** over 25 (overweight) or 30 (obese)
        - **Age** over 45 years
        - **Family history** of diabetes
        - **Physical inactivity**
        - **High blood pressure** (over 140/90 mmHg)
        - **Abnormal cholesterol levels**
        - **History of gestational diabetes**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Prediction result
        if submitted:
            # Prepare input data
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            # Get prediction
            prediction, probability, full_input_df = predict_diabetes(input_data, model, scaler, model_columns)
            
            # Display result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 class="card-title"> Prediction Result</h3>', unsafe_allow_html=True)
            
            # Risk level
            if probability >= 0.7:
                st.markdown(f'<div class="card risk-high">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ High Risk of Diabetes ({probability*100:.1f}%)")
            elif probability >= 0.4:
                st.markdown(f'<div class="card risk-medium">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ Moderate Risk of Diabetes ({probability*100:.1f}%)")
            else:
                st.markdown(f'<div class="card risk-low">', unsafe_allow_html=True)
                st.markdown(f"### ✅ Low Risk of Diabetes ({probability*100:.1f}%)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Interpretation
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                **Recommendations:**
                - Consult with a healthcare provider
                - Monitor blood sugar levels regularly
                - Adopt a balanced diet and regular exercise
                - Consider annual diabetes screening
                """)
            else:
                st.markdown("""
                **Recommendations:**
                - Maintain healthy lifestyle habits
                - Get regular health check-ups
                - Monitor risk factors periodically
                """)
            st.markdown('</div>', unsafe_allow_html=True)
            

        
        # Data visualizations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📈 Diabetes Data Analysis</h3>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Glucose vs Outcome", "Age Distribution", "BMI Impact"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Outcome', y='Glucose', data=df, palette='Set2', ax=ax)
            ax.set_title('Glucose Levels by Diabetes Outcome', fontsize=14)
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            ax.set_ylabel('Glucose (mg/dL)')
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='Age', hue='Outcome', bins=20, kde=True, 
                         palette={0: '#0ea5e9', 1: '#ef4444'}, ax=ax)
            ax.set_title('Age Distribution by Diabetes Outcome', fontsize=14)
            ax.set_xlabel('Age')
            st.pyplot(fig)
            
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='BMI', y='Glucose', hue='Outcome', 
                           palette={0: '#0ea5e9', 1: '#ef4444'}, alpha=0.7, ax=ax)
            ax.set_title('BMI vs Glucose with Diabetes Outcome', fontsize=14)
            ax.set_xlabel('Body Mass Index (BMI)')
            ax.set_ylabel('Glucose (mg/dL)')
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #64748b;">
        <p>This diabetes risk assessment tool uses a machine learning model trained on clinical data.</p>
        <p>It is intended for informational purposes only and not a substitute for professional medical advice.</p>
        <p>Always consult with a healthcare provider for medical diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
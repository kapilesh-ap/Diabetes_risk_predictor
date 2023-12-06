import pandas as pd
import joblib
import streamlit as st
from xgboost import XGBClassifier
import os
from sklearn.preprocessing import StandardScaler

# Header
st.write("""
# Big Data Assignment
# Diabetes Risk Prediction App
Answer 6 questions to find out if you may be at risk for Type II Diabetes.
""")

with st.expander("Click for FAQ:"):
    st.write("""
    * **Questions**:
        1. Weight
        2. Height
        3. Age
        4. High Cholesterol
        5. High Blood Pressure
        6. General Health
    """)

st.write("### Answer the following 6 Questions:")

# create the colums to hold user inputs
col1, col2, col3,col4,col5,col6 = st.columns(6)

# gather user inputs

# 1. Weight
weight = col1.text_input(
    '1. Enter your Weight (lbs)',170)

# 2. Height
height = col2.text_input(
    '2. Enter your Height (inches): ',68)

# 3. Age
age = col3.selectbox(
    '3. Select your Age:', ('Age 18 to 24',
                            'Age 25 to 29',
                            'Age 30 to 34',
                            'Age 35 to 39',
                            'Age 40 to 44',
                            'Age 45 to 49',
                            'Age 50 to 54',
                            'Age 55 to 59',
                            'Age 60 to 64',
                            'Age 65 to 69',
                            'Age 70 to 74',
                            'Age 75 to 79',
                            'Age 80 or older'), index=4)

# 4. HighChol
highchol = col1.selectbox(
    "4. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your Blood Cholesterol is high?",
    ('Yes', 'No'), index=1)

# 5. HighBP
highbp = col2.selectbox(
    "5. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that you have high Blood Pressure?",
    ('Yes', 'No'), index=0)

# 6. GenHlth
genhlth = col3.selectbox("6. General Health: How would you rank your General Health on a scale from 1 = Excellent to 5 = Poor? Consider physical and mental health.",
                         ('Excellent', 'Very Good', 'Good', 'Fair', 'Poor'), index=3)

# 7. HeartDiseaseorAttack
HeartDiseaseorAttack = col4.selectbox("7. Do you have coronary heart disease?",
                         ('Yes', 'No'), index=1)

# 8. PhysActivity
PhysActivity = col5.selectbox("8. Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?",
                         (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30), index=3)


# 9. PhysActivity
DiffWalk = col6.selectbox("9. Do you have serious difficulty walking or climbing stairs?",
                         ('Yes', 'No'), index=1)


# Create dataframe:
df1 = pd.DataFrame([[weight, height, age, highchol, highbp, genhlth,HeartDiseaseorAttack,PhysActivity,DiffWalk]], columns=[
                   'Weight', 'Height', 'Age', 'HighChol', 'HighBP', 'GenHlth','HeartDiseaseorAttack','PhysActivity','DiffWalk'])


def calculate_bmi(weight, height):
    """
    Calculate BMI from weight in lbs and height in inches.
    Args:
        weight: the weight in lbs
        height: the height in inches

    Returns:
        bmi - the body mass index

    """
    bmi = round((703 * weight)/(height**2))

    return bmi


def prep_df(df):
    """Prepare user .

    Args:
        df: the dataframe containing the 6 user inputs.

    Returns:
        the dataframe with 5 outputs. BMI, Age, HighChol, HighBP, and GenHlth

    """
    # BMI
    df['Weight']=df['Weight'].astype(int)
    df['Height']=df['Height'].astype(int)
    df['BMI'] = df.apply(lambda row: calculate_bmi(
        row['Weight'], row['Height']), axis=1)

    # Drop Weight and Height
    df = df.drop(columns=['Weight', 'Height'])
    df['PhysHlth'] = df['PhysActivity']
    # Re-Order columns
    df = df[['HighBP','HighChol', 'BMI', 'GenHlth', 'DiffWalk', 'Age', 'HeartDiseaseorAttack', 'PhysHlth']]
    # Age
    df['Age'] = df['Age'].replace({'Age 18 to 24': 1, 'Age 25 to 29': 2, 'Age 30 to 34': 3, 'Age 35 to 39': 4, 'Age 40 to 44': 5, 'Age 45 to 49': 6,
                                   'Age 50 to 54': 7, 'Age 55 to 59': 8, 'Age 60 to 64': 9, 'Age 65 to 69': 10, 'Age 70 to 74': 11, 'Age 75 to 79': 12, 'Age 80 or older': 13})
    # HighChol
    df['HighChol'] = df['HighChol'].replace({'Yes': 1, 'No': 0})
    # HighBP
    df['HighBP'] = df['HighBP'].replace({'Yes': 1, 'No': 0})
    # GenHlth
    df['GenHlth'] = df['GenHlth'].replace(
        {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5})
    # HeartDiseaseorAttack
    df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].replace({'Yes': 1, 'No': 0})


    # DiffWalk
    df['DiffWalk'] = df['DiffWalk'].replace({'Yes': 1, 'No': 0})

    return df


# prepare the user inputs for the model to accept
df = prep_df(df1)

with st.expander("Click to see user inputs"):
    st.write("**User Inputs** ", df1)
with st.expander("Click to see what goes into the Model for prediction"):
    st.write("**User Inputs Prepared ** ", df,
             "** Note that BMI is calculated from the Weight and Height you entered. Age has 14 categories from 1 to 13 in steps of 5 years. HighChol and HighBP are 0 for No and 1 for Yes. GenHlth is on a scale from 1=Excellent to 5=Poor. These come directly from BRFSS questions the model learned from.")

# Get the directory of the current script or module
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Construct the full path to the model file
model_path = os.path.join(script_dir, 'dt_model.pkl')

# Load the model
model = joblib.load(model_path)


# Make the prediction:
if st.button('Click here to predict your Type II Diabetes Risk'):

    # make the predictions
    df_scaled=model['scaler'].transform(df)
    prediction = model['model'].predict(df_scaled)
    prediction_probability = model['model'].predict_proba(df)
    low_risk_proba = round(prediction_probability[0][0] * 100)
    high_risk_proba = round(prediction_probability[0][1] * 100)


    if(prediction[0] == 0):
        st.write("You are at **low-risk** for Type II Diabetes or prediabetes")
        st.write("Predicted probality of low-risk",
                 low_risk_proba, "%")
        st.write("Predicted probality of high-risk",
                 high_risk_proba, "%")
    else:
        st.write("You are at **high-risk** for Type II Diabetes or prediabetes")
        st.write("Predicted probality of low-risk",
                 low_risk_proba, "%")
        st.write("Predicted probality of high-risk",
                 high_risk_proba, "%")

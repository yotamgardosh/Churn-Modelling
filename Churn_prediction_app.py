import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('best_xgb_model.pkl')  # Save your model to this file

# Define the feature names
feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Create input fields for user data
st.title('Churn Prediction App')

# Add introductory explanation
st.markdown("""
    ### Welcome to the Churn Prediction App!
    This app helps predict whether a customer will leave the business (churn) or stay with the business.
    - **Input Parameters:** Provide customer information such as credit score, age, balance, etc.
    - **Prediction:** The app will predict whether the customer is likely to leave or stay.
    - **Model Details:** The model used is an XGBoost classifier, which is a powerful machine learning algorithm known for its performance and efficiency.
    - **Visualizations:** Explore visualizations showing the relationship between various features and customer churn.

    **Note:** Churn means the customer has left the business.
            
    ## In this section you can input a customer's parameters and get the models prediction:
""")

def user_input_features():
    CreditScore = st.number_input('Credit Score', min_value=0, max_value=1000, value=500)
    Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=0, max_value=100, value=30)
    Tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
    Balance = st.number_input('Balance', min_value=0.0, max_value=1e7, value=50000.0)
    NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox('Has Credit Card', [0, 1])
    IsActiveMember = st.selectbox('Is Active Member', [0, 1])
    EstimatedSalary = st.number_input('Estimated Salary, in Euro', min_value=0.0, max_value=1e7, value=100000.0)

    data = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode the categorical features
input_df['Geography'] = input_df['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})

# Scale the features
scaler = joblib.load('scaler.pkl')  # Save your scaler to this file
input_df_scaled = scaler.transform(input_df)

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Predict using the trained model
prediction = model.predict(input_df_scaled)
prediction_proba = model.predict_proba(input_df_scaled)

# Display the prediction
st.subheader('Prediction')
st.info('Left the Business' if prediction[0] == 1 else 'Stayed with the Business')

st.subheader('Prediction Probability')
st.info(f'Probability of Leaving: {prediction_proba[0][1]:.2f}')


# Add image
st.image("https://www.cleartouch.in/wp-content/uploads/2022/11/Customer-Churn.png")

# Display explanatory text box
st.markdown("""
    ### These charts visualize important features from the dataset:
""")

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Filter DataFrame to include only selected features
selected_features = ['Age', 'NumOfProducts', 'Balance', 'IsActiveMember', 'EstimatedSalary', 'Exited']
df_selected = df[selected_features]


# Display correlation bar chart for selected features
st.subheader('Correlation Bar Chart for most importent Features')
correlation_data = df_selected.corr().loc['Exited', :].drop('Exited')  # Get correlation values with target
st.bar_chart(correlation_data)

churn_labels = {0: 'Stayed with the Business', 1: 'Left the Business'}

# Plot age distribution for churned and non-churned customers using a line chart
st.subheader('Age Distribution by Churn')
age_churn_data = df.groupby(['Age', 'Exited']).size().unstack().fillna(0)
age_churn_data.columns = [churn_labels[col] for col in age_churn_data.columns]  # Change column labels
st.line_chart(age_churn_data)


# Plot number of products distribution for churned and non-churned customers using an area chart
st.subheader('Number of Products Distribution by Churn')
num_products_churn_data = df.groupby(['NumOfProducts', 'Exited']).size().unstack().fillna(0)
num_products_churn_data.columns = [churn_labels[col] for col in num_products_churn_data.columns]  # Change column labels
st.area_chart(num_products_churn_data)

# Display explanatory text box
st.markdown("""
    #### Explanation:
    - The Age Distribution by Churn chart shows the count of customers at different ages, categorized by whether they stayed with the company or left the company. We can see that the older the cutomer the more likely they are to leave.
    - The Number of Products Distribution by Churn chart shows the count of customers based on the number of products they have, categorized by whether they stayed with the company or left the company. We can see that customers with less products tend to leave the company in higher rates then one's with more products
""")
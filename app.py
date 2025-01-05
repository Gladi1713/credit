import streamlit as st
from textblob import TextBlob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset (ensure the CSV is in the same directory as this script)
data = pd.read_csv('synthetic_credit_data.csv')

# Standardize the numerical data
scaler = StandardScaler()
data[['age', 'annual_income', 'credit_card_debt', 'utility_payments', 'sentiment_score']] = scaler.fit_transform(
    data[['age', 'annual_income', 'credit_card_debt', 'utility_payments', 'sentiment_score']])

# Train a logistic regression model
X = data[['age', 'annual_income', 'credit_card_debt', 'utility_payments', 'sentiment_score']]
y = data['default_risk']
model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title("Real-Time Credit Risk Assessment")

st.sidebar.header("Enter User Details")
age = st.sidebar.slider("Age", 18, 65, 30)
income = st.sidebar.number_input("Annual Income ($)", 20000, 100000, 50000)
debt = st.sidebar.number_input("Credit Card Debt ($)", 1000, 20000, 5000)
utility = st.sidebar.slider("Utility Payments (per year)", 1, 12, 6)
social_media_text = st.sidebar.text_area("Social Media Text", "I love this service!")

# Sentiment analysis
sentiment_score = TextBlob(social_media_text).sentiment.polarity

# Predict credit risk
user_data = scaler.transform([[age, income, debt, utility, sentiment_score]])
risk_prediction = model.predict(user_data)[0]
risk_proba = model.predict_proba(user_data)[0][1]

# Display results
st.write("### Predicted Credit Risk:")
st.write("High Risk" if risk_prediction == 1 else "Low Risk")
st.write(f"Risk Probability: {risk_proba:.2f}")

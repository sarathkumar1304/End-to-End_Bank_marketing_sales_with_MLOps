import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Encode categorical values as numbers
def encode_inputs(inputs):
    encoding_dict = {
        'job': {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10},
        'marital': {"married": 1, "single": 0, "divorced": 2},
        'education': {"secondary": 1, "tertiary": 2, "primary": 3},
        'default': {"no": 0, "yes": 1},
        'housing': {"yes": 1, "no": 0},
        'loan': {"no": 0, "yes": 1},
        'contact': {"cellular": 0, "telephone": 1},
        'month': {
            "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, 
            "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
        },
        'day_of_week': {
            "Monday": 1, "Tuesday": 2, "Wednesday": 3, 
            "Thursday": 4, "Friday": 5
        }
    }

    for key, value in inputs.items():
        if key in encoding_dict:
            inputs[key] = encoding_dict[key][inputs[key]]
    return inputs

# Define the prediction function
def predict_output(model, input_data):
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit app
def main():

# App Name and Header
    st.title("Bank Term Deposit Subscription Predictor")
    st.subheader("An Intelligent Solution for Direct Marketing Campaigns")

    # Description
    st.markdown("""
    ### About This Application
    This app is designed to help banking institutions predict whether a client will subscribe to a term deposit based on various attributes from direct marketing campaigns. The data comes from Portuguese banking campaigns conducted between May 2008 and November 2010.

    ### How It Works
    Using machine learning models, this app analyzes client and campaign-related features to predict the likelihood of subscription to term deposits. This information can help banks:
    - Optimize marketing strategies.
    - Identify high-probability clients.
    - Enhance resource allocation for better efficiency.

    ### Key Features:
    - **Client Information**: Age, job, marital status, education level, housing loan, etc.
    - **Campaign Details**: Contact type, duration, last contact day and month.
    - **Performance Metrics**: Campaign success rate, previous outcomes, and more.

    ### Why Use This App?
    - Gain actionable insights from historical campaign data.
    - Improve customer targeting for term deposit offers.
    - Save time and resources by focusing on potential subscribers.

    ### Get Started:
    Enter client and campaign details in the form below to predict the likelihood of term deposit subscription.
    """)

    st.write("### Enter the Details Below to Predict the Outcome:")


    # Input fields for user input
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox(
        "Job",
        ["blue-collar", "management", "technician", "admin.", "services",
         "retired", "self-employed", "entrepreneur", "unemployed", 
         "housemaid", "student"]
    )
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education Level", ["secondary", "tertiary", "primary"])
    default = st.selectbox("Has Credit Default?", ["no", "yes"])
    balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=500)
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    month = st.selectbox(
        "Month",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    duration = st.number_input("Duration (in seconds)", min_value=0, max_value=5000, value=120)
    campaign = st.number_input("Campaign (Number of contacts)", min_value=1, max_value=50, value=1)
    pdays = st.number_input("Pdays (Number of days since last contact)", min_value=-1, max_value=1000, value=999)
    previous = st.number_input("Previous (Number of previous contacts)", min_value=0, max_value=50, value=0)

    # Model loading
    model_path = "Artifact/model.pkl"
    model = load_model(model_path)

    # Input data preparation
    if st.button("Predict"):
        if model:
            inputs = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'balance': balance,
                'housing': housing,
                'loan': loan,
                'contact': contact,
                'day_of_week': day_of_week,
                'month': month,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous
            }
            
            # Encode categorical inputs
            encoded_inputs = encode_inputs(inputs)

            # Convert to DataFrame
            input_data = pd.DataFrame([encoded_inputs])

            # Make prediction
            predictions = predict_output(model, input_data)
            if predictions is not None:
                if predictions[0] == 0:
                    st.success("Predicted Output: Not Subscribe")
                elif predictions[0] == 1:
                    st.success("Predicted Output: Will Subscribe")


if __name__ == "__main__":
    main()

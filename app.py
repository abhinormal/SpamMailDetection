import streamlit as st
import pickle

# Load your trained model
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Title of the app
st.title('Spam Mail Predictor')

# Input email text
email_text = st.text_area("Enter the email text:")

# Button to make prediction
if st.button('Predict'):
    # Transform the input using the loaded vectorizer
    input_vector = vectorizer.transform([email_text])
    
    # Predict using the loaded model
    prediction = model.predict(input_vector)
    
    # Display result
    if prediction[0] == 0:
        st.write("The email is predicted to be **SPAM**.")
    else:
        st.write("The email is predicted to be **NOT SPAM**.")

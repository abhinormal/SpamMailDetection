import streamlit as st
import pickle
from PIL import Image

# Load your trained model
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Set up the page configuration
st.set_page_config(
    page_title="Spam Mail Predictor",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and description
st.title('📧 Spam Mail Predictor')
st.write("""
Welcome to the Spam Mail Predictor! This app uses a machine learning model to classify emails as **SPAM** or **NOT SPAM**. Enter the text of an email below to get started.
""")

# Load and display an image
image = Image.open('email_image.jpg')
st.image(image, caption='Spam or Not Spam?', use_column_width=True)

# Input email text
st.header("Enter the email text:")
email_text = st.text_area("", height=200)

# Button to make prediction
if st.button('Predict'):
    if email_text.strip() == "":
        st.warning("Please enter some email text to predict.")
    else:
        # Transform the input using the loaded vectorizer
        input_vector = vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(input_vector)
        
        # Display result
        st.subheader("Prediction:")
        if prediction[0] == 0:
            st.error("The email is predicted to be **SPAM**.")
        else:
            st.success("The email is predicted to be **NOT SPAM**.")

# Sidebar with additional information
st.sidebar.header("About")
st.sidebar.write("""
This app uses a logistic regression model trained on email data to predict whether an email is spam or not. The model was trained using the scikit-learn library and the TfidfVectorizer for text feature extraction.
""")

st.sidebar.header("How to use")
st.sidebar.write("""
1. Enter the email text in the provided text area.
2. Click the "Predict" button to see the result.
3. The result will be displayed as either **SPAM** or **NOT SPAM**.
""")

# Footer
st.markdown("""
<style>
footer {visibility: hidden;}
footer:after {
    visibility: visible;
    display: block;
    position: relative;
    padding: 5px;
    top: 2px;
    color: gray;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

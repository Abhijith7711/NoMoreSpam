import streamlit as st
import joblib

# Load the saved model and Vectorizer
model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title of the app
st.title('Spam Detector')

# Add text input box for the user to enter a message
user_input = st.text_area("Enter the message to classify:")

# Add a "Predict" button
if st.button("Detect"):
    if user_input.strip():  # Ensure the text box is not empty
        # Convert the user's input into the same TF-IDF format as the training data
        user_input_tfidf = vectorizer.transform([user_input])

        # Predict using the loaded model
        prediction = model.predict(user_input_tfidf)

        # Show the result
        if prediction == 1:
            st.success("The message is **Spam**.")
        else:
            st.success("The message is **Not Spam**.")
    else:
        st.warning("Please enter a message to classify.")

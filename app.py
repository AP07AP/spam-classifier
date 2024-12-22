import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize PorterStemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # Pre-load stopwords

# Preprocessing function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Stem the words
    text = [ps.stem(word) for word in text]
    
    # Join back into a single string
    return " ".join(text)

# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit App Title
st.title("Email/SMS Spam Classifier")

# Input Field
input_sms = st.text_area("Enter the message")

# Predict Button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        try:
            # Preprocess the input
            transformed_sms = transform_text(input_sms)
            # Vectorize the input
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]
            # Display Result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"Error during processing: {e}")

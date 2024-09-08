from flask import Flask, render_template, request, redirect, url_for
import joblib
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tf_vec = joblib.load('vectorizer.pkl')

app = Flask(__name__)

# Download NLTK data once at the start
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.data.path.append('/path/to/nltk_data')
nltk.download('punkt', download_dir='/path/to/nltk_data')
# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('spam_classifier_model.html')

# Prediction route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        # Preprocess and vectorize the message
        cleaned_message = clean_text(message)
        vectorized_message = tf_vec.transform([cleaned_message]).toarray()

        # Predict whether it's spam or not and calculate confidence
        prediction_proba = model.predict_proba(vectorized_message)[0]
        prediction = model.predict(vectorized_message)[0]
        confidence = np.max(prediction_proba) * 100  # Convert to percentage

        # Return result to the user
        result = "SPAM" if prediction == 'spam' else "NOT SPAM"
        return render_template('spam_classifier_model.html', result=f"The message is classified as {result}.", confidence=round(confidence, 2))
    else:
        return redirect(url_for('home'))

# Preprocess function (same as in the training script)
def clean_text(text):
    ps = PorterStemmer()

    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters and apply stemming
    cleaned_words = [ps.stem(word) for word in text if word.isalnum()]

    # Return the cleaned and stemmed text
    return " ".join(cleaned_words)

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 5000)

import joblib
import re

nb_classifier = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = re.sub(r'\W', ' ', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    text = text.lower() # Convert to lowercase
    return text

user_input = input("Enter tweet: ")

# Clean and vectorize new reviews
new_text_cleaned = clean_text(user_input)
new_X = vectorizer.transform([new_text_cleaned])

# Predict sentiment
predictions = nb_classifier.predict(new_X)
status_mapping = {2: "positive", 1: "neutral", 0: "negative"}
recommendation_status = [status_mapping[pred] for pred in predictions]
print("Predicted class labels:", predictions)
print("Recommendation statuses:", recommendation_status)
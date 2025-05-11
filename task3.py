import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Download necessary NLTK data files
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []
classes = []
responses = {}

# Loop through each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    
    # Add intent response options to the dictionary
    responses[intent['tag']] = intent['responses']
    
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_sentences, training_labels, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(tokenizer=nltk.word_tokenize), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Function to predict the intent of a sentence
def predict_intent(text):
    intent = model.predict([text])[0]
    return intent

# Function to get a response based on the predicted intent
def get_response(user_input):
    predicted_intent = predict_intent(user_input)
    response = random.choice(responses[predicted_intent])
    return response

# Chat loop
print("ChatBot: Hello! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        print("ChatBot: Goodbye!")
        break
    
    response = get_response(user_input)
    print("ChatBot:", response)

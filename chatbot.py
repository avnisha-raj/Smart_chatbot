

# Step 1: Install Required Libraries
!pip install nltk scikit-learn pandas numpy gradio tensorflow

# Import all necessary libraries
import nltk
import pandas as pd
import numpy as np
import re
import json
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print(" All libraries installed successfully!")
print(" NLTK data downloaded!")

# Step 2: Create Training Data
training_data = {
    'greeting': [
        'hello', 'hi', 'hey', 'good morning', 'good evening', 'namaste',
        'how are you', 'what\'s up', 'greetings', 'good afternoon'
    ],
    'goodbye': [
        'bye', 'goodbye', 'see you later', 'take care', 'farewell',
        'see you soon', 'catch you later', 'until next time', 'bye bye'
    ],
    'thanks': [
        'thank you', 'thanks', 'thank you so much', 'appreciate it',
        'thanks a lot', 'grateful', 'much appreciated', 'thanks for help'
    ],
    'about': [
        'what can you do', 'who are you', 'tell me about yourself',
        'what are your capabilities', 'how can you help', 'about you'
    ],
    'technical_help': [
        'python help', 'coding problem', 'programming issue', 'debug code',
        'technical support', 'development help', 'coding assistance'
    ],
    'weather': [
        'weather today', 'how is weather', 'temperature', 'is it raining',
        'weather forecast', 'climate', 'sunny today', 'weather update'
    ],
    'time': [
        'what time is it', 'current time', 'time now', 'what\'s the time',
        'tell me time', 'clock', 'time please'
    ]
}

# Create responses for each intent
responses = {
    'greeting': [
        'Hello! How can I help you today?',
        'Hi there! What can I do for you?',
        'Hey! Nice to see you. How may I assist?',
        'Greetings! I\'m here to help.'
    ],
    'goodbye': [
        'Goodbye! Have a great day!',
        'See you later! Take care!',
        'Bye! Feel free to come back anytime.',
        'Until next time! Stay safe!'
    ],
    'thanks': [
        'You\'re welcome! Happy to help!',
        'My pleasure! Anything else?',
        'Glad I could help!',
        'You\'re most welcome!'
    ],
    'about': [
        'I\'m an AI chatbot built with machine learning! I can help with various queries.',
        'I\'m here to assist you with questions and provide helpful information.',
        'I\'m an intelligent assistant powered by ML algorithms.'
    ],
    'technical_help': [
        'I can help with programming questions! What specific issue are you facing?',
        'Technical support is my specialty. Please describe your problem.',
        'I\'m here to help with coding and technical issues!'
    ],
    'weather': [
        'I don\'t have real-time weather data, but you can check weather apps for current conditions.',
        'For accurate weather information, I recommend checking a weather service.',
        'I can\'t access live weather data, but local weather apps will help!'
    ],
    'time': [
        'I don\'t have access to real-time clock, please check your device time.',
        'For current time, please check your system clock.',
        'I can\'t tell exact time, but your device shows the current time.'
    ]
}

# Convert to DataFrame for training
data_list = []
for intent, phrases in training_data.items():
    for phrase in phrases:
        data_list.append({'text': phrase, 'intent': intent})

df = pd.DataFrame(data_list)
print(f"Training data created with {len(df)} samples")
print(f" Intents: {df['intent'].value_counts().to_dict()}")
print("\nSample data:")
print(df.head(10))

# CELL 3: Text Preprocessing Functions (FIXED)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

print(" Setting up text preprocessing...")

# Download missing NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print(" Additional NLTK data downloaded!")
except:
    print("ℹ NLTK data already available")

# Try different tokenizers based on availability
try:
    from nltk.tokenize import word_tokenize
    tokenize_func = word_tokenize
except:
    # Fallback tokenizer
    tokenize_func = lambda x: x.split()

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords
            self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'])

        self.lemmatizer = WordNetLemmatizer()
        print(" Text preprocessor initialized!")

    def clean_text(self, text):
        """Clean and preprocess text"""
        try:
            # Convert to lowercase
            text = str(text).lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize (with fallback)
            try:
                tokens = tokenize_func(text)
            except:
                tokens = text.split()

            # Remove stopwords and lemmatize
            cleaned_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        cleaned_token = self.lemmatizer.lemmatize(token)
                        cleaned_tokens.append(cleaned_token)
                    except:
                        cleaned_tokens.append(token)

            return ' '.join(cleaned_tokens)
        except Exception as e:
            print(f"Warning: Error processing text '{text}': {e}")
            return str(text).lower()

    def preprocess_dataset(self, df):
        """Preprocess entire dataset"""
        print(" Cleaning all text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        return df

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Clean the training data
print(" Processing training data...")
df_cleaned = preprocessor.preprocess_dataset(df.copy())

print(" Text preprocessing completed!")
print("\n Before vs After cleaning (Sample):")
print("=" * 60)
for i in range(5):
    print(f"Original: '{df.iloc[i]['text']}'")
    print(f"Cleaned:  '{df_cleaned.iloc[i]['cleaned_text']}'")
    print(f"Intent:   {df.iloc[i]['intent']}")
    print("-" * 40)

print(f"\n Dataset stats:")
print(f"Total samples: {len(df_cleaned)}")
print(f"Unique intents: {df_cleaned['intent'].nunique()}")
print(" Ready for model training!")

# Verify data quality
empty_cleaned = df_cleaned[df_cleaned['cleaned_text'].str.strip() == ''].shape[0]
if empty_cleaned > 0:
    print(f" Warning: {empty_cleaned} samples became empty after cleaning")
else:
    print(" All samples successfully cleaned!")
    
# Step 4: Model Training
class ChatbotModel:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        self.responses = responses
        self.is_trained = False

    def train(self, df):
        """Train the chatbot model"""
        # Prepare training data
        X = df['cleaned_text'].values
        y = df['intent'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f" Model trained successfully!")
        print(f" Accuracy: {accuracy:.2%}")
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))

        self.is_trained = True
        return accuracy

    def predict_intent(self, text):
        """Predict intent for given text"""
        if not self.is_trained:
            return "unknown", 0.0

        # Preprocess input
        cleaned_text = self.preprocessor.clean_text(text)

        # Predict
        intent = self.pipeline.predict([cleaned_text])[0]
        confidence = self.pipeline.predict_proba([cleaned_text]).max()

        return intent, confidence

    def get_response(self, text):
        """Get chatbot response"""
        intent, confidence = self.predict_intent(text)

        # If confidence is too low, return default response
        if confidence < 0.3:
            return "I'm not sure I understand. Could you please rephrase that?"

        # Get random response for the predicted intent
        if intent in self.responses:
            response = np.random.choice(self.responses[intent])
            return f"{response} (Confidence: {confidence:.1%})"
        else:
            return "I'm still learning. Could you ask something else?"

# Create and train the model
chatbot = ChatbotModel()
accuracy = chatbot.train(df_cleaned)

# Test the model
test_phrases = [
    "Hello there!",
    "Thanks for your help",
    "What can you do?",
    "Bye bye",
    "How's the weather?"
]

print("\n Testing the model:")
print("=" * 50)
for phrase in test_phrases:
    response = chatbot.get_response(phrase)
    print(f"User: {phrase}")
    print(f"Bot: {response}")
    print("-" * 30)
    
def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
    try:
        response = requests.get(url)
        data = response.json()

        if "result" in data:
            converted_amount = round(data["result"], 2)
            return f"{amount} {from_currency.upper()} = {converted_amount} {to_currency.upper()}"
        else:
            return "Sorry, couldn't get the conversion rate."
    except Exception as e:
        return f" Error during currency conversion: {e}"

import requests

API_KEY = "1cb7f6b6e86a9bc433b62ae456eac6cd"  # ← paste your real key here

def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.exchangerate.host/convert?access_key={API_KEY}&from={from_currency.upper()}&to={to_currency.upper()}&amount={amount}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(" Raw API response:", result)

            if 'result' in result and result['result'] is not None:
                converted = result['result']
                return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
            else:
                return " Could not retrieve conversion — maybe invalid currency code?"
        else:
            return " API error!"
    except Exception as e:
        return f"Exception: {e}"

# Test
print(convert_currency(100, "usd", "inr"))

import gradio as gr
from datetime import datetime
import random
import re
import numpy as np

class ChatbotInterface:
    def __init__(self, model):
        self.model = model
        self.chat_history = []

    def improved_get_response(self, user_input):
        """Improved response function with pattern matching and currency support"""
        try:
            processed_input = user_input.lower().strip()

            # 🪙 Currency conversion pattern
            currency_pattern = r'convert (\d+(?:\.\d+)?) (\w{3}) to (\w{3})'
            match = re.search(currency_pattern, processed_input)
            if match:
                amount = float(match.group(1))
                from_curr = match.group(2).upper()
                to_curr = match.group(3).upper()
                return convert_currency(amount, from_curr, to_curr)

            if not processed_input:
                return "Please say something! I'm here to help."

            # Pattern-based responses
            greeting_patterns = ['hi', 'hello', 'hey', 'hlo', 'helo', 'good morning', 'good evening', 'namaste']
            goodbye_patterns = ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'exit', 'quit']
            thanks_patterns = ['thank', 'thanks', 'thx', 'appreciate', 'grateful']
            how_are_you_patterns = ['how are you', 'how do you do', 'whats up', 'how you doing']
            help_patterns = ['help', 'can you help', 'what can you do', 'capabilities']
            name_patterns = ['what is your name', 'who are you', 'what are you']

            if any(p in processed_input for p in greeting_patterns):
                return random.choice([
                    "Hello! How can I help you today? ",
                    "Hi there! What can I do for you?",
                    "Hey! Nice to meet you!",
                    "Hello! I'm here to assist you!",
                    "Hi! What would you like to know?"
                ])
            elif any(p in processed_input for p in goodbye_patterns):
                return random.choice([
                    "Goodbye! Have a great day! ",
                    "See you later! Take care!",
                    "Bye! It was nice talking to you!",
                    "Farewell! Come back anytime!",
                    "Take care! Have a wonderful day!"
                ])
            elif any(p in processed_input for p in thanks_patterns):
                return random.choice([
                    "You're welcome! ",
                    "No problem at all!",
                    "Glad I could assist you!",
                    "My pleasure!",
                    "You're welcome! Feel free to ask more!"
                ])
            elif any(p in processed_input for p in how_are_you_patterns):
                return random.choice([
                    "I'm doing great! How about you?",
                    "All good! What can I help you with?",
                    "Fine, thank you! How can I assist?",
                    "Wonderful! What brings you here today?"
                ])
            elif any(p in processed_input for p in help_patterns):
                return random.choice([
                    "I can help with various questions! Ask me anything!",
                    "I'm here to assist with your queries!",
                    "Try asking about greetings, capabilities, or currency conversion!",
                    "I love helping! Let's chat!"
                ])
            elif any(p in processed_input for p in name_patterns):
                return random.choice([
                    "I'm ChatBot, your virtual assistant! ",
                    "You can call me AI ChatBot!",
                    "I'm an AI here to help you!",
                    "I'm ChatBot, ready to assist!"
                ])
                            # Check for calculator queries
            elif any(op in processed_input for op in ['+', '-', '*', '/', 'calculate', 'what is', '^', '%']):
                try:
                    # Remove all characters except numbers, operators, and brackets
                    expression = re.sub(r'[^0-9\.\+\-\*\/\(\)\%\^ ]', '', processed_input)
                    expression = expression.replace('^', '**')  # Power operator

                    result = eval(expression)
                    return f"🧮 Result: {result}"
                except:
                    return "Sorry, I couldn't calculate that. Make sure it's a valid math expression."

            else:
                original_response = self.model.get_response(user_input)
                if "not sure" in original_response.lower() or "rephrase" in original_response.lower():
                    return random.choice([
                        "Could you rephrase that?",
                        "Hmm... I didn't catch that. Try again?",
                        "I'm still learning. Want to ask in a different way?",
                        "Interesting! Can you explain more?"
                    ])
                else:
                    return original_response

        except Exception as e:
            print(f"Error: {e}")
            return "I'm having technical difficulties. Try again later."

    def chat_function(self, message, history):
        if not message.strip():
            return history, ""
        bot_response = self.improved_get_response(message)
        history.append([message, bot_response])
        self.chat_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user': message,
            'bot': bot_response
        })
        return history, ""

    def clear_chat(self):
        self.chat_history = []
        return []

    def get_stats(self):
        total_messages = len(self.chat_history)
        if total_messages == 0:
            return "No conversations yet!"
        all_user_messages = ' '.join([chat['user'].lower() for chat in self.chat_history])
        common_words = {}
        for word in all_user_messages.split():
            if len(word) > 2:
                common_words[word] = common_words.get(word, 0) + 1
        top_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:3]
        stats = f"📊 Stats:\nTotal messages: {total_messages}\n"
        stats += f"Avg message length: {sum(len(chat['user']) for chat in self.chat_history) // total_messages}\n"
        if top_words:
            stats += f"Top words: {', '.join([f'{w}({c})' for w, c in top_words])}\n"
        stats += f"Last message: {self.chat_history[-1]['timestamp']}"
        return stats

# Create interface
interface = ChatbotInterface(chatbot)

with gr.Blocks(title="💬 ChatBot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Smart ChatBot with Currency Converter
    Say things like:
    - "hlo", "how are you"
    - "convert 100 usd to inr"
    - "thank you", "bye"
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                height=400,
                label="AI Chat Assistant",
                show_label=True,
                placeholder="Start chatting here..."
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message...",
                    label="Your Message",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary")

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                stats_btn = gr.Button("Stats", variant="secondary")

        with gr.Column(scale=1):
            stats_box = gr.Textbox(
                label="Statistics",
                lines=6,
                interactive=False,
                placeholder="Click 'Stats' after chatting!"
            )

    msg.submit(interface.chat_function, [msg, chatbot_ui], [chatbot_ui, msg])
    send_btn.click(interface.chat_function, [msg, chatbot_ui], [chatbot_ui, msg])
    clear_btn.click(interface.clear_chat, outputs=chatbot_ui)
    stats_btn.click(interface.get_stats, outputs=stats_box)

# Launch interface
demo.launch(share=True, debug=True)


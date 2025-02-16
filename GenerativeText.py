import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np

# Load GPT-2 Model
def generate_text_gpt2(prompt, max_length=100):
    generator = pipeline("text-generation", model="gpt2")
    response = generator(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]['generated_text']

# Prepare LSTM Model for Text Generation
def build_lstm_model(vocab_size, embedding_dim=64, lstm_units=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Generate text using LSTM model
def generate_text_lstm(model, tokenizer, seed_text, max_length=50):
    for _ in range(max_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example Usage
if __name__ == "__main__":
    prompt = "The future of AI in cybersecurity is"
    print("GPT-2 Generated Text:")
    print(generate_text_gpt2(prompt))

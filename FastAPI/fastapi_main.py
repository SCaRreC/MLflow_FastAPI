from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import time

app = FastAPI()

class Input(BaseModel):
    prompt: str

# Hugging Face Pipelines 
sentiment_analyzer = pipeline("sentiment-analysis")
text_generator = pipeline("text-generation", model="gpt2")

# 1. basic endpoint
@app.get("/")
def home():
    return {"message": "Welcome to my API with FastAPI and Hugging Face"}

# 2. Sentiment analysis (pipeline HF)
@app.get("/sentiment")
def analize_sentiment(prompt: str = "I find this exercise quite cool!"):
    result = sentiment_analyzer(prompt)
    return {"prompt": prompt, "result": result}

# 3. Text generator (pipeline HF)
@app.get("/generate")
def generate_text(prompt: str = "Once Upon a time"):
    result = text_generator(prompt, max_length=30, num_return_sequences=1)
    return {"prompt": prompt, "result": result}

# 4. Words counter
@app.get("/count")
def count_words(prompt: str = "You can’t judge a book by its cover"):
    n_words = len(prompt.split())
    return {"prompt": prompt, "n_words": n_words}

# 5. text inverter 
@app.get("/invert")
def invert_text(text: str = "Hello word"):
    return {"original": text, "inverted_text": text[::-1]}

# Post
#@app.post("/input_text")
#def incoming_text(text:str):
#    time.sleep(200)
#    return 

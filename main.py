from fastapi import FastAPI
from pydantic import BaseModel
import cohere
import groq
import os
import requests
from googlesearch import search

app = FastAPI()

# API Keys
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

co = cohere.Client(COHERE_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str

# Brain - Category detect karta hai
def classify(text):
    response = co.classify(
        inputs=[text],
        examples=[
            cohere.ClassifyExample(text="Hello kaise ho", label="general"),
            cohere.ClassifyExample(text="Tumhara naam kya hai", label="general"),
            cohere.ClassifyExample(text="Ek joke sunao", label="general"),
            cohere.ClassifyExample(text="Mujhe neend aa rahi hai", label="general"),
            cohere.ClassifyExample(text="Aaj ka news kya hai", label="news"),
            cohere.ClassifyExample(text="Latest news batao", label="news"),
            cohere.ClassifyExample(text="India mein kya ho raha hai", label="news"),
            cohere.ClassifyExample(text="Aaj ka mausam kaisa hai", label="weather"),
            cohere.ClassifyExample(text="Kal barish hogi kya", label="weather"),
            cohere.ClassifyExample(text="Temperature kya hai", label="weather"),
            cohere.ClassifyExample(text="Tata Motors stock price", label="realtime"),
            cohere.ClassifyExample(text="IPL ka score kya hai", label="realtime"),
            cohere.ClassifyExample(text="Aaj ka dollar rate", label="realtime"),
            cohere.ClassifyExample(text="Ek sunset ki image banao", label="image"),
            cohere.ClassifyExample(text="Billi ki photo banao", label="image"),
            cohere.ClassifyExample(text="Mountains ka wallpaper chahiye", label="image"),
            cohere.ClassifyExample(text="YouTube kholo", label="os_task"),
            cohere.ClassifyExample(text="Calculator band karo", label="os_task"),
            cohere.ClassifyExample(text="WhatsApp open karo", label="os_task"),
        ]
    )
    return response.classifications[0].prediction

# General Chat
def general_chat(text):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Reply in the same language the user uses."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# Realtime Search
def realtime_search(text):
    results = []
    for result in search(text, num_results=3):
        results.append(result)
    
    context = "\n".join(results)
    
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer based on search results. Be concise."},
            {"role": "user", "content": f"Query: {text}\nSearch Results: {context}"}
        ]
    )
    return response.choices[0].message.content

# Weather
def get_weather(text):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a weather assistant. Ask user for their city if not mentioned."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# News
def get_news(text):
    results = []
    for result in search("latest news today " + text, num_results=3):
        results.append(result)
    
    context = "\n".join(results)
    
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Summarize latest news. Be concise."},
            {"role": "user", "content": f"News Query: {text}\nResults: {context}"}
        ]
    )
    return response.choices[0].message.content

# Main Route
@app.post("/jarvis")
async def jarvis(user_input: UserInput):
    text = user_input.message
    category = classify(text)
    
    if category == "general":
        answer = general_chat(text)
    elif category == "realtime":
        answer = realtime_search(text)
    elif category == "weather":
        answer = get_weather(text)
    elif category == "news":
        answer = get_news(text)
    elif category == "image":
        answer = "Image generation coming soon!"
    elif category == "os_task":
        answer = f"OS_TASK:{text}"
    else:
        answer = general_chat(text)
    
    return {
        "category": category,
        "response": answer
    }

@app.get("/")
async def root():
    return {"status": "Jarvis Backend Running!"}
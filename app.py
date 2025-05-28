import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Set page config at the very top
st.set_page_config(page_title="Complete Social Media Analyser", layout="wide")

# Load the BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Fetch comments from a Facebook post
def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {'access_token': token, 'summary': 'true', 'limit': 100}
    res = requests.get(url, params=params).json()
    return [c["message"] for c in res.get("data", []) if "message" in c]

# Perform sentiment analysis on each comment
def classify_sentiment(comment):
    inputs = tokenizer.encode_plus(comment, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    score = torch.argmax(probs).item()
    if score <= 1:
        return "Negative"
    elif score == 2:
        return "Neutral"
    else:
        return "Positive"

# Get basic page information
def fetch_page_info(token, page_id):
    url = f"https://graph.facebook.com/v18.0/{page_id}"
    params = {'fields': 'name,fan_count', 'access_token': token}
    return requests.get(url, params=params).json()

# Get likes and shares for a specific post
def fetch_post_metrics(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}"
    params = {
        'fields': 'reactions.summary(true),shares',
        'access_token': token
    }
    res = requests.get(url, params=params).json()
    likes = res.get('reactions', {}).get('summary', {}).get('total_count', 0)
    shares = res.get('shares', {}).get('count', 0)
    return likes, shares

# UI starts here
st.title("📊 Complete Social Media Analyser")

token = st.text_input("🔐 Facebook Graph API Access Token", type="password")
page_id = st.text_input("📄 Facebook Page ID")
post_id = st.text_input("📝 Facebook Post ID")

if token and page_id and post_id:
    with st.spinner("Fetching comments and metrics..."):
        try:
            # Get and analyze comments
            comments = fetch_comments(token, post_id)
            sentiments = [classify_sentiment(c) for c in comments]
            df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

            # Show sentiment distribution
            st.subheader("📈 Sentiment Analysis")
            st.bar_chart(df["Sentiment"].value_counts())
            st.dataframe(df)

            # Post metrics
            st.subheader("📣 Post Metrics")
            likes, shares = fetch_post_metrics(token, post_id)
            st.write(f"👍 Reactions (likes etc.): {likes}")
            st.write(f"🔄 Shares: {shares}")

            # Page Info
            st.subheader("📘 Page Info")
            info = fetch_page_info(token, page_id)
            if 'error' in info:
                st.error(f"Page Info Error: {info['error']['message']}")
            else:
                st.write(f"Page Name: {info.get('name', 'N/A')}")
                st.write(f"Followers: {info.get('fan_count', 'N/A')}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

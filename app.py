import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Page setup
st.set_page_config(page_title="Facebook Post Analyser", layout="wide")

# Load BERT model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Fetch Facebook post details
def fetch_post_details(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}"
    params = {
        'access_token': token,
        'fields': 'message,created_time,reactions.summary(true),shares,comments.summary(true),permalink_url'
    }
    response = requests.get(url, params=params)
    return response.json()

# Fetch Facebook comments
def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {
        'access_token': token,
        'summary': 'true',
        'limit': 100
    }
    response = requests.get(url, params=params)
    return [c["message"] for c in response.json().get("data", []) if "message" in c]

# Sentiment classifier
def classify_sentiment(text):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    score = torch.argmax(probs).item()
    if score <= 1:
        return "Negative"
    elif score == 2:
        return "Neutral"
    else:
        return "Positive"

# UI
st.title("📊 Facebook Post Analyser")

token = st.text_input("🔐 Facebook Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID")

if token and post_id:
    st.success("✅ Token and Post ID entered")
    try:
        st.info("Fetching post details...")
        post = fetch_post_details(token, post_id)
        if "error" in post:
            st.error(f"Facebook API Error: {post['error']['message']}")
        else:
            st.subheader("🧾 Post Information")
            st.write(f"📅 Created: {post.get('created_time', 'N/A')}")
            st.write(f"📝 Message: {post.get('message', 'No message')}")
            st.write(f"👍 Reactions: {post.get('reactions', {}).get('summary', {}).get('total_count', 0)}")
            st.write(f"🔄 Shares: {post.get('shares', {}).get('count', 0)}")
            st.write(f"💬 Comments: {post.get('comments', {}).get('summary', {}).get('total_count', 0)}")
            if post.get("permalink_url"):
                st.markdown(f"🔗 [View Post on Facebook]({post['permalink_url']})")

            st.info("Fetching and analyzing comments...")
            comments = fetch_comments(token, post_id)
            if not comments:
                st.warning("No comments found on this post.")
            else:
                sentiments = [classify_sentiment(c) for c in comments]
                df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

                st.subheader("💬 Comment Sentiment Analysis")
                st.bar_chart(df["Sentiment"].value_counts())
                st.dataframe(df)

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
else:
    st.info("Enter your access token and post ID to begin.")

import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

st.set_page_config(page_title="Facebook Post Analyser", layout="wide")

# Load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Fetch post metadata
def fetch_post_info(token, post_id):
    fields = "message,created_time,reactions.summary(true),shares,comments.summary(true),permalink_url,attachments{media_type,media,url}"
    url = f"https://graph.facebook.com/v18.0/{post_id}"
    params = {'access_token': token, 'fields': fields}
    return requests.get(url, params=params).json()

# Fetch comments
def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {'access_token': token, 'summary': 'true', 'limit': 100}
    res = requests.get(url, params=params).json()
    return [c["message"] for c in res.get("data", []) if "message" in c]

# Sentiment classifier
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

# App UI
st.title("📊 Facebook Post Analyser")

token = st.text_input("🔐 Facebook Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID")

if token and post_id:
    with st.spinner("Fetching post and comments..."):
        try:
            post = fetch_post_info(token, post_id)
            comments = fetch_comments(token, post_id)
            sentiments = [classify_sentiment(c) for c in comments]
            df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

            # Section 1: Post Details
            st.subheader("🧾 Post Details")
            st.write(f"📅 Created: {post.get('created_time', 'N/A')}")
            st.write(f"📝 Message: {post.get('message', 'No message')}")
            st.write(f"👍 Reactions: {post.get('reactions', {}).get('summary', {}).get('total_count', 0)}")
            st.write(f"🔄 Shares: {post.get('shares', {}).get('count', 0)}")
            st.write(f"💬 Comments: {post.get('comments', {}).get('summary', {}).get('total_count', 0)}")
            if post.get("permalink_url"):
                st.markdown(f"🔗 [View Post]({post['permalink_url']})")


            # Section 2: Sentiment Analysis
            st.subheader("💬 Comment Sentiment")
            st.bar_chart(df["Sentiment"].value_counts())
            st.dataframe(df)

        except Exception as e:
            st.error(f"❌ Error: {e}")

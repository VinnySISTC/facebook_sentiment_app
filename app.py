import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Page config
st.set_page_config(page_title="Facebook Post Analyser", layout="wide")

# Load BERT tokenizer and model
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

# Fetch comments
def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {
        'access_token': token,
        'summary': 'true',
        'limit': 100
    }
    response = requests.get(url, params=params)
    return [c["message"] for c in response.json().get("data", []) if "message" in c]

# Classify sentiment using BERT
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

# Streamlit UI
st.title("📊 Facebook Post Analyser")

token = st.text_input("🔐 Facebook Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID")

if token and post_id:
    st.success("✅ Token and Post ID entered")

    try:
        # Fetch post data
        st.info("Fetching post data...")
        post = fetch_post_details(token, post_id)
        if "error" in post:
            st.error(f"Facebook API Error: {post['error']['message']}")
        else:
            # Display Post Metadata
            st.subheader("🧾 Post Information")
            st.write(f"📅 Created: {post.get('created_time', 'N/A')}")
            st.write(f"📝 Message: {post.get('message', 'No message')}")
            st.write(f"👍 Reactions: {post.get('reactions', {}).get('summary', {}).get('total_count', 0)}")
            st.write(f"🔄 Shares: {post.get('shares', {}).get('count', 0)}")
            st.write(f"💬 Comments: {post.get('comments', {}).get('summary', {}).get('total_count', 0)}")
            if post.get("permalink_url"):
                st.markdown(f"🔗 [View Post on Facebook]({post['permalink_url']})")

            # Fetch and analyze comments
            st.info("Fetching comments...")
            comments = fetch_comments(token, post_id)
            if not comments:
                st.warning("No comments found.")
            else:
                sentiments = [classify_sentiment(c) for c in comments]
                df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

                # Side-by-side chart and table
                st.subheader("💬 Comment Sentiment Analysis")
                col1, col2 = st.columns([1, 2])  # Reduced width chart

                with col1:
                    st.markdown("**Sentiment Distribution**")
                    st.bar_chart(df["Sentiment"].value_counts())

                with col2:
                    st.markdown("**Classified Comments**")
                    st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
else:
    st.info("Enter your access token and post ID to begin.")

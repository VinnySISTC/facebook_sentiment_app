import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Page setup
st.set_page_config(page_title="Facebook Post Sentiment Analyzer", layout="wide")

# Load BERT model and tokenizer
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
        'fields': 'message,created_time,shares,likes.summary(true),comments.summary(true)'
    }
    response = requests.get(url, params=params)
    return response.json()

# Fetch all Facebook comments using pagination
def fetch_all_comments(token, post_id):
    comments = []
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {'access_token': token, 'limit': 100}

    while url:
        res = requests.get(url, params=params if '?' not in url else {}).json()
        batch = [c["message"] for c in res.get("data", []) if "message" in c]
        comments.extend(batch)
        url = res.get("paging", {}).get("next")

    return comments

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

# Streamlit interface
st.title("📘 Facebook Post Sentiment Analyzer")

token = st.text_input("🔐 Facebook Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID (e.g., 1234567890_0987654321)")

if token and post_id:
    st.success("✅ Token and Post ID entered")
    try:
        st.info("Fetching post details...")
        post = fetch_post_details(token, post_id)

        if "error" in post:
            st.error(f"Facebook API Error: {post['error']['message']}")
        else:
            st.subheader("🧾 Post Information")
            st.write(f"🗓️ Created: {post.get('created_time', 'N/A')}")
            st.write(f"📄 Message: {post.get('message', 'No message')}")
            st.write(f"👍 Likes: {post.get('likes', {}).get('summary', {}).get('total_count', 'N/A')}")
            st.write(f"💬 Total Comments: {post.get('comments', {}).get('summary', {}).get('total_count', 'N/A')}")
            st.write(f"🔁 Shares: {post.get('shares', {}).get('count', 'N/A')}")

            # Fetch and analyze comments
            st.info("Fetching and analyzing all comments...")
            comments = fetch_all_comments(token, post_id)
            if not comments:
                st.warning("No comments found on this post.")
            else:
                sentiments = [classify_sentiment(comment) for comment in comments]
                df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

                st.subheader("💬 Comment Sentiment Analysis")
                col1, col2 = st.columns([1, 2])

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

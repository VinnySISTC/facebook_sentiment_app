import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

st.set_page_config(page_title="Post Insight Analyser", layout="wide")

# Load BERT Model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Get comment text from the post
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

# Get all available post info
def fetch_post_info(token, post_id):
    fields = "message,created_time,reactions.summary(true),shares,comments.summary(true),permalink_url,attachments{media_type,media,url}"
    url = f"https://graph.facebook.com/v18.0/{post_id}"
    params = {'access_token': token, 'fields': fields}
    return requests.get(url, params=params).json()

# Main Interface
st.title("📊 Facebook Post Insight Analyser")

token = st.text_input("🔐 Facebook Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID")

if token and post_id:
    with st.spinner("Loading post data and comments..."):
        try:
            # Fetch and display post details
            post_data = fetch_post_info(token, post_id)
            st.subheader("🧾 Post Details")
            st.write(f"📅 Created: {post_data.get('created_time', 'N/A')}")
            st.write(f"📝 Message: {post_data.get('message', 'No message')}")
            st.write(f"👍 Reactions: {post_data.get('reactions', {}).get('summary', {}).get('total_count', 0)}")
            st.write(f"🔄 Shares: {post_data.get('shares', {}).get('count', 0)}")
            st.write(f"💬 Comments: {post_data.get('comments', {}).get('summary', {}).get('total_count', 0)}")
            if post_data.get("permalink_url"):
                st.markdown(f"🔗 [View Post]({post_data['permalink_url']})")

            # If media attached, show it
            media = post_data.get("attachments", {}).get("data", [{}])[0]
            if media.get("media_type") in ["photo", "video"]:
                media_url = media.get("media", {}).get("image", {}).get("src", "")
                if media_url:
                    st.image(media_url, caption="Post Media", use_column_width=True)

            # Comment Sentiment
            st.subheader("💬 Comment Sentiment")
            comments = fetch_comments(token, post_id)
            sentiments = [classify_sentiment(c) for c in comments]
            df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

            st.bar_chart(df["Sentiment"].value_counts())
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {e}")

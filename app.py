import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

# Set Streamlit Page Config FIRST
st.set_page_config(page_title="FB Sentiment BERT Analyzer", layout="wide")

# Load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Fetch comments from Facebook post
def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments"
    params = {'access_token': token, 'summary': 'true', 'limit': 100}
    res = requests.get(url, params=params).json()
    return [c["message"] for c in res.get("data", []) if "message" in c]

# Perform sentiment analysis using BERT
def classify(comment):
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

# Get page-level insights
def fetch_page_insights(token, page_id):
    metrics = "page_impressions,page_views_total,page_engaged_users"
    url = f"https://graph.facebook.com/v18.0/{page_id}/insights"
    params = {'metric': metrics, 'access_token': token}
    return requests.get(url, params=params).json()

# Streamlit Interface
st.title("📊 Facebook Comment Sentiment Analysis using BERT")

token = st.text_input("🔐 Facebook Graph API Access Token", type="password")
page_id = st.text_input("📄 Facebook Page ID")
post_id = st.text_input("📝 Facebook Post ID")

if token and page_id and post_id:
    with st.spinner("Analyzing comments..."):
        try:
            comments = fetch_comments(token, post_id)
            sentiments = [classify(c) for c in comments]
            df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

            # Display sentiment distribution
            st.subheader("📈 Sentiment Distribution")
            st.bar_chart(df["Sentiment"].value_counts())

            # Display comments with sentiment
            st.subheader("💬 Comments Table")
            st.dataframe(df)

            # Page Info
            st.subheader("📘 Page Information")
            info = fetch_page_info(token, page_id)
            st.write(f"Page Name: {info.get('name', 'N/A')}")
            st.write(f"Followers: {info.get('fan_count', 'N/A')}")

            # Page Insights
            st.subheader("📊 Page Insights")
            insights = fetch_page_insights(token, page_id)
            for item in insights.get("data", []):
                metric_name = item.get("name", "").replace("_", " ").title()
                value = item.get("values", [{}])[0].get("value", "N/A")
                st.write(f"{metric_name}: {value}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

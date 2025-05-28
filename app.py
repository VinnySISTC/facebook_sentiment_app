import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

def fetch_comments(token, post_id):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments?access_token={token}&summary=true&limit=100"
    r = requests.get(url)
    data = r.json()
    return [c["message"] for c in data.get("data", []) if "message" in c]

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

def fetch_page_info(token, page_id):
    url = f"https://graph.facebook.com/v18.0/{page_id}?fields=name,fan_count&access_token={token}"
    return requests.get(url).json()

def fetch_page_insights(token, page_id):
    metrics = "page_impressions,page_views_total,page_engaged_users"
    url = f"https://graph.facebook.com/v18.0/{page_id}/insights?metric={metrics}&access_token={token}"
    return requests.get(url).json()

st.set_page_config(page_title="FB Sentiment BERT", layout="wide")

st.title("📊 Facebook Comment Sentiment Analysis using BERT")

token = st.text_input("🔐 Access Token", type="password")
page_id = st.text_input("📄 Page ID")
post_id = st.text_input("📝 Post ID")

if token and page_id and post_id:
    with st.spinner("Fetching comments..."):
        comments = fetch_comments(token, post_id)
        sentiments = [classify(c) for c in comments]
        df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})
        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(df["Sentiment"].value_counts())

        st.subheader("💬 Comments Table")
        st.dataframe(df)

    st.subheader("📘 Page Info")
    info = fetch_page_info(token, page_id)
    st.write(f"Page Name: {info.get('name')}")
    st.write(f"Followers: {info.get('fan_count')}")

    st.subheader("📊 Page Insights")
    insights = fetch_page_insights(token, page_id)
    for item in insights.get("data", []):
        name = item["name"]
        value = item["values"][0]["value"]
        st.write(f"{name.replace('_', ' ').title()}: {value}")

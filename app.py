import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Facebook Sentiment Analyzer", layout="centered")

# Title
st.title("📘 Facebook Comment Sentiment Analyzer (BERT)")

# Load BERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_model()

# User inputs
access_token = st.text_input("🔐 Facebook Page Access Token", type="password")
post_id = st.text_input("📝 Facebook Post ID")

# Button
if st.button("Analyze Comments"):
    if not access_token or not post_id:
        st.warning("Please enter both Access Token and Post ID.")
    else:
        # STEP 1: Get Comments
        comment_url = f"https://graph.facebook.com/v19.0/{post_id}/comments"
        params = {"access_token": access_token, "limit": 100}
        all_comments = []

        while comment_url:
            res = requests.get(comment_url, params=params).json()
            for comment in res.get("data", []):
                text = comment.get("message")
                if text:
                    all_comments.append(text)
            comment_url = res.get("paging", {}).get("next")
            params = {}  # reset for next page

        if not all_comments:
            st.warning("No comments found or token permissions may be limited.")
        else:
            # STEP 2: Sentiment Analysis
            def classify(text):
                result = sentiment_model(text[:512])[0]
                label = result["label"].lower()
                if label in ["positive", "label_2"]:
                    return "Positive"
                elif label in ["negative", "label_0"]:
                    return "Negative"
                else:
                    return "Neutral"

            df = pd.DataFrame(all_comments, columns=["Comment"])
            df["Sentiment"] = df["Comment"].apply(classify)

            # STEP 3: Visualize
            st.subheader("📊 Sentiment Breakdown")
            sentiment_counts = df["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
            ax.set_ylabel("Number of Comments")
            ax.set_title("Comment Sentiment Distribution")
            st.pyplot(fig)

            # STEP 4: Show Table
            st.subheader("📋 Comment Sentiment Table")
            st.dataframe(df)

            # STEP 5: Download Option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "facebook_comments.csv", "text/csv")

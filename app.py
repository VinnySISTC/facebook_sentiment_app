import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from transformers import pipeline
from streamlit.components.v1 import html

# Set page configuration
st.set_page_config(
    page_title="Facebook Sentiment Analyzer",
    layout="wide",
    page_icon="📘"
)

# Load BERT sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_model()

# UI Inputs
st.title("📘 Facebook Sentiment & Insights Analyzer")

access_token = st.text_input("🔐 Facebook Access Token", type="password")
page_id = st.text_input("🆔 Facebook Page ID")
post_id = st.text_input("📝 Facebook Post ID (Full Format: PageID_PostID)")

if st.button("🚀 Run Analysis") and access_token and page_id and post_id:
    try:
        # PAGE INFO
        st.header("📄 Page Information")
        page_url = f"https://graph.facebook.com/v19.0/{page_id}"
        page_params = {
            "fields": "name,fan_count,link,talking_about_count,category",
            "access_token": access_token
        }
        page_data = requests.get(page_url, params=page_params).json()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Page Name", page_data.get("name", "N/A"))
            st.metric("Category", page_data.get("category", "N/A"))
        with col2:
            st.metric("Followers", page_data.get("fan_count", 0))
            st.metric("People Talking", page_data.get("talking_about_count", 0))

        # POST INSIGHTS
        st.header("📈 Post Insights")
        post_url = f"https://graph.facebook.com/v19.0/{post_id}?fields=created_time,message,likes.summary(true),comments.summary(true)&access_token={access_token}"
        post_data = requests.get(post_url).json()

        col3, col4 = st.columns(2)
        with col3:
            st.metric("👍 Likes", post_data.get("likes", {}).get("summary", {}).get("total_count", 0))
        with col4:
            st.metric("💬 Comments", post_data.get("comments", {}).get("summary", {}).get("total_count", 0))

        # COMMENTS
        st.header("💬 Comment Sentiment Analysis")
        comments_url = f"https://graph.facebook.com/v19.0/{post_id}/comments"
        comments_params = {
            "access_token": access_token,
            "limit": 100
        }
        comment_response = requests.get(comments_url, params=comments_params).json()
        comments_data = comment_response.get("data", [])

        if not comments_data:
            st.warning("No comments found or access denied.")
        else:
            comments_list = [c["message"] for c in comments_data if "message" in c]
            df = pd.DataFrame(comments_list, columns=["Comment"])
            df["Sentiment"] = df["Comment"].apply(
                lambda x: "Positive" if "label_2" in sentiment_model(x[:512])[0]['label']
                else ("Negative" if "label_0" in sentiment_model(x[:512])[0]['label'] else "Neutral")
            )

            col5, col6 = st.columns(2)
            with col5:
                st.markdown("#### 📊 Sentiment Chart")
                chart_data = df["Sentiment"].value_counts()
                fig, ax = plt.subplots()
                chart_data.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
                ax.set_ylabel("Count")
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)

            with col6:
                st.markdown("#### 📝 Comments Table")
                st.dataframe(df)

            st.download_button("📥 Download Comments CSV", df.to_csv(index=False).encode("utf-8"), "comments.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from transformers import pipeline
from streamlit.components.v1 import html

# Set page configuration
st.set_page_config(
    page_title="Facebook Dashboard",
    layout="wide",
    page_icon="📘",
    initial_sidebar_state="expanded",
    menu_items={"About": "Sentiment and Insights Analysis using BERT"}
)

# Load BERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_model()

# Set background color and style
st.markdown("""
    <style>
    body {
        background-color: #f1f4f8;
    }
    .block-container {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📘 Facebook Sentiment & Insights Dashboard</h1>", unsafe_allow_html=True)

# Input Section
st.subheader("🔐 Step 1: Enter Facebook Access Token")
access_token = st.text_input("Access Token", type="password", label_visibility="collapsed", placeholder="Enter your Facebook Access Token")

st.subheader("📝 Step 2: Enter Facebook Post ID Only")
post_suffix = st.text_input("Post ID Only (not full format)", placeholder="Example: 123456789012345")

page_id = ""
if access_token:
    # Automatically get first page ID
    pages_url = f"https://graph.facebook.com/v19.0/me/accounts?access_token={access_token}"
    res = requests.get(pages_url).json()
    pages = res.get("data", [])
    if pages:
        first_page = pages[0]
        page_id = first_page["id"]
        page_name = first_page.get("name", "Unknown")
        st.success(f"Using Page: {page_name} (ID: {page_id})")

if access_token and page_id and post_suffix:
    post_id = f"{page_id}_{post_suffix}"
    st.success(f"Formatted Post ID: {post_id}")

    # Get page info
    page_info_url = f"https://graph.facebook.com/v19.0/{page_id}?fields=name,fan_count,link,talking_about_count,category&access_token={access_token}"
    page_data = requests.get(page_info_url).json()

    # Get posts and insights
    posts_url = f"https://graph.facebook.com/v19.0/{page_id}/posts"
    posts_params = {
        'fields': 'id,message,created_time,likes.summary(true),comments.summary(true)',
        'limit': 5,
        'access_token': access_token
    }
    posts_data = requests.get(posts_url, params=posts_params).json()
    posts = posts_data.get("data", [])

    st.subheader("📄 Page Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Page Name", page_data.get("name", "-"))
        st.metric("Category", page_data.get("category", "-"))
    with col2:
        st.metric("Followers", page_data.get("fan_count", 0))
        st.metric("People Talking", page_data.get("talking_about_count", 0))

    if not posts:
        st.warning("No posts found or access restricted.")
    else:
        for post in posts:
            if post["id"] == post_id:
                st.markdown(f"### 📌 Post ID: {post['id']}")

                # COMMENTS
                comment_url = f"https://graph.facebook.com/v19.0/{post['id']}/comments"
                params = {"access_token": access_token, "limit": 100}
                comments = []
                response = requests.get(comment_url, params=params).json()
                for comment in response.get("data", []):
                    text = comment.get("message")
                    if text:
                        comments.append(text)

                if comments:
                    df_comments = pd.DataFrame(comments, columns=["Comment"])
                    df_comments["Sentiment"] = df_comments["Comment"].apply(
                        lambda x: "Positive" if "label_2" in sentiment_model(x[:512])[0]['label']
                        else ("Negative" if "label_0" in sentiment_model(x[:512])[0]['label'] else "Neutral")
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📊 Sentiment Chart")
                        chart_data = df_comments["Sentiment"].value_counts()
                        fig, ax = plt.subplots()
                        chart_data.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_title("Sentiment Distribution")
                        st.pyplot(fig)

                    with col2:
                        st.markdown("#### 📝 Comments Table")
                        st.dataframe(df_comments)

                    st.download_button("Download Comments CSV", df_comments.to_csv(index=False).encode("utf-8"), "comments.csv", "text/csv")
                else:
                    st.info("No comments found.")

                # POST INSIGHTS
                st.markdown("#### 📈 Post Insights")
                likes = post.get("likes", {}).get("summary", {}).get("total_count", 0)
                comments_count = post.get("comments", {}).get("summary", {}).get("total_count", 0)
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("👍 Likes", likes)
                    with col2:
                        st.metric("💬 Comments", comments_count)

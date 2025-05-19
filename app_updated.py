import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Manually defined stopwords
custom_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now"
])

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

# Load model and vectorizer
model = joblib.load("xgboost_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Load fact-checked claims
claims_df = pd.read_csv("scraped_claims.csv")
claims_df["clean_claim"] = claims_df["claim_text"].apply(clean_text)

# Streamlit UI
st.set_page_config(page_title="Kenya Hate Speech & Misinformation Detector", page_icon="🇰🇪")
st.title("🇰🇪 Kenya Hate Speech & Misinformation Detector")
st.markdown("Detect hate speech and misinformation in political tweets in real-time.")

# Sensitivity slider
threshold = st.slider("Hate Speech Threshold (Lower = More Sensitive)", 0.1, 0.9, 0.5, step=0.05)

# Single tweet analysis
tweet_input = st.text_area("✍️ Enter a tweet:", height=150)

if st.button("Analyze Tweet"):
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet_input)
        vectorized = vectorizer.transform([cleaned])
        prob = model.predict_proba(vectorized)[0][1]
        prediction = 1 if prob >= threshold else 0

        # Misinformation detection
        all_texts = list(claims_df["clean_claim"]) + [cleaned]
        misinfo_vectorizer = TfidfVectorizer().fit(all_texts)
        tweet_vec = misinfo_vectorizer.transform([cleaned])
        claims_vec = misinfo_vectorizer.transform(claims_df["clean_claim"])
        similarity_scores = cosine_similarity(tweet_vec, claims_vec)
        is_misinfo = similarity_scores.max() >= 0.85

        st.markdown(f"**Confidence:** `{prob:.2%}` (Threshold: {threshold})")
        if prediction == 1:
            st.error("🚨 Hate Speech Detected")
        else:
            st.success("✅ Safe Tweet")

        if is_misinfo:
            st.warning("⚠️ Matches known misinformation (per PesaCheck or AfricaCheck)")

st.divider()

# Batch processing section
st.markdown("### 📥 Upload CSV (with 'text' column) for Batch Classification")
uploaded_file = st.file_uploader("Choose CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("❌ CSV must include a column named 'text'.")
        else:
            df["clean_text"] = df["text"].apply(clean_text)
            X = vectorizer.transform(df["clean_text"])
            probs = model.predict_proba(X)[:, 1]
            df["confidence"] = probs
            df["prediction"] = (probs >= threshold).astype(int)
            df["label"] = df["prediction"].map({0: "Safe", 1: "Hate Speech"})

            # Misinformation detection
            all_clean_claims = claims_df["clean_claim"].tolist()
            all_texts = all_clean_claims + df["clean_text"].tolist()
            misinfo_vectorizer = TfidfVectorizer().fit(all_texts)
            claims_vec = misinfo_vectorizer.transform(all_clean_claims)
            tweet_vecs = misinfo_vectorizer.transform(df["clean_text"])
            sim_matrix = cosine_similarity(tweet_vecs, claims_vec)
            df["misinformation_flag"] = (sim_matrix.max(axis=1) >= 0.85)

            st.success("Batch analysis complete.")
            st.dataframe(df[["text", "label", "confidence", "misinformation_flag"]].head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Results",
                data=csv,
                file_name="classified_tweets.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")



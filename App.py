

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer




st.markdown("## 📰 AI News Anomaly Detection Dashboard")



@st.cache_resource
def load_models():
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca_model.pkl")
    mean_embedding = joblib.load("mean_embedding.pkl")
    iso_model = joblib.load("iso_model.pkl")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    loc_stats = joblib.load("loc_stats.pkl")
    nlp = spacy.load("en_core_web_sm")
    analyzer = SentimentIntensityAnalyzer()

    return model, le, embedder, mean_embedding, iso_model, nlp, loc_stats,scaler,pca,analyzer

model, le, embedder, mean_embedding, iso_model, nlp, loc_stats,scaler,pca,analyzer = load_models()

@st.cache_data
def load_data():
    df_csv = pd.read_csv("Articles.csv", encoding="latin1")
    return df_csv

df_csv = load_data()

@st.cache_data
def get_embedding(text):
    return embedder.encode([text]).astype(np.float32)


def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]
def label_extraction(text):
    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents]
    return list(set(labels))
def location_extraction(text):
    doc = nlp(text)
    locations = [ent.text.lower() for ent in doc.ents if ent.label_ in ["GPE","LOC"]]
    # remove numbers and duplicates
    locations = [loc for loc in locations if not any(char.isdigit() for char in loc)]
    
    return list(set(locations))

import re

COUNTRY_KEYWORDS = {
    "pakistan": ["punjab", "sindh", "karachi", "lahore", "islamabad", "quetta"],
    "india": ["maharashtra", "karnataka", "tamil nadu", "delhi", "mumbai", "bangalore", "kerala"],
    "usa": ["california", "texas", "new york", "florida", "washington", "chicago"],
    "uk": ["london", "manchester", "birmingham", "scotland", "wales"],
    "china": ["beijing", "shanghai", "guangdong", "shenzhen"],
    "japan": ["tokyo", "osaka", "kyoto"],
    "sri lanka": ["colombo", "kandy", "galle"]
}

def keyword_feature(text):
    text = text.lower()
    
    features = []

    for country, regions in COUNTRY_KEYWORDS.items():
        country_score = 0
        region_score = 0

        if country in text:
            country_score = 1

        for r in regions:
            if r in text:
                region_score += 1

        features.append(country_score)
        features.append(region_score)

    return features

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)

    compound = scores['compound']

    if compound >= 0.05:
        label = "Positive 😊"
    elif compound <= -0.05:
        label = "Negative 😡"
    else:
        label = "Neutral 😐"

    return compound, scores, label

def temporal_score(predicted_location,loc_stats):
    row = loc_stats[loc_stats['location'] == predicted_location]
    if row.empty:
        return 0.0

    loc_mean = row['loc_mean'].values[0]
    loc_std  = row['loc_std'].values[0] + 1e-6

    # expected baseline
    expected = loc_mean

    # for a single article, observed ≈ 1 event
    observed = 1.0

    # z-score style deviation
    z = (observed - expected) / loc_std
    return abs(z)

tab1, tab2 = st.tabs(["🔍 Live Analysis", "📊 Model Insights"])

with tab1:

    mode = st.radio("Choose Input Mode", ["Enter News Article", "Select from Dataset"])

    user_text = None

    if mode == "Enter News Article":
        input_text = st.text_area("Enter news article here:")
        if input_text:
            user_text = input_text

    elif mode == "Select from Dataset":
        selected = st.selectbox("Select article", df_csv['Article'])
        if selected:
            user_text = selected

    if st.button("Analyze") and user_text:
    # ensure string
        if isinstance(user_text, list):
            user_text = user_text[0]

        # FORCE clean string
        clean_text = str(user_text)

        #Sentiment analysis
        compound, scores, sentiment_label = get_sentiment(clean_text)

    # embedding only → PCA
        embedding = get_embedding(clean_text)
        # keyword features separately
        kw_features = np.array(keyword_feature(clean_text)).reshape(1, -1)

        # scale features
        input_features = np.hstack([embedding, kw_features])
        input_scaled = scaler.transform(input_features)
        embedding_scaled = input_scaled[:, :embedding.shape[1]]
        kw_features_scaled = input_scaled[:, embedding.shape[1]:]
        embedding_reduced = pca.transform(embedding_scaled)
    

        # final input
        final_input = np.hstack([embedding_reduced, kw_features_scaled])

        # prediction
        probs = model.predict_proba(final_input)
        confidence = float(probs.max(axis=1)[0])
        pred = probs.argmax(axis=1)[0]
        predicted_location = le.inverse_transform([pred])[0]

        mean_embedding = np.array(mean_embedding).astype(np.float32)

        sim = np.dot(embedding, mean_embedding) / (
        np.linalg.norm(embedding) * np.linalg.norm(mean_embedding)
    )
        sim = sim.item()  # convert to scalar

        linguistic_score = 1 - sim

        iso_score = -iso_model.score_samples(embedding)[0]

        # # safe normalization
        ling_norm = (linguistic_score - 0) / (1 - 0)   # already 0–1 usually

        iso_norm = float(iso_score)
        temp_score = temporal_score(predicted_location, loc_stats)

        final_anomaly_score = (
            0.4 * ling_norm +
            0.3 * iso_norm +
            0.3 * temp_score
        )
        
        print(np.percentile(confidence, [25, 50, 75, 90]))
        # Determine anomaly flag

        if final_anomaly_score > 0.6:
            anomaly_flag = "🚨 High Anomaly"
        elif final_anomaly_score > 0.3:
            anomaly_flag = "⚠️ Medium Anomaly"
        else:
            anomaly_flag = "✅ Normal"

        st.write(f"### Anomaly Score: {round(float(final_anomaly_score), 2)}")
        # NER
        named_entities = extract_named_entities(clean_text)
        locations = location_extraction(clean_text)
        newstype = label_extraction(clean_text)

        if locations:
            ner_text = " ".join(locations).lower()
            features = kw_features[0].tolist()  # get the keyword features for this text

            scores = []
            countries = list(COUNTRY_KEYWORDS.keys())

            for i in range(0, len(features), 2):
                score = features[i] + features[i+1]
                scores.append(score)

            if max(scores) == 0:
                ner_country = "unknown"
            else:
                ner_country = countries[np.argmax(scores)]

        else:
                    ner_country = "unknown"

        if final_anomaly_score < 0.3 and confidence > 0.6 and ner_country == predicted_location:
            anomaly_flag = "✅ Likely Normal News"

        if confidence < 0.5:
            anomaly_flag = "🚨 Suspicious (Low Confidence)"

        if ner_country != "unknown" and ner_country != predicted_location:
            anomaly_flag += " | 🚨 Source Mismatch"

        if abs(compound) > 0.7:
            st.warning("⚠️ Strong emotional tone detected")

        if abs(compound) > 0.7 and final_anomaly_score > 0.5:
            st.error("🚨 High risk: Emotional + Anomalous content")

        st.write("Anomaly flag: ", anomaly_flag)

        st.subheader("💬 Sentiment Analysis")

        c1, c2, c3 = st.columns(3)

        c1.metric("Sentiment", sentiment_label)
        c2.metric("Compound Score", round(compound, 2))
        c3.metric("Confidence", round(confidence, 2))

        # final decision

        if ner_country == predicted_location:
            final_location = predicted_location

        elif confidence > 0.85:
            final_location = predicted_location   # strong model trust

        elif ner_country != "unknown":
            final_location = ner_country          # fallback to NER

        else:
            final_location = predicted_location

        risk_score = (
        final_anomaly_score * 0.5 +
        (1 if confidence < 0.5 else 0) * 0.2 +
        (1 if (ner_country != "unknown" and ner_country != predicted_location) else 0) * 0.2+
        (1 if temp_score > 2 else 0) * 0.1
    )

        st.subheader("🚨 Overall Risk Score")
        risk_score = float(risk_score)
        st.progress(risk_score)
        st.write(f"Risk Score: {round(float(risk_score), 2)}")

        c1, c2 = st.columns(2)

        c1.metric("Linguistic", round(float(ling_norm), 2))
        c2.metric("Isolation", round(float(iso_norm), 2))
        

        if temp_score == 0:
            st.info("Temporal analysis not applicable for single article")
        else:
            st.metric("Temporal Score", round(temp_score, 2))

        st.subheader("🌍 Source Verification")

            
        if ner_country == "unknown":
            st.info("ℹ️ No clear location found in text")

        elif predicted_location == ner_country:
            st.success("✅ Model and NER agree")

        elif confidence > 0.8:
            st.warning("⚠️ Model trusted over NER (high confidence)")

        else:
            st.error("🚨 Source mismatch detected")

        import matplotlib.pyplot as plt
        st.subheader("📊 Anomaly Score Breakdown")

        labels = ['Linguistic', 'Isolation', 'Temporal']
        values = [ling_norm, iso_norm, temp_score]
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        st.subheader("🤖 Model Confidence")

        st.progress(float(confidence))

        if confidence < 0.5:
            st.warning("Low confidence prediction")
        else:
            st.success("High confidence prediction")


        st.subheader("🧾 Why flagged?")

        reasons = []

        if ling_norm > 0.4:
            reasons.append("Unusual writing pattern")

        if iso_norm > 0.5:
            reasons.append("Statistical anomaly")

        if temp_score > 2:
            reasons.append("Temporal spike")

        if confidence < 0.5:
            reasons.append("Low confidence")

        for r in reasons:
            st.warning(r)

        st.subheader("🌍 Location Analysis")

        c1, c2,c3 = st.columns(3)

        c1.metric("🤖 Model Prediction:", predicted_location)
        c2.metric("📌 NER Location:", ner_country)
        c3.metric("🎯 Final Decision:", final_location)
        

        if predicted_location == ner_country:
            st.success("✅ Location Match")
        else:
            st.warning("⚠️ Location Mismatch")

    with tab2:
          
        st.subheader("📊 Model Training Insights")

        cm = joblib.load("confusion_matrix.pkl")
        labels = le.classes_

        import matplotlib.pyplot as plt

        st.markdown("### 🔍 Confusion Matrix")

        fig, ax = plt.subplots()
        im = ax.imshow(cm)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))

        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)
        
        from collections import Counter

        class_counts = joblib.load("class_distribution.pkl")

        st.markdown("### 🌍 Training Data Distribution")

        fig, ax = plt.subplots()
        ax.bar(class_counts.keys(), class_counts.values())

        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("### 🧠 Feature Importance")

        importance = model.feature_importances_

        num_embed = pca.n_components_
        kw_imp = importance[num_embed:]

        kw_names = []
        for c in COUNTRY_KEYWORDS.keys():
            kw_names.append(f"{c}_flag")
            kw_names.append(f"{c}_region")

        fig, ax = plt.subplots()
        ax.barh(kw_names, kw_imp)

        st.pyplot(fig)
   
   
        st.markdown("### 🤖 Confidence Distribution")

        conf_scores = joblib.load("confidence_scores.pkl")

        fig, ax = plt.subplots()
        ax.hist(conf_scores, bins=30)

        st.pyplot(fig)

        st.markdown("### 📋 Classification Report")

        report = joblib.load("classification_report.pkl")
        df_report = pd.DataFrame(report).transpose()

        st.dataframe(df_report)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![NLP](https://img.shields.io/badge/NLP-Transformers-orange)
![Model](https://img.shields.io/badge/Model-XGBoost-green)
![Embeddings](https://img.shields.io/badge/Embeddings-MPNet-red)
![Framework](https://img.shields.io/badge/Framework-Streamlit-pink)
![Cloud](https://img.shields.io/badge/Deployment-AWS-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-73%25-brightgreen)







🎯 Problem Statement

Detect misleading or unusual news articles by analyzing:

Content semantics

Location consistency

Temporal behavior

🧠 System Architecture
User Input → Embedding Model → Feature Engineering →

    → Anomaly Detection (Hybrid)
    
    → XGBoost Classifier → Prediction + Confidence
    
⚙️ Tech Stack

NLP: Sentence Transformers (MPNet)

ML: XGBoost

Anomaly Detection: Isolation Forest + Cosine Similarity

Temporal anomaly detection:Statistical baselines

Visualization: Matplotlib

Deployment: AWS EC2 + Streamlit

🚀 Key Features

Hybrid anomaly detection (semantic + statistical)

Confidence-based prediction filtering

Source discrepancy identification

Interactive Streamlit web application

Here comes the coding part:

      📊 Data Preprocessing
      
    from collections import Counter

    import joblib
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import spacy
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from prophet import Prophet
    import xgboost as xgb
    from xgboost import XGBClassifier,callback
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim  
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    import re
    from xgboost import XGBClassifier
    from sklearn.decomposition import PCA

            
        # -------------------------------
        # 1. LOAD DATA
        # -------------------------------
        df = pd.read_csv("Articles.csv", encoding="latin1")
        
        # -------------------------------
        # 2. BASIC CLEANING
        # -------------------------------
        df.columns = df.columns.str.strip()
        
        df['Article'] = df['Article'].fillna("").str.lower()
        df['Heading'] = df['Heading'].fillna("").str.lower()
        
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Combine text ONCE
        df['text'] = df['Heading'] + " " + df['Article']
        
        # Remove special characters
        df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # -------------------------------
        # 3. LOAD SPACY MODEL
        # -------------------------------
        nlp = spacy.load("en_core_web_md")  
        
        # -------------------------------
        # 4. LOCATION EXTRACTION
        # -------------------------------
        def extract_location(text):
            doc = nlp(text)
            locs = [ent.text.lower() for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        
            locs = [l for l in locs if not any(c.isdigit() for c in l)]
        
            return locs[0] if locs else "unknown"
        
        df['location'] = df['text'].apply(extract_location)
        
        # -------------------------------
        # 5. NORMALIZE LOCATION (CRITICAL)
        # -------------------------------
        def normalize_location(x):
            x = str(x).lower()
        
            if any(k in x for k in ["pakistan", "karachi", "islamabad"]):
                return "pakistan"
        
            if any(k in x for k in ["uk", "england", "london"]):
                return "uk"
        
            if any(k in x for k in ["usa", "us", "new york", "texas"]):
                return "usa"
        
            if "india" in x:
                return "india"
        
            if "china" in x or "hong kong" in x:
                return "china"
        
            if "uae" in x or "dubai" in x:
                return "uae"
        
            if "japan" in x or "tokyo" in x:
                return "japan"
        
            if "australia" in x:
                return "australia"
        
            if "new zealand" in x:
                return "new zealand"
        
            if "sri lanka" in x:
                return "sri lanka"
        
            if "south africa" in x:
                return "south africa"
        
            if "france" in x:
                return "france"
        
            if "iran" in x:
                return "iran"
        
            if "saudi" in x:
                return "saudi arabia"
        
            return "other"
        
        
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
        df['location'] = df['location'].apply(normalize_location)
        
        # -------------------------------
        # 6. REMOVE NOISE
        # -------------------------------
        df = df[df['location'] != "unknown"]
        df = df[df['location'] != "other"]
        
        # print(df.columns)
        # Remove rare classes
        counts = df['location'].value_counts()
        df = df[df['location'].isin(counts[counts >= 50].index)]
        
        keyword_features = np.array(
            df['text'].apply(keyword_feature).tolist()
        )
        joblib.dump(keyword_features, "keyword_features.pkl")
        # -------------------------------
        # 7. EMBEDDINGS (ONLY ONCE)
        # -------------------------------
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        df['embedding'] = list(model.encode(df['text'].tolist(), show_progress_bar=True))
        df['keyword_features'] = list(keyword_features)
        
        # # # -------------------------------
        # # # 8. FINAL DATA
        # # # -------------------------------
        embeddings = np.vstack(df['embedding'].values)   # (n, 384)
        keywords = np.array(df['keyword_features'].tolist())  # (n, 7)

🔍 Linguistic Anomaly Detection (Hybrid)
      
    def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

        

          df['anomaly_score_norm'] = 0.0
        df['iso_score'] = 0.0
        df['final_score'] = 0.0
        embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
        
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(embeddings)
        
        
        # joblib.dump(iso, "iso_model.pkl")
        mean_embedding = np.mean(embeddings, axis=0)
        # joblib.dump(mean_embedding, "mean_embedding.pkl")
        # cosine anomaly
        scores = cos_sim(embeddings, mean_embedding).cpu().numpy()
        anomaly_scores = 1 - scores.flatten()
        
        # normalize
        cosine_norm = minmax(anomaly_scores)
        
        # iso
        iso_scores = -iso.score_samples(embeddings)
        iso_norm = minmax(iso_scores)
        
        min_val = iso_scores.min()
        max_val = iso_scores.max()
        
        # joblib.dump((min_val, max_val), "iso_scaler.pkl")
        
        # final score
        final_score = 0.5 * cosine_norm + 0.5 * iso_norm
        
        joblib.dump(final_score, "anomaly_scores.pkl")

⏳ Temporal Anomaly Detection
         
        # df['date'] = pd.to_datetime(df['Date'])
        # df['dayofweek'] = df['date'].dt.dayofweek   # 0=Mon
        # df['month'] = df['date'].dt.month
        
        # # counts per location per day
        # daily_counts = df.groupby(['location', 'date']).size().reset_index(name='count')
        
        # # mean & std per location
        # loc_stats = daily_counts.groupby('location')['count'].agg(['mean','std']).reset_index()
        # loc_stats.rename(columns={'mean':'loc_mean','std':'loc_std'}, inplace=True)
            
          
🌍 Source Discrepancy Detection

            X = np.hstack([embeddings, keyword_features])
            y= df['location']
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            joblib.dump(scaler, "scaler.pkl")
            
            pca = PCA(n_components=0.95, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
            
            joblib.dump(pca, "pca_model.pkl")
            
            X_final = np.hstack([embeddings_reduced, keywords])
            
            le = LabelEncoder()
            y = le.fit_transform(df['location'])
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X_final, y, df.index, test_size=0.2, random_state=42, stratify=y
            )
            
            joblib.dump(le, "label_encoder.pkl")
            
            weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train
            )
            model1 = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=7,
                eval_metric='mlogloss',
            
                # core parameters
                max_depth=2,
                learning_rate=0.05,
                n_estimators=150,
            
                # 🔥 L1 regularization (feature sparsity)
                reg_alpha=0.2,
            
                # 🔥 L2 regularization (weight smoothing)
                reg_lambda=2.0,
            
                # extra regularization (VERY IMPORTANT)
                subsample=0.8,
                colsample_bytree=0.7,
                gamma=0.1  
            )
            
            model1.fit(
                X_train, y_train,
                sample_weight=weights,
                verbose=True
            )
            y_pred = model1.predict(X_test)
            
            # print(classification_report(y_test, y_pred))
            from sklearn.model_selection import cross_val_score
            
            scores = cross_val_score(model1, X_final, y, cv=5)
            # print(scores, scores.mean())
            
            import joblib
            
            # # Save model
            # joblib.dump(model, "xgb_model.pkl")
            
            # # Save label encoder
            # joblib.dump(le, "label_encoder.pkl")
            
            # print("✅ Models saved successfully")

# Prediction + Confidence
    
      # # Source discrepancy anomaly
            preds = model1.predict(X_test)
            probs = model1.predict_proba(X_test)
            
            confidence = probs.max(axis=1)
            
            final_labels = []
            
            for i in range(len(preds)):
                if confidence[i] < 0.6:
                    final_labels.append("Uncertain")
                else:
                    final_labels.append(le.inverse_transform([preds[i]])[0])
            
            df.loc[idx_test, 'confidence'] = confidence
            df.loc[idx_test, 'final_prediction'] = final_labels
            
            df.loc[idx_test, 'source_discrepancy'] = (
                df.loc[idx_test, 'location'] != df.loc[idx_test, 'final_prediction']
            )
            
            # print(np.percentile(confidence, [25, 50, 75, 90]))
            mask = np.array(final_labels) != "Uncertain"
            
            y_test_labels = le.inverse_transform(y_test)
            final_labels = np.array(final_labels)
            
            filtered_acc = accuracy_score(
                y_test_labels[mask],
                final_labels[mask]
            )
            
            coverage = mask.mean()
            
            # print("Filtered Accuracy:", filtered_acc)
            # print("Coverage:", coverage)
            train_preds = model1.predict(X_train)
            test_preds = model1.predict(X_test)
            
            # print("Train accuracy:", accuracy_score(y_train, train_preds))
            # print("Test accuracy:", accuracy_score(y_test, test_preds))
            
            joblib.dump(confusion_matrix(y_test, y_pred), "confusion_matrix.pkl")
            joblib.dump(Counter(y_train), "class_distribution.pkl")
            joblib.dump(classification_report(y_test, y_pred, output_dict=True), "classification_report.pkl")
            joblib.dump(confidence, "confidence_scores.pkl")

            importance = model1.feature_importances_
            num_embed = pca.n_components_
            embed_imp = importance[:num_embed]
            kw_imp = importance[num_embed:]
            joblib.dump(kw_imp, "Keyword_feature_importance.pkl")

## 🚀 Model Performance

* **Test Accuracy:** 81%
* **Train Accuracy:** 91%
* **Filtered Accuracy:** 88%
* **Coverage:** 77%

---

## 🎯 Highlights

* Well-balanced model with **minimal overfitting**
* Confidence-based filtering improves reliability to **88% accuracy**
* Covers **~77% of inputs** with high-confidence predictions
* Designed as a **hybrid system** combining embeddings, keyword signals, anomaly detection, and NER

---

## 🧠 Key Insight

> The model prioritizes **high-confidence, trustworthy predictions** over forcing decisions on uncertain inputs—making it more reliable for real-world usage.




📊 Visualization

1. Anomaly Distribution

        # flag low confidence OR mismatch
        df['source_flag'] = (df['source_discrepancy']) | (df['confidence'] < 0.5)
        
        df[df['source_flag'] == 1][['Heading', 'location', 'predicted_location', 'confidence']].head(10)
        
        df['source_flag'].value_counts().plot(kind='bar')
        plt.title("Source Discrepancy Count")
        plt.show()

   <img width="1280" height="612" alt="Source flag" src="https://github.com/user-attachments/assets/a17765a8-3a08-4847-abc9-45fd54a18937" />

2. Top Anomalies

             # top_anomalies = df.sort_values(by='final_score', ascending=False).head(10)

            # plt.figure(figsize=(10, 6))
            # plt.barh(top_anomalies['Heading'], top_anomalies['final_score'])
            # plt.xlabel('Anomaly Score')
            # plt.ylabel('Heading')
            # plt.title('Top 10 Anomalous News Articles')
            
            # plt.gca().invert_yaxis()  # highest on top
            # plt.tight_layout()
            # plt.show()
   
<img width="1280" height="612" alt="Top 10 Anomalous articles - latest" src="https://github.com/user-attachments/assets/a1decd71-8ff5-4258-973f-d61d1b5cb5bb" />

3. Anomaly Type Distribution

        def anomaly_type(row):
            if row['location'] != row['predicted_location'] and row['confidence'] > 0.7:
                return "Strong Mismatch"
            elif row['confidence'] < 0.6:
                return "Low Confidence"
            else:
                return "Normal"
        
        df['anomaly_type'] = df.apply(anomaly_type, axis=1)
        counts = df['anomaly_type'].value_counts()
        plt.figure(figsize=(8,5))
        counts.plot(kind='bar')
        plt.title("Anomaly Type Distribution")
        plt.xlabel("Type")
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()

   <img width="800" height="500" alt="Anomaly type distribution after tuning" src="https://github.com/user-attachments/assets/2e01a86e-ca69-4c0c-8444-95f731810821" />

5.Top locations by Average Anomaly score
        
        # location_anomalies = df.groupby('location')['final_score'].mean().sort_values(ascending=False)
        # plt.figure(figsize=(10, 6))
        # location_anomalies.head(10).plot(kind='bar')
        # plt.title('Top Locations by Average Anomaly Score')
        # plt.ylabel('Average Anomaly Score')
        # plt.xlabel('Location')
        
        # plt.tight_layout()
        # plt.show()

<img width="1280" height="612" alt="Top locations by anomaly score-latest" src="https://github.com/user-attachments/assets/fe9c2beb-e2c0-4037-96c5-cd1130f3f1a0" />





     
    






  


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

Time Series: Prophet

Visualization: Matplotlib

Deployment: AWS EC2 + Streamlit

🚀 Key Features

Hybrid anomaly detection (semantic + statistical)

Confidence-based prediction filtering

Source discrepancy identification

Temporal trend analysis

Interactive Streamlit web application

Here comes the coding part:

      📊 Data Preprocessing
      
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
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from bertopic import BERTopic

    
      nltk.download('punkt_tab')
      nltk.download('stopwords')
      nltk.download('wordnet')
      df['Article'] = df['Article'].str.lower()
      df['Heading'] = df['Heading'].str.lower()
      df['date'] = pd.to_datetime(df['Date'])
      df['NewsType'] = df['NewsType'].astype('category')
      stop_words = set(stopwords.words('english'))
      lemmatizer = WordNetLemmatizer()
      df['Article'] = df['Article'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
      df['tokens'] = df['Article'].apply(word_tokenize)
    
      df['tokens_headings'] = df['Heading'].apply(word_tokenize)
      df['tokens'] = df['tokens'].apply(
          lambda x: [word for word in x if word.isalpha() and word not in stop_words]
      )
    
      df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
      
      # Display the preprocessed tokens
      df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))
      nlp = spacy.load("en_core_web_md")
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
      df['named_entities'] = df['Article'].apply(extract_named_entities)
      df['Newstype'] = df['Article'].apply(label_extraction)
      df['location'] = df['Article'].apply(location_extraction)
      df["combined_text"] = df["Heading"] + "[SEP]" + df["Article"]
      
      🧠 Embedding Generation
      
      from sentence_transformers import SentenceTransformer
      
      model = SentenceTransformer("all-mpnet-base-v2")
      
      df['embedding'] = df['combined_text'].apply(lambda x: model.encode(x))

🔵 BERTopic (Topic Extraction)

    texts = df["combined_text"].tolist()
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(texts)
    
    df["topic"] = topics
    
    # Topic frequency
    
    topic_counts = df['topic'].value_counts()
    df['topic_freq'] = df['topic'].map(topic_counts)
    
    # Rare topic = anomaly
    df['topic_anomaly'] = df['topic_freq'] < 5

    topics_over_time = topic_model.topics_over_time(
    texts,
    df["date"],
    nr_bins=20
    )
    fig = topic_model.visualize_topics()
    fig.show()
      
🔍 Linguistic Anomaly Detection (Hybrid)
      
      from sklearn.ensemble import IsolationForest
      from sentence_transformers.util import cos_sim
      
      df.columns = df.columns.str.strip()
      df = df.dropna(subset=['Newstype', 'location'])
      
      def clean_location(x):
          if isinstance(x, list):
              return x[0] if len(x) > 0 else 'Unknown'
          return str(x)
      
      df['location'] = df['location'].apply(clean_location)
      def clean_newstype(x):
      if isinstance(x, list):
          return x[0] if len(x) > 0 else 'Unknown'
      return str(x)
  
      df['Newstype'] = df['Newstype'].apply(clean_newstype)
      
      df['anomaly_score_norm'] = 0.0
      df['iso_score'] = 0.0
      df['final_score'] = 0.0
      
      for key, group in df.groupby(['NewsType', 'location']):
          
          if len(group) < 5:
              continue  # s
          embeddings = np.vstack(df['embedding'].values)
      
      # Cosine similarity anomaly
          mean_embedding = np.mean(embeddings, axis=0)
    
          scores = cos_sim(embeddings, mean_embedding)
          anomaly_scores = 1 - scores.flatten()
          
          std = anomaly_scores.std()
          
          if std != 0:
              z_scores = (anomaly_scores - anomaly_scores.mean()) / std
          else:
              z_scores = np.zeros(len(group))
          
          # -------------------------------
          # 🔹 2. Isolation Forest
          # -------------------------------
          iso = IsolationForest(contamination=0.1, random_state=42)
          iso.fit(embeddings)
          
          iso_scores = -iso.score_samples(embeddings)  # higher = more anomalous
          
          # normalize iso scores
          iso_std = iso_scores.std()
          
          if iso_std != 0:
              iso_scores_norm = (iso_scores - iso_scores.mean()) / iso_std
          else:
              iso_scores_norm = np.zeros(len(group))
          
          # -------------------------------
          # 🔹 3. Combine (Hybrid Score)
          # -------------------------------
          z_scores = np.asarray(z_scores).flatten()
          iso_scores_norm = np.asarray(iso_scores_norm).flatten()
          
          # Final hybrid score
          final_score = 0.5 * z_scores + 0.5 * iso_scores_norm
          
          # -------------------------------
          # 🔹 Store Results
          # -------------------------------
          df.loc[group.index, 'anomaly_score_norm'] = z_scores.tolist()
          df.loc[group.index, 'iso_score'] = iso_scores_norm.tolist()
          df.loc[group.index, 'final_score'] = final_score.tolist()

          report = df.sort_values(by='final_score', ascending=False)[['Heading', 'NewsType', 'location', 'final_score']].head(10)

⏳ Temporal Anomaly Detection
         
        from prophet import Prophet
              
        df = df.sort_values(by='date')
        split_date = '2016-06-01'
        
        train_df = df[df['date'] < split_date]
        test_df  = df[df['date'] >= split_date]
        
        df['prophet_score'] = 0.0
        df['temporal_anomaly_flag'] = 0
        
        for loc in df['location'].unique():
            
            train_group = train_df[train_df['location'] == loc]
            test_group  = test_df[test_df['location'] == loc]
            
            if len(train_group) < 10 or len(test_group) == 0:
                continue
            
            # Prepare training data
            prophet_train = train_group[['date', 'final_score']].rename(
                columns={'date': 'ds', 'final_score': 'y'}
            )
            
            # Train model
            model = Prophet()
            model.fit(prophet_train)
            train_forecast = model.predict(prophet_train)
            train_actual = train_group['final_score'].values
            train_pred   = train_forecast['yhat'].values
        
            train_dev = train_actual - train_pred
        
            train_mean = train_dev.mean()
            train_std  = train_dev.std()
                
            # Predict on test dates
            prophet_test = test_group[['date']].rename(columns={'date': 'ds'})
            forecast = model.predict(prophet_test)
            
            # Actual vs predicted
            actual = test_group['final_score'].values
            predicted = forecast['yhat'].values
            
            test_dev = actual - predicted
            
            # Normalize using training stats
            std = test_dev.std()
            
            if train_std != 0:
                anomaly_score = (test_dev - train_mean) / train_std
            else:
                anomaly_score = np.zeros(len(test_dev))
            
            # Store results
            df.loc[test_group.index, 'prophet_score'] = anomaly_score
            df.loc[test_group.index, 'temporal_anomaly_flag'] = (np.abs(anomaly_score) > 2.5).astype(int)
        
        result = df[df['temporal_anomaly_flag'] == 1][
            ['Heading', 'location', 'date', 'final_score', 'prophet_score']
        ].head(10)
    
          
🌍 Source Discrepancy Detection

          # Train-Test Split
          X = np.vstack(df['embedding'].values)
          
          
          scaler = StandardScaler()
          X = scaler.fit_transform(X)
          y = df['location']
          le = LabelEncoder()
          y_encoded = le.fit_transform(y)
          X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
              X, y_encoded, df.index, test_size=0.2, random_state=42
          )
          # Model Training
          
          model = XGBClassifier(
              n_estimators=250,        # more trees → better learning
              max_depth=6,             # safe depth (avoid overfit)
              learning_rate=0.05,      # slower → better generalization
              subsample=0.8,           # prevent overfitting
              colsample_bytree=0.8,    # feature sampling
              eval_metric='mlogloss',
              random_state=42
          )
          # Balancing the weights of the classes
          
          from sklearn.utils.class_weight import compute_sample_weight
          
          sample_weights = compute_sample_weight(
              class_weight='balanced',
              y=y_train
          )
          
          model.fit(X_train, y_train, sample_weight=sample_weights)
          
          y_pred = model.predict(X_test)
          
          print(classification_report(y_test, y_pred))

The report is as follows:
      
                     precision    recall  f1-score   support
      
                 0       0.53      0.42      0.47        24
                 1       0.86      0.46      0.60        13
                 2       0.76      0.82      0.79        74
                 3       0.35      0.35      0.35        17
                 4       0.62      0.55      0.58        29
                 5       0.46      0.59      0.52        29
                 6       0.73      0.85      0.79        13
                 7       0.75      0.68      0.71        22
      
          accuracy                           0.64       221
         macro avg       0.63      0.59      0.60       221
      weighted avg       0.65      0.64      0.64       221

    # Prediction + Confidence
    
    df['predicted_location'] = le.inverse_transform(model.predict(X))
    df['source_discrepancy'] = df['location'] != df['predicted_location']
    probs = model.predict_proba(X_test)
    
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    threshold = 0.6
    df['confidence'] = np.nan
    df['predicted_location'] = None
    
    df.loc[idx_test, 'confidence'] = confidence
    df.loc[idx_test, 'predicted_location'] = le.inverse_transform(preds)
    final_preds = []
    for i in range(len(preds)):
        if confidence[i] < threshold:
            final_preds.append(-1)  # uncertain
        else:
            final_preds.append(preds[i])
    
    import numpy as np
    
    mask = np.array(final_preds) != -1
    
    from sklearn.metrics import accuracy_score
    
    print("Filtered Accuracy:",
          accuracy_score(y_test[mask], np.array(final_preds)[mask]))
    
    coverage = len(y_test[mask]) / len(y_test)
    print("Coverage:", coverage)

The result is

Filtered Accuracy: 0.7358490566037735

Coverage: 0.7194570135746606

which implies 

- Base Accuracy: 64%

- Weighted F1-score: 0.64

- Filtered Accuracy: 73.6%

- Coverage: 71.9%

- Insight: Model is reliable for majority of predictions while safely flagging uncertain cases.


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

             top = df.sort_values(by='final_score', ascending=False).head(10)
             plt.figure(figsize=(10,6))
             plt.barh(top['Heading'], top['final_score'])
             plt.gca().invert_yaxis()
             plt.title("Top 10 Anomalous News Articles")
             plt.xlabel("Final Anomaly Score")
             plt.tight_layout()
             plt.show()

   <img width="1000" height="600" alt="Top 10 anomalous score after tuning" src="https://github.com/user-attachments/assets/88c1a7b8-488c-4cd1-9567-78d645db8395" />

3. Temporal Trends
          
            plt.figure(figsize=(10,5))
            plt.plot(df['date'], df['final_score'], label='Anomaly Score')
            anomalies = df[df['prophet_score'].abs() > 2.5]
            plt.scatter(anomalies['date'], anomalies['final_score'], marker='x')
            plt.title("Temporal Anomaly Detection")
            plt.xlabel("Date")
            plt.ylabel("Anomaly Score")
            plt.tight_layout()
            plt.show()

   <img width="1000" height="500" alt="Temporal anomaly detection after tuning" src="https://github.com/user-attachments/assets/8d1999cb-a7f3-4124-b7db-91f6ac3d5cdd" />


4. Anomaly Type Distribution

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

        location_anomalies = df.groupby('location')['anomaly_score_norm'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        location_anomalies.head(10).plot(kind='bar')
        plt.title('Top Locations by Average Anomaly Score')
        plt.ylabel('Average Anomaly Score')
        plt.xlabel('Location')
        
        plt.tight_layout()
        plt.show()

<img width="1280" height="612" alt="top location anomaly" src="https://github.com/user-attachments/assets/0365f503-aa00-4918-8161-09b1e613ce59" />







     
    






  


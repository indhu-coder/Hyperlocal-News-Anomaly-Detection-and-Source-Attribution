With the rapid growth of hyperlocal news platforms, ensuring the authenticity and consistency of news content has become increasingly challenging. News articles may exhibit inconsistencies between their reported publication location and their underlying linguistic patterns, sentiment, or thematic content. Such discrepancies can indicate potential misattribution, misleading information, or unusual narrative shifts.

This project aims to develop an **advanced Natural Language Processing (NLP) system** capable of:

* Detecting **anomalous linguistic patterns** in news articles
* Identifying **discrepancies between stated and inferred publication locations**
* Monitoring the **evolution of narratives over time** within hyperlocal news

The system leverages a combination of:

* **Contextual text embeddings**
* **Sentiment analysis**
* **Topic modeling**
* **Named Entity Recognition (NER) for location extraction**
* **Hybrid anomaly detection techniques**
* **Time-series analysis for temporal pattern shifts**

By integrating these components, the system evaluates whether a news article aligns with the expected characteristics of its claimed origin. Articles that significantly deviate from learned patterns are flagged as **potential anomalies**, enabling the detection of misleading, inconsistent, or unusual news events.

---

### 🎯 Objectives

* Extract and normalize **geographical locations** from unstructured text
* Learn **normal linguistic and thematic patterns** for different locations and news types
* Detect **linguistic anomalies** using unsupervised learning techniques
* Identify **source discrepancies** by comparing predicted vs stated locations
* Analyze **temporal trends** to capture sudden narrative shifts
* Provide an interpretable framework for **flagging suspicious news articles**

---

### 💡 Key Insight

The core idea of this project is that:

> *News originating from a specific location tends to follow consistent linguistic, emotional, and thematic patterns. Deviations from these patterns may indicate anomalies or inconsistencies.*

This system leverages that insight to build a robust, multi-dimensional anomaly detection pipeline.

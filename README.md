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

## ☁️ Deployment (AWS EC2 + Docker)

The application is deployed on an **AWS EC2 instance** using Docker for portability and WinSCP for file transfer.

---

### ⚙️ Setup Overview

* **Cloud Platform:** AWS EC2 (t3.micro)
* **Containerization:** Docker
* **File Transfer:** WinSCP
* **App Framework:** Streamlit

---

### 🚀 Deployment Steps

1. **Launch EC2 Instance**

   * Create a Linux-based EC2 instance (e.g., Ubuntu)
   * Configure security group to allow:

     * Port **22** (SSH)
     * Port **8501** (Streamlit)

2. **Connect to EC2**

   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-public-ip
   ```

3. **Install Docker**

   ```bash
   sudo apt update
   sudo apt install docker.io -y
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

4. **Transfer Project Files**

   * Use WinSCP to upload project folder to EC2 instance

5. **Build Docker Image**

   ```bash
   docker build -t news-anomaly-app .
   ```

6. **Run Container**

   ```bash
   docker run -d -p 8501:8501 news-anomaly-app
   ```

7. **Access Application**

   * Open in browser:

     ```
     http://<EC2-Public-IP>:8501
     ```

---

### 🧠 Benefits of This Setup

* **Scalable:** Easily upgrade EC2 instance
* **Portable:** Docker ensures consistent environment
* **Lightweight:** Suitable for low-resource instances (t3.micro)
* **Simple Deployment:** Minimal setup with reproducible steps

---

### 📌 Notes

* Ensure required ports are open in EC2 security groups
* Keep Docker image optimized for faster startup
* Use `--restart always` in production for reliability

---

### 🚀 Conclusion

This deployment setup enables **real-time access** to the anomaly detection system with a lightweight, scalable, and production-ready architecture.


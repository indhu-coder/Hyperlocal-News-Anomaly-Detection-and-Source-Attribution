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

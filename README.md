# CINE-BOT
Autonomous Robot that can detect and follow April-Tags using computer vision.
This project demonstrates how to build an **autonomous robot car** that can detect and follow **AprilTags** using computer vision.  
A webcam mounted on the robot captures live video, detects AprilTags in real-time, and estimates their **3D pose (x, y, z, roll, pitch, yaw)**.  
Based on these coordinates, the system can guide the robot to track the tag and maintain a safe distance automatically.  

---

## ✨ Features
- 🎯 **AprilTag Detection** using [pupil-apriltags](https://pypi.org/project/pupil-apriltags/)  
- 📷 **Live Camera Feed** from a USB or laptop webcam  
- 📐 **Pose Estimation** → X, Y, Z coordinates + Roll, Pitch, Yaw  
- 🛠️ **3D Axes Overlay** (X = red, Y = green, Z = blue)  
- 📊 Logs coordinates in real-time (optional)  
- 🚙 Foundation for **autonomous tracking bots & AR applications**  

---

## 🛠️ Tech Stack
- Python 3.10+  
- OpenCV  
- pupil-apriltags  
- NumPy  

---

## 📦 Installation

Clone the repository:
--bash
git clone https://github.com/your-username/april-tag-bot.git
cd april-tag-bot

## Virtual Environment
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Linux/Mac

## Install Dependencies
pip install -r requirements.txt

## Run the Live Script
python apriltag_live.py

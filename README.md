# 🚗 Automated Number Plate Detection (ANPR)

This project is a **Flask-based web application** for **Automatic Number Plate Recognition (ANPR)**.  
It allows users to upload vehicle videos, automatically detects license plates using **YOLOv8**, and extracts the recognized plates for further processing or analysis.

---

## 🧠 Features
- Upload vehicle videos via a simple web interface.
- Detect and highlight license plates in each frame using YOLOv8.
- Generate processed output videos with bounding boxes and labels.
- Save logs and extracted data for analysis.
- Lightweight and easy to deploy using Flask.

---

## 🏗️ Tech Stack
- **Python 3.9+**
- **Flask** (Web framework)
- **YOLOv8 (Ultralytics)** for object detection
- **OpenCV** for video processing
- **PyTorch** for deep learning inference
- **HTML, CSS, JS** for frontend

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/saisriabhiramnelluri/Automated-Number-Plate-Detection.git
cd Automated-Number-Plate-Detection


### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# or
source venv/bin/activate  # On macOS/Linux

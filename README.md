#  Automated Number Plate Detection (ANPR)

This project is a **Flask-based web application** for **Automatic Number Plate Recognition (ANPR)**.  
It allows users to upload vehicle videos, automatically detects license plates using **YOLOv8**, and extracts the recognized plates for further processing or analysis.

---

##  Features
- Upload vehicle videos via a simple web interface.
- Detect and highlight license plates in each frame using YOLOv8.
- Generate processed output videos with bounding boxes and labels.
- Save logs and extracted data for analysis.
- Lightweight and easy to deploy using Flask.

---

##  Tech Stack
- **Python 3.9+**
- **Flask** (Web framework)
- **YOLOv8 (Ultralytics)** for object detection
- **OpenCV** for video processing
- **PyTorch** for deep learning inference
- **HTML, CSS, JS** for frontend

---

##  Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/saisriabhiramnelluri/Automated-Number-Plate-Detection.git
cd Automated-Number-Plate-Detection
```
### Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Add Model Weights
```bash
license_plate_detector.pt
yolov8n.pt
```
### Run the Flask App
```bash
python app.py
```

Then open your browser and go to:
 http://127.0.0.1:5000/

## How It Works

- Upload an input video (e.g., input_video.mp4).

- The YOLOv8 model detects vehicles and extracts number plates.

- A processed video with bounding boxes is generated.

- You can view or download the output video.

## Project Structure
```bash
anpr_flask/
├── app.py
├── requirements.txt
├── license_plate_detector.pt
├── yolov8n.pt
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│       └── background.jpg
├── templates/
│   ├── index.html
│   └── output_ready.html
├── uploads/              # Input videos 
├── processed_videos/     # Output videos 
└── .gitignore
```


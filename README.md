SafeDrive: AI-Powered Collision Warning System 🚗⚠️
SafeDrive is an advanced Advanced Driver Assistance System (ADAS) leveraging Computer Vision and Deep Learning to enhance road safety. By processing real-time video feeds, the system detects vehicles, tracks their movement, and assesses collision risks through predictive mathematical modeling.

🚀 Key Features
Real-time Object Detection: High-speed vehicle identification (cars, trucks, motorcycles) powered by YOLOv8.

Behavioral State Machine: Implementation of the State Design Pattern to manage vehicle risk levels: Safe (Green), Warning (Yellow), and Danger (Red).

Predictive Risk Analysis (TTC): Dynamic calculation of Time To Collision based on optical expansion rates.

Dynamic Lane Modeling: Use of a Trapezoidal Region of Interest (ROI) to calibrate the perspective and filter out non-threatening vehicles in adjacent lanes.

Visual Stability System: An Anti-Flickering buffer (8/10 frame consensus) ensures a stable and non-intrusive Head-Up Display (HUD) experience.

Data Archiving & OCR: Automated license plate recognition via EasyOCR and historical data persistence using MongoDB.

🛠️ Tech Stack
Language: Python 3.8+

Vision: OpenCV, Ultralytics YOLOv8

Database: MongoDB

Evaluation: CLEAR MOT metrics (MOTA, IDF1)

👥 The Team & Contributions
This project was developed with a modular architecture, dividing technical responsibilities among three core developers:

Giuseppe Montaruli: Core Developer & Computer Vision Specialist. Focused on YOLO integration and the TOOCM tracking logic for occlusion management.

Michele Bernocco (Me): Behavioral Logic & Risk Analysis. Responsible for the State Machine development, TTC mathematical modeling, and dynamic lane calibration.

Pietro Sica: Data Architect & OCR Integration. Focused on MongoDB persistence and asynchronous OCR optimization for plate recognition.


---
### 📑 Technical Documentation
For a detailed analysis of the project architecture, mathematical models, and experimental results, please refer to the:
[Full Project Report (PDF)](./docs/ReportDefinitivo.pdf)
---

💻 Installation & Setup
1. Clone the Repository
Bash

git clone https://github.com/your-username/SafeDrive.git
cd SafeDrive

2. Set Up Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies:

Bash

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies
Bash

pip install -r requirements.txt

4. Database Configuration
Ensure a MongoDB instance is running. You can verify the connection by running:

Bash

python test_db.py

5. Run the Application
Bash

python main.py

📊 Experimental Results
The system underwent rigorous validation using CLEAR MOT metrics:

MOTA (Multi-Object Tracking Accuracy): 0.431.

IDF1 (Identity F1 Score): 0.767, ensuring high identity stability over time.

False Positive Reduction: 45% decrease in erroneous warnings due to the Trapezoidal Lane filter.

Latency: Performance kept under 200ms, enabling true real-time operation.

🔮 Future Roadmap
Implementation of LSTM (Long Short-Term Memory) networks for advanced trajectory prediction.

Deployment on Edge Devices such as NVIDIA Jetson for on-board hardware integration.

Sensor Fusion combining visual data with GPS and LiDAR for 360° environmental awareness.

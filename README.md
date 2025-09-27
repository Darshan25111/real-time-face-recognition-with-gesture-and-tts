***# 🧑‍💻 Real-Time Face Recognition with Gesture-Triggered Capture & Text-to-Speech***
A Python application that performs live face recognition, detects a thumbs-up gesture to trigger automatic image capture, and speaks the recognized person’s name aloud.
It combines ONNX deep-learning models, MediaPipe hand tracking, OpenCV, and gTTS into a single streamlined system.



***#🚀 Key Features***
***Real-Time Face Recognition***
Uses a pre-trained ArcFace (GlintR100) ONNX model to create 512-D facial embeddings and match them against a local database.
***Gesture-Based Capture***
Shows a 5-second countdown and saves a webcam frame when a thumbs-up gesture is detected.
***Text-to-Speech (TTS)***
Speaks the recognized person’s name using Google Text-to-Speech.
***Automatic Database Management***
Generates and caches face embeddings; automatically rebuilds if the dataset changes.
***Smart Cooldown***
Prevents repeated announcements of the same name within a 5-minute window.


***#⚙️ Configuration***
Customize the application by modifying these key variables in main.py or the main script file:

***DATASET_DIR***: Path to the dataset folder containing subfolders of person images.

***DB_PATH***: Path for saving/loading the face embeddings database file (face_db.pkl).

***MODEL_PATH***: Path to the ONNX face recognition model file (glintr100.onnx).

***SAVE_DIR***: Directory where captured webcam frames are saved.

***THRESHOLD***: Similarity threshold to recognize a face (default: 0.60).

***COUNTDOWN_DURATION***: Seconds for the countdown before image capture (default: 5).

***COOLDOWN_TIME***: Cooldown in seconds to avoid repeating name announcements (default: 300).
Adjust these paths and parameters as needed to fit your environment and preferences.


***# 🏗️ Project Structure***
```
face-recognition/
├─ dataset/                # Training images
├─ image/                  # Captured frames are stored here
├─ face_db.pkl             # Auto-generated face embedding database
├─ face_db.pkl.hash        # Hash file to detect dataset changes
├─ glintr100.onnx          # Pretrained ONNX model for face embeddings
└─ main.py                 # Main Python script

```


***# 🖼️ Dataset Preparation***

Create a folder for each person inside dataset/.
Folder name = Person’s Name (used as the recognition label).
Add multiple clear, front-facing images per person.

Example:
text
```
dataset/
├─ Rohit/
│  ├─ 1.jpg
│  ├─ 2.jpg
├─ Anjali/
│  ├─ img1.png
│  ├─ img2.png
```

***#💻 Installation***

1️⃣ Prerequisites
Python 3.8+

2️⃣ Install Dependencies
pip install opencv-python onnxruntime numpy mediapipe gTTS

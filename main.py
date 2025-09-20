import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
import hashlib
import time
from numpy.linalg import norm
import mediapipe as mp
from gtts import gTTS

# Paths
DATASET_DIR = r"C:\Users\OneDrive\Desktop\dataset"
DB_PATH = r"C:\Users\OneDrive\Desktop\new\face_db.pkl"
MODEL_PATH = r"C:\Users\Downloads\antelopev2\antelopev2\glintr100.onnx"
SAVE_DIR = r"C:\Users\OneDrive\Desktop\image"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Haar Cascades
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Constants
THRESHOLD = 0.65
INPUT_SIZE = 112

# ✅ MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# ✅ Compute dataset hash
def compute_dataset_hash(dataset_dir):
    md5 = hashlib.md5()
    for root, dirs, files in os.walk(dataset_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, f)
                md5.update(path.encode())
                with open(path, "rb") as imgf:
                    md5.update(imgf.read())
    return md5.hexdigest()

def check_model_exists(path):
    if not os.path.exists(path):
        print(f"Model file not found at {path}. Please check the path.")
        return False
    return True

def create_session(model_path):
    sess_opts = ort.SessionOptions()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=sess_opts)
    return session

def preprocess_face(bgr_img, target_size=112):
    img = cv2.resize(bgr_img, (target_size, target_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb - 127.5) / 128.0
    inp = np.transpose(img_norm, (2, 0, 1))
    inp = np.expand_dims(inp, 0).astype(np.float32)
    return inp

def get_embedding_from_image(session, img_bgr):
    inp = preprocess_face(img_bgr)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    emb = outputs[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb

def build_database(session, dataset_dir, db_path, rebuild=False):
    dataset_hash_path = db_path + ".hash"
    current_hash = compute_dataset_hash(dataset_dir)
    previous_hash = None
    if os.path.exists(dataset_hash_path):
        with open(dataset_hash_path, "r") as f:
            previous_hash = f.read().strip()

    if os.path.exists(db_path) and not rebuild and previous_hash == current_hash:
        print("✅ Database loaded from cache.")
        with open(db_path, "rb") as f:
            return pickle.load(f)

    print("🔄 Building database...")
    db = {}
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for file in os.listdir(person_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                emb = get_embedding_from_image(session, img)
                embeddings.append(emb)

        if embeddings:
            db[person] = np.mean(embeddings, axis=0)

    with open(db_path, "wb") as f:
        pickle.dump(db, f)
    with open(dataset_hash_path, "w") as f:
        f.write(current_hash)

    print("✅ Database built and saved.")
    return db

# ✅ Detect thumbs-up gesture
def is_thumbs_up(landmarks):
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    if thumb_tip < index_tip and thumb_tip < middle_tip and thumb_tip < ring_tip and thumb_tip < pinky_tip:
        folded_fingers = sum([
            index_tip > landmarks[5].y,
            middle_tip > landmarks[9].y,
            ring_tip > landmarks[13].y,
            pinky_tip > landmarks[17].y
        ])
        return folded_fingers >= 3
    return False

# ✅ TTS Function (Modified to use os.startfile)
def speak_name(name):
    tts = gTTS(text=name, lang='en')
    audio_path = os.path.join(SAVE_DIR, "name.mp3")
    tts.save(audio_path)
    os.startfile(audio_path)  # ✅ Opens default audio player instead of playsound

def recognize_realtime(session, db):
    cap = cv2.VideoCapture(0)
    img_counter = 0
    thumbs_up_start = None
    countdown_active = False
    countdown_start_time = None
    countdown_duration = 5  # seconds
    last_spoken_name = None  # To prevent repeating TTS too frequently
    tts_cooldown = 3  # seconds
    last_spoken_time = 0

    # ✅ Added dictionary for 5-minute cooldown per person
    last_announced = {}
    cooldown_time = 300  # 5 minutes in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Hand detection
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_thumbs_up(hand_landmarks.landmark):
                    if not countdown_active:
                        countdown_active = True
                        countdown_start_time = time.time()
                else:
                    countdown_active = False
                    countdown_start_time = None

        # ✅ Countdown logic
        if countdown_active:
            elapsed = time.time() - countdown_start_time
            remaining = countdown_duration - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f"Capturing in {remaining}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                img_counter += 1
                full_img_path = os.path.join(SAVE_DIR, f"frame_{img_counter}.jpg")
                cv2.imwrite(full_img_path, frame)
                cv2.putText(frame, "Image Captured!", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                countdown_active = False
                countdown_start_time = None

        # ✅ Face Recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emb = get_embedding_from_image(session, face_img)

            best_name = "Unknown"
            best_score = -1.0
            for name, db_emb in db.items():
                score = float(np.dot(emb, db_emb) / (norm(emb) * norm(db_emb)))
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_score < THRESHOLD:
                best_name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{best_name} ({best_score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ✅ Speak detected name (with 5-min cooldown per person)
            current_time = time.time()
            if best_name != "Unknown":
                if best_name not in last_announced or (current_time - last_announced[best_name]) > cooldown_time:
                    print(f"✅ Recognized: {best_name}")
                    speak_name(best_name)
                    last_announced[best_name] = current_time

        cv2.imshow("Real-time Face Recognition + Thumbs Up + Countdown + TTS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not check_model_exists(MODEL_PATH):
        exit(1)
    session = create_session(MODEL_PATH)
    database = build_database(session, DATASET_DIR, DB_PATH)
    recognize_realtime(session, database)


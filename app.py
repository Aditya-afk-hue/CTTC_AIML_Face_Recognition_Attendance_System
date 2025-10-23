# app.py (Final Version with Automatic Stop)

import os
import cv2
import pickle
import numpy as np
import streamlit as st
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global Variables & Constants ---
CASCADE_FILE = "haarcascade_frontalface_default.xml"
MODEL_FILE = "final_model.h5"
DATA_DIR = "data"
IMAGES_DIR = "images"
LABEL_ENCODER_FILE = os.path.join(DATA_DIR, 'label_encoder.p')
DB_FILE = "attendance.db"
CONFIDENCE_THRESHOLD = 75
LOG_INTERVAL_MINUTES = 5

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_attendance(name, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp FROM attendance WHERE name = ? ORDER BY timestamp DESC LIMIT 1', (name,))
    last_entry = cursor.fetchone()
    current_time = datetime.now()
    log_entry = True
    if last_entry:
        last_time = datetime.fromisoformat(last_entry[0])
        if current_time - last_time < timedelta(minutes=LOG_INTERVAL_MINUTES):
            log_entry = False
    if log_entry:
        cursor.execute('INSERT INTO attendance (name, timestamp, confidence) VALUES (?, ?, ?)', (name, current_time, confidence))
        conn.commit()
        st.toast(f"✅ Attendance logged for {name}")
    conn.close()
    return log_entry

def get_attendance_data(start_date=None, end_date=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT name, timestamp, confidence FROM attendance"
    params = []
    if start_date and end_date:
        query += " WHERE DATE(timestamp) BETWEEN ? AND ?"
        params.extend([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
    elif start_date:
        query += " WHERE DATE(timestamp) = ?"
        params.append(start_date.strftime('%Y-%m-%d'))
    query += " ORDER BY timestamp DESC"
    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    if not df.empty:
         df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    return df

def clear_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='attendance'")
    conn.commit()
    conn.close()

init_db()

# --- Helper Functions ---
def ensure_dirs_exist():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def load_face_classifier():
    if not os.path.exists(CASCADE_FILE):
        st.error(f"Error: Cascade file '{CASCADE_FILE}' not found.")
        st.stop()
    return cv2.CascadeClassifier(CASCADE_FILE)

ensure_dirs_exist()
classifier = load_face_classifier()

# --- Streamlit UI ---
st.title("Face Recognition Attendance System ✅")
st.sidebar.title("Menu")
selected = st.sidebar.selectbox("Choose an option", ["Mark Attendance", "View Attendance", "Collect Data", "Train Model"])

# --- UI Section: Collect Data ---
if selected == "Collect Data":
    st.header("Collect Face Data 🧑‍🤝‍🧑")
    name = st.text_input("Enter the person's name:", key="collect_name")

    if not name:
        st.warning("Please enter a name before starting collection.")
        st.stop()
    
    person_dir = os.path.join(IMAGES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    if 'collecting' not in st.session_state:
        st.session_state.collecting = False
    if 'img_count' not in st.session_state:
        st.session_state.img_count = 0
    
    def start_collecting():
        try:
            existing_files = [f for f in os.listdir(person_dir) if f.startswith(name + "_") and f.endswith(".jpg")]
            indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            st.session_state.img_count = max(indices) + 1 if indices else 0
        except Exception as e:
            st.error(f"Error checking existing images: {e}")
            st.session_state.img_count = 0
        st.session_state.collecting = True

    def stop_collecting():
        st.session_state.collecting = False

    st.info("Press 'Start Collecting' to begin. The camera will stop after 100 images or when you press 'Stop'.")
    
    col1, col2 = st.columns(2)
    col1.button("Start Collecting", on_click=start_collecting, type="primary")
    col2.button("Stop Collecting", on_click=stop_collecting)

    stframe = st.empty()

    if st.session_state.get('collecting', False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open video device.")
        else:
            while st.session_state.collecting:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from camera.")
                    break
                
                display_frame = frame.copy()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(gray_frame, 1.3, 5)

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    if st.session_state.img_count < 100:
                        face_frame = frame[y:y+h, x:x+w]
                        file_path = os.path.join(person_dir, f"{name}_{st.session_state.img_count}.jpg")
                        cv2.imwrite(file_path, face_frame)
                        st.session_state.img_count += 1
                    else:
                        st.warning("Reached 100 images. Stopping collection.")
                        stop_collecting()
                
                progress_text = f"Collected: {st.session_state.img_count}/100"
                cv2.putText(display_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                stframe.image(display_frame, channels="BGR")

            cap.release()
            st.success(f"Collection stopped. Total images for '{name}': {st.session_state.img_count}")
            st.rerun()

# --- UI Section: Train Model ---
elif selected == "Train Model":
    # (No changes needed here, code is the same as previous version)
    st.header("Train the Recognition Model 🧠")
    if st.button("Start Training", key="train_button"):
        with st.spinner("Processing images and training model... This may take a while."):
            image_data, labels = [], []
            if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
                 st.error("Image directory is empty. Please collect data first.")
            else:
                st.write(f"Scanning `{IMAGES_DIR}` directory...")
                found_images = False
                for person_name in os.listdir(IMAGES_DIR):
                    person_dir = os.path.join(IMAGES_DIR, person_name)
                    if os.path.isdir(person_dir):
                        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if image_files:
                            st.write(f"- Found {len(image_files)} images for {person_name}")
                            found_images = True
                            for img_file in image_files:
                                try:
                                    img_path = os.path.join(person_dir, img_file)
                                    image = cv2.imread(img_path)
                                    if image is None: continue
                                    image = cv2.resize(image, (100, 100))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                    image_data.append(image)
                                    labels.append(person_name)
                                except Exception as e:
                                    st.warning(f"Error processing {img_path}: {e}")
                if not found_images or not image_data:
                    st.error("No valid images found to train.")
                elif len(set(labels)) < 1:
                     st.error("Training requires images for at least one person.")
                else:
                    images = np.array(image_data, dtype='float32')/255.0
                    images = images.reshape(images.shape[0], 100, 100, 1)
                    label_encoder = LabelEncoder()
                    integer_labels = label_encoder.fit_transform(labels)
                    categorical_labels = to_categorical(integer_labels)
                    st.write(f"Found {len(label_encoder.classes_)} unique individuals: {', '.join(label_encoder.classes_)}")
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(LABEL_ENCODER_FILE, 'wb') as f: pickle.dump(label_encoder, f)
                    st.write(f"Label encoder saved.")
                    if len(images) < 5: st.error(f"Too few images ({len(images)}) to train."); st.stop()
                    X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42, stratify=categorical_labels)
                    num_classes = len(label_encoder.classes_)
                    model = Sequential([
                        Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)), MaxPooling2D((2,2)),
                        Conv2D(64, (3,3), activation='relu'), MaxPooling2D((2,2)),
                        Flatten(), Dense(128, activation='relu'), Dense(num_classes, activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    st.write("Starting model training...")
                    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                    model.save(MODEL_FILE)
                    st.success(f"Model training complete and saved.")
                    st.subheader("Training Performance")
                    perf_df = pd.DataFrame(history.history).rename(columns={'accuracy':'Training Accuracy', 'val_accuracy': 'Validation Accuracy'})
                    st.line_chart(perf_df)
                    st.write(f"Final Validation Accuracy: {perf_df['Validation Accuracy'].iloc[-1]:.2%}")

# --- UI Section: Mark Attendance (Live Recognition) ---
elif selected == "Mark Attendance":
    st.header("Mark Attendance 👋")

    if 'attendance_marked' not in st.session_state: st.session_state.attendance_marked = False
    if 'person_name' not in st.session_state: st.session_state.person_name = ""
    if 'running_recognition' not in st.session_state: st.session_state.running_recognition = False

    if st.session_state.attendance_marked:
        st.success(f"Attendance marked for **{st.session_state.person_name}**!")
        if st.button("Mark Another Attendance"):
            st.session_state.attendance_marked = False
            st.session_state.person_name = ""
            st.session_state.running_recognition = False
            st.rerun()
    else:
        if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
             st.warning("Model not found. Please go to the 'Train Model' tab to train it.")
        else:
            if st.button("Start Camera", type="primary"):
                st.session_state.running_recognition = True
                st.rerun()
            
            stframe = st.empty()

            if st.session_state.running_recognition:
                if st.button("Stop Camera"):
                    st.session_state.running_recognition = False
                    st.rerun()
                try:
                    model = load_model(MODEL_FILE)
                    with open(LABEL_ENCODER_FILE, 'rb') as f:
                        label_encoder = pickle.load(f)
                    
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Could not open video device.")
                    else:
                        st.info("Camera is running... Looking for a face.")
                        last_log_time, log_message = {}, None

                        while st.session_state.running_recognition:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to grab frame."); break
                            
                            img = frame.copy()
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = classifier.detectMultiScale(gray, 1.3, 5)

                            for (x, y, w, h) in faces:
                                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                face = gray[y:y+h, x:x+w]
                                try:
                                    face_resized = cv2.resize(face, (100, 100))
                                    face_processed = face_resized.reshape(1, 100, 100, 1) / 255.0
                                    prediction = model.predict(face_processed, verbose=0)
                                except: continue
                                
                                pred_index = np.argmax(prediction)
                                confidence = np.max(prediction) * 100
                                
                                if confidence >= CONFIDENCE_THRESHOLD:
                                    pred_label = label_encoder.inverse_transform([pred_index])[0]
                                    text, text_color = f"{pred_label} ({confidence:.1f}%)", (0, 255, 0)
                                    
                                    if not log_message: # Only log once per session
                                        if log_attendance(pred_label, confidence):
                                            st.session_state.attendance_marked = True
                                            st.session_state.person_name = pred_label
                                            log_message = f"LOGGED: {pred_label}"
                                            st.session_state.running_recognition = False
                                else:
                                    text, text_color = f"Unknown ({confidence:.1f}%)", (0, 0, 255)
                                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                            if log_message:
                                font, font_scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                                text_size = cv2.getTextSize(log_message, font, font_scale, thickness)[0]
                                text_x = (img.shape[1] - text_size[0]) // 2
                                text_y = (img.shape[0] + text_size[1]) // 2
                                cv2.putText(img, log_message, (text_x, text_y), font, font_scale, color, thickness)
                            
                            stframe.image(img, channels="BGR")

                            if log_message:
                                time.sleep(1.5)

                        cap.release()
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during recognition: {e}")
                    st.session_state.running_recognition = False
            else:
                st.info("Click 'Start Camera' to begin.")

# --- UI Section: View Attendance ---
elif selected == "View Attendance":
    # (No changes needed here, code is the same as previous version)
    st.header("Attendance Records 📊")
    filter_option = st.radio("Filter by:", ("All", "Today", "Date Range"), horizontal=True)
    start_date, end_date = None, None
    today = datetime.now().date()
    if filter_option == "Today":
        start_date, end_date = today, today
    elif filter_option == "Date Range":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start date", today - timedelta(days=7))
        end_date = col2.date_input("End date", today)
        if start_date > end_date: st.warning("Start date cannot be after end date.")
    try:
        attendance_df = get_attendance_data(start_date, end_date)
        if attendance_df.empty:
            st.info("No records found for the selected period.")
        else:
            st.dataframe(
                attendance_df, use_container_width=True,
                column_config={"confidence": st.column_config.NumberColumn("Confidence", format="%.2f%%")},
                hide_index=True
            )
            csv = attendance_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download data as CSV", data=csv, file_name=f'attendance.csv', mime='text/csv')
        st.subheader("Database Management")
        with st.expander("⚠️ Danger Zone"):
            st.warning("This will permanently delete all attendance records.")
            if st.button("Clear All Attendance Records", type="primary"):
                with st.spinner("Clearing database..."): clear_db()
                st.success("All attendance records have been deleted.")
                st.rerun()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
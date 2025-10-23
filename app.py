import os
import cv2
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
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

if selected == "Collect Data":
    st.header("Collect Face Data 🧑‍🤝‍🧑")
    st.warning("Data collection uses your local webcam and is not available on the deployed app. Please run this locally.")

elif selected == "Train Model":
    st.header("Train the Recognition Model 🧠")
    st.warning("Model training is computationally intensive. Please run your `train_model.py` script locally and push the model files to GitHub.")

elif selected == "Mark Attendance":
    st.header("Mark Attendance 👋")
    if 'attendance_marked' not in st.session_state:
        st.session_state.attendance_marked = False
    if 'person_name' not in st.session_state:
        st.session_state.person_name = ""

    if st.session_state.attendance_marked:
        st.success(f"Attendance successfully marked for **{st.session_state.person_name}**!")
        st.info("You can now navigate away or mark another attendance.")
        if st.button("Mark Another Attendance"):
            st.session_state.attendance_marked = False
            st.session_state.person_name = ""
            st.rerun()
    else:
        st.info("Click 'START' to use your browser's webcam for recognition.")
        if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
            st.warning("Model files not found. Please train the model locally and push the files to your GitHub repository.")
        else:
            try:
                model = load_model(MODEL_FILE)
                with open(LABEL_ENCODER_FILE, 'rb') as f:
                    label_encoder = pickle.load(f)
                
                class AttendanceVideoTransformer(VideoTransformerBase):
                    def __init__(self):
                        super().__init__()
                        self.last_log_time = {}
                        self.log_message = None
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")
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
                                last_time = self.last_log_time.get(pred_label)
                                if not self.log_message:
                                    if not last_time or (datetime.now() - last_time) >= timedelta(minutes=LOG_INTERVAL_MINUTES):
                                        if log_attendance(pred_label, confidence):
                                            self.last_log_time[pred_label] = datetime.now()
                                            st.session_state.attendance_marked = True
                                            st.session_state.person_name = pred_label
                                            self.log_message = f"LOGGED: {pred_label}"
                            else:
                                text, text_color = f"Unknown ({confidence:.1f}%)", (0, 0, 255)
                            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                        if self.log_message:
                            font, font_scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                            text_size = cv2.getTextSize(self.log_message, font, font_scale, thickness)[0]
                            text_x = (img.shape[1] - text_size[0]) // 2
                            text_y = (img.shape[0] + text_size[1]) // 2
                            cv2.putText(img, self.log_message, (text_x, text_y), font, font_scale, color, thickness)
                        
                        return img

                webrtc_streamer(
                    key="attendance", mode=WebRtcMode.SENDRECV,
                    video_processor_factory=AttendanceVideoTransformer,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                st.caption("After recognition, click the red 'STOP' button above to finalize.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif selected == "View Attendance":
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
            st.info("No records found.")
        else:
            st.dataframe(attendance_df, use_container_width=True,
                column_config={"confidence": st.column_config.NumberColumn("Confidence", format="%.2f%%")},
                hide_index=True)
            csv = attendance_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download as CSV", data=csv, file_name='attendance.csv', mime='text/csv')
        st.subheader("Database Management")
        with st.expander("⚠️ Danger Zone"):
            st.warning("This will permanently delete all attendance records.")
            if st.button("Clear All Records", type="primary"):
                with st.spinner("Clearing..."): clear_db()
                st.success("All records deleted.")
                st.rerun()
    except Exception as e:
        st.error(f"Error fetching data: {e}")

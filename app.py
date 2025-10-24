import os
import cv2
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io

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
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    return df

def clear_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    init_db()

init_db()

# --- Helper Functions ---
def load_face_classifier():
    if not os.path.exists(CASCADE_FILE):
        st.error(f"Error: Cascade file '{CASCADE_FILE}' not found. Please ensure it's in your GitHub repository.")
        st.stop()
    return cv2.CascadeClassifier(CASCADE_FILE)

classifier = load_face_classifier()

# --- Session State Initialization ---
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'capture_image_flag' not in st.session_state:
    st.session_state.capture_image_flag = False

# --- Streamlit UI ---
st.title("Face Recognition Attendance System ✅")
st.sidebar.title("Menu")
selected = st.sidebar.selectbox("Choose an option", ["Mark Attendance", "View Attendance", "Collect Data", "Train Model"])

# ============================== COLLECT DATA PAGE ==============================
if selected == "Collect Data":
    st.header("Collect Face Data 🧑‍🤝‍🧑")
    st.info("Ensure good lighting and face the camera directly. Capture 100 images.")

    person_name = st.text_input("Enter the name of the person:")

    class DataCollectorTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray, 1.3, 5)

            # Draw rectangle on the main image
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Capture logic
            if st.session_state.get('capture_image_flag', False):
                if len(faces) > 0:
                    # Assuming only one face, take the first one
                    (x, y, w, h) = faces[0]
                    face_img = img[y:y+h, x:x+w]
                    st.session_state.captured_images.append(face_img)
                st.session_state.capture_image_flag = False # Reset flag

            # Display progress on the frame
            progress_text = f"Captured: {len(st.session_state.captured_images)}/100"
            cv2.putText(img, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img

    webrtc_streamer(
        key="data_collector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DataCollectorTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Capture Image", disabled=(len(st.session_state.captured_images) >= 100 or not person_name)):
            if not person_name:
                st.warning("Please enter a name first.")
            else:
                st.session_state.capture_image_flag = True

    with col2:
        if st.session_state.captured_images and st.button("Clear Captured Images"):
            st.session_state.captured_images = []
            st.rerun()

    if st.session_state.captured_images:
        st.write(f"Collected {len(st.session_state.captured_images)} images for **{person_name}**.")
        
        # Display thumbnails of captured images
        cols = st.columns(5)
        for i, image in enumerate(st.session_state.captured_images[-5:]): # Show last 5 images
            cols[i % 5].image(image, channels="BGR", width=120)

    if len(st.session_state.captured_images) >= 100:
        st.success("✅ 100 images collected! You can now download them.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, img_data in enumerate(st.session_state.captured_images):
                is_success, buffer = cv2.imencode(".jpg", img_data)
                if is_success:
                    # Create a file-like object in memory
                    file_object = io.BytesIO(buffer)
                    # Write to zip with the correct directory structure
                    zip_file.writestr(f"{person_name}/{person_name}_{i+1}.jpg", file_object.read())

        st.download_button(
            label="Download Images as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{person_name}_images.zip",
            mime="application/zip",
        )
        st.info("After downloading, run your training scripts locally and upload the new model to GitHub.")

# ============================== TRAIN MODEL PAGE ==============================
elif selected == "Train Model":
    st.header("Train the Recognition Model 🧠")
    st.error("This functionality is not available on the deployed app.")
    st.warning("Model training is computationally intensive and must be run on your local machine. Please use the downloaded images to run your `consolidated_data.py` and `train_model.py` scripts locally.")

# ============================== MARK ATTENDANCE PAGE ==============================
elif selected == "Mark Attendance":
    st.header("Mark Attendance 👋")
    
    if 'attendance_marked' not in st.session_state:
        st.session_state.attendance_marked = {}

    st.info("Position your face in the frame and wait for recognition.")
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
        st.error("Model files not found! Please train the model locally and push the files to your GitHub repository.")
    else:
        try:
            model = load_model(MODEL_FILE, compile=False)
            with open(LABEL_ENCODER_FILE, 'rb') as f:
                label_encoder = pickle.load(f)
            
            class AttendanceVideoTransformer(VideoTransformerBase):
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = classifier.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        face = gray[y:y+h, x:x+w]
                        try:
                            face_resized = cv2.resize(face, (100, 100))
                            face_equalized = cv2.equalizeHist(face_resized)
                            face_processed = face_equalized.reshape(1, 100, 100, 1) / 255.0
                            prediction = model.predict(face_processed, verbose=0)
                            
                            pred_index = np.argmax(prediction)
                            confidence = np.max(prediction) * 100
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                pred_label = label_encoder.inverse_transform([pred_index])[0]
                                text = f"{pred_label} ({confidence:.1f}%)"
                                text_color = (0, 255, 0) # Green for recognized
                                
                                if not st.session_state.attendance_marked.get(pred_label):
                                    if log_attendance(pred_label, confidence):
                                        st.session_state.attendance_marked[pred_label] = True
                            else:
                                text = f"Unknown ({confidence:.1f}%)"
                                text_color = (0, 0, 255) # Red for unknown
                            
                            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        except Exception as e:
                            print(f"Error processing face: {e}")
                            continue
                    return img

            webrtc_streamer(
                key="attendance",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=AttendanceVideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")

# ============================== VIEW ATTENDANCE PAGE ==============================
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

    attendance_df = get_attendance_data(start_date, end_date)
    if attendance_df.empty:
        st.info("No records found for the selected period.")
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
            clear_db()
            st.success("All attendance records have been deleted.")
            st.rerun()

import streamlit as st
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
import time
import av  # Required for streamlit-webrtc
import queue # For passing data between threads
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
from threading import Lock # To make operations thread-safe

# --- TensorFlow and Keras for Model Training ---
# Import only if TensorFlow is installed and needed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow/Keras not found. Model training and recognition features will be disabled.")
    # Define dummy functions/classes if needed to prevent NameErrors later
    Sequential, load_model = None, None
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout = None, None, None, None, None
    to_categorical = None
    train_test_split = None
    LabelEncoder = None


# --- Configuration & Setup ---
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATA_DIR = 'data'
IMG_DIR = 'images'
MODEL_FILE = 'final_model.h5'
IMAGES_PKL = os.path.join(DATA_DIR, 'images.p')
LABELS_PKL = os.path.join(DATA_DIR, 'labels.p')
LE_PKL = os.path.join(DATA_DIR, 'label_encoder.p')
ATTENDANCE_LOG = os.path.join(DATA_DIR, 'attendance.csv')
IMG_SIZE = (100, 100)
CAPTURE_COUNT = 100 # Number of images to capture per person

# Create necessary directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# --- Load Haar Cascade Classifier ---
# This needs to be thread-safe for streamlit-webrtc
classifier_lock = Lock()
classifier = None
try:
    with classifier_lock:
        # Check if file exists before loading
        if not os.path.exists(CASCADE_PATH):
            st.error(f"Haar Cascade file not found at {CASCADE_PATH}. Please upload it.")
            st.stop()
        
        classifier = cv2.CascadeClassifier(CASCADE_PATH)
        
        if classifier.empty():
            st.error(f"Failed to load Haar Cascade from {CASCADE_PATH}. The file might be corrupt or invalid.")
            st.stop()
except Exception as e:
    st.error(f"Error loading Haar Cascade: {e}")
    st.stop()

# --- Helper Functions ---

def preprocess_image(img, target_size=IMG_SIZE):
    """Converts image to grayscale, resizes, and optionally equalizes histogram."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        return resized
    except cv2.error as e:
        # st.warning(f"Image processing error: {e}. Skipping frame.")
        return None

def preprocess_for_model(img):
    """Prepares a single image for model prediction."""
    if img is None:
        return None
    try:
        # Assuming the input 'img' is already grayscale and resized
        img_equalized = cv2.equalizeHist(img) # Apply histogram equalization
        img_reshaped = img_equalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
        img_normalized = img_reshaped / 255.0
        return img_normalized
    except Exception as e:
        st.warning(f"Error preparing image for model: {e}")
        return None

def load_attendance():
    """Loads attendance data from CSV or creates an empty DataFrame."""
    if os.path.exists(ATTENDANCE_LOG):
        try:
            return pd.read_csv(ATTENDANCE_LOG)
        except pd.errors.EmptyDataError:
             return pd.DataFrame(columns=['Name', 'Timestamp']) # File is empty
        except Exception as e:
            st.error(f"Error loading attendance log: {e}")
            return pd.DataFrame(columns=['Name', 'Timestamp'])
    else:
        return pd.DataFrame(columns=['Name', 'Timestamp'])

def save_attendance(df):
    """Saves attendance data to CSV."""
    try:
        df.to_csv(ATTENDANCE_LOG, index=False)
    except Exception as e:
        st.error(f"Error saving attendance log: {e}")

# Use session state to manage attendance dataframe and last marked times
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = load_attendance()
if 'last_marked' not in st.session_state:
    st.session_state.last_marked = {}
if 'attendance_needs_update' not in st.session_state:
    st.session_state.attendance_needs_update = False

# This lock will protect access to the attendance dataframe and last_marked dict
attendance_lock = Lock()

def mark_attendance(name):
    """Adds an attendance entry if the person hasn't been marked recently (e.g., 5 min)."""
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')

    with attendance_lock:
        # Check if marked recently
        last_time = st.session_state.last_marked.get(name)
        if last_time and (now - last_time).total_seconds() < 300: # 5 minute cooldown
            return False # Not marked

        # Add new entry
        new_entry = pd.DataFrame({'Name': [name], 'Timestamp': [timestamp_str]})
        st.session_state.attendance_df = pd.concat([st.session_state.attendance_df, new_entry], ignore_index=True)
        st.session_state.last_marked[name] = now # Update last marked time
        
        save_attendance(st.session_state.attendance_df)
        st.session_state.attendance_needs_update = True # Flag for main thread to update UI
    
    st.toast(f"Attendance marked for {name} at {timestamp_str}") # Use toast for less intrusive notice
    return True # Marked


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Face Recognition Attendance")
st.title("Face Recognition Attendance System")

# --- Sidebar for Mode Selection ---
st.sidebar.title("Mode")
mode = st.sidebar.radio("Select Operation:", ["Home", "Collect Data", "Consolidate Data", "Train Model", "Recognize & Mark Attendance", "View Attendance"])

if mode == "Home":
    st.header("Welcome!")
    st.write("""
        Select a mode from the sidebar to:
        1.  **Collect Data:** Register faces of new individuals using your browser's webcam.
        2.  **Consolidate Data:** Process collected images for training.
        3.  **Train Model:** Train the face recognition model (requires collected data).
        4.  **Recognize & Mark Attendance:** Start your camera to recognize faces and log attendance.
        5.  **View Attendance:** View and clear the attendance log.
    """)
    st.info(f"Make sure `{CASCADE_PATH}` is in the same directory as this script.")
    st.info(f"This app uses `streamlit-webrtc` to access your browser's camera. Please grant camera permissions when prompted.")
    if TENSORFLOW_AVAILABLE:
        st.success("TensorFlow/Keras found.")
    else:
        st.error("TensorFlow/Keras is required for model training and recognition.")

# --- Mode: Collect Data ---
elif mode == "Collect Data":
    st.header("Collect Face Data")
    name = st.text_input("Enter Person's Name:", key="collect_name").strip().lower()

    # We use session state to track the count across reruns
    if 'capture_count' not in st.session_state:
        st.session_state.capture_count = 0

    capture_lock = Lock()
    
    class VideoCollector(VideoTransformerBase):
        def __init__(self, person_name, capture_limit):
            self.person_name = person_name
            self.capture_limit = capture_limit
            # Use session state for the counter
            st.session_state.capture_count = 0
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            with capture_lock:
                current_count = st.session_state.capture_count
            
            if current_count >= self.capture_limit:
                # Once limit is reached, just return the frame
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, f"Capture Complete: {current_count}/{self.capture_limit}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = []
            with classifier_lock:
                if not classifier.empty():
                    faces = classifier.detectMultiScale(gray, 1.3, 5)
                else:
                    # This should not happen based on startup check, but as a safeguard
                    print("Classifier not loaded in transform thread.")


            found_face = False
            for (x, y, w, h) in faces:
                # Draw rectangle on the *original* color image
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if current_count < self.capture_limit:
                    face_roi = img[y:y+h, x:x+w]
                    filename = os.path.join(IMG_DIR, f"{self.person_name}_{current_count}.jpg")
                    try:
                        cv2.imwrite(filename, face_roi)
                        with capture_lock:
                            st.session_state.capture_count += 1
                            current_count = st.session_state.capture_count
                    except Exception as e:
                        print(f"Error saving image: {e}") # Log to console
                
                found_face = True
                break # Only capture one face per frame
            
            # Add count to the frame
            cv2.putText(img, f"Captured: {current_count}/{self.capture_limit}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    if not name:
        st.warning("Please enter a name before starting.")
    else:
        st.info(f"Preparing to capture {CAPTURE_COUNT} images for '{name}'.")
        st.info("The video stream will start. Allow camera access in your browser.")
        
        webrtc_ctx = webrtc_streamer(
            key="collect",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: VideoCollector(name, CAPTURE_COUNT),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        while webrtc_ctx.state.playing:
            with capture_lock:
                current_count = st.session_state.capture_count
            
            progress = current_count / CAPTURE_COUNT if CAPTURE_COUNT > 0 else 0
            progress_bar.progress(progress)
            progress_text.text(f"Captured: {current_count}/{CAPTURE_COUNT}")
            
            if current_count >= CAPTURE_COUNT:
                progress_bar.progress(1.0)
                progress_text.success(f"Capture complete for {name}!")
                break # Stop the status update loop
            
            time.sleep(0.1) # Poll for updates

        st.caption("Press the 'STOP' button on the video stream when done.")


# --- Mode: Consolidate Data ---
elif mode == "Consolidate Data":
    st.header("Consolidate Image Data")
    if st.button("Start Consolidation", key="consolidate_btn"):
        image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
        if not image_files:
            st.warning("No images found in the 'images' directory. Please collect data first.")
        else:
            st.info(f"Found {len(image_files)} images. Processing...")
            image_data = []
            labels = []
            processed_count = 0
            consolidation_progress = st.progress(0.0)

            for i, filename in enumerate(image_files):
                try:
                    image_path = os.path.join(IMG_DIR, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        st.warning(f"Could not read image: {filename}. Skipping.")
                        continue

                    processed_image = preprocess_image(image, target_size=IMG_SIZE)
                    if processed_image is None:
                        st.warning(f"Could not process image: {filename}. Skipping.")
                        continue

                    label = filename.split("_")[0] # Extract label (name) from filename

                    image_data.append(processed_image)
                    labels.append(label)
                    processed_count += 1
                except Exception as e:
                    st.warning(f"Error processing {filename}: {e}")

                # Update progress bar
                consolidation_progress.progress((i + 1) / len(image_files))

            if not image_data or not labels:
                 st.error("No images could be processed successfully.")
            else:
                image_data_np = np.array(image_data)
                labels_np = np.array(labels)

                st.write(f"Data shape: {image_data_np.shape}")
                st.write(f"Labels shape: {labels_np.shape}")
                unique_labels = np.unique(labels_np)
                st.write(f"Unique labels found: {unique_labels} ({len(unique_labels)} classes)")


                if len(unique_labels) < 2:
                    st.warning("Warning: Only one person's data found. The model needs at least two people to train effectively.")

                try:
                    with open(IMAGES_PKL, 'wb') as f:
                        pickle.dump(image_data_np, f)
                    with open(LABELS_PKL, 'wb') as f:
                        pickle.dump(labels_np, f)
                    st.success(f"Consolidated data saved to '{IMAGES_PKL}' and '{LABELS_PKL}'.")
                    st.success(f"Successfully processed {processed_count} images.")
                except Exception as e:
                    st.error(f"Error saving pickle files: {e}")

# --- Mode: Train Model ---
elif mode == "Train Model":
    st.header("Train Face Recognition Model")
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow/Keras is required for model training. Please install it.")
        st.stop()

    if not os.path.exists(IMAGES_PKL) or not os.path.exists(LABELS_PKL):
        st.warning("Pickle files ('images.p', 'labels.p') not found. Please consolidate data first.")
    else:
        if st.button("Start Training", key="train_btn"):
            try:
                st.info("Loading consolidated data...")
                with open(IMAGES_PKL, 'rb') as f:
                    images = pickle.load(f)
                with open(LABELS_PKL, 'rb') as f:
                    labels = pickle.load(f)

                if len(np.unique(labels)) < 2:
                    st.error("Need data for at least two different people to train the model.")
                else:
                    st.info(f"Data loaded: {images.shape[0]} images, {len(np.unique(labels))} classes.")
                    st.write(f"Classes: {np.unique(labels)}")

                    # Preprocess labels
                    le = LabelEncoder()
                    labels_encoded = le.fit_transform(labels)
                    labels_categorical = to_categorical(labels_encoded)
                    num_classes = labels_categorical.shape[1]

                    # Save the label encoder
                    with open(LE_PKL, 'wb') as f:
                        pickle.dump(le, f)
                    st.info(f"Label encoder saved to {LE_PKL}")

                    # Preprocess images for CNN
                    images_processed = images.reshape(images.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1)
                    images_normalized = images_processed / 255.0

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        images_normalized, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
                    )

                    st.info(f"Training data shape: {X_train.shape}")
                    st.info(f"Test data shape: {X_test.shape}")

                    # Define a simple CNN model
                    model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
                        MaxPooling2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(num_classes, activation='softmax')
                    ])

                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    # Print summary to a string and display in Streamlit
                    stringlist = []
                    model.summary(print_fn=lambda x: stringlist.append(x))
                    st.text("\n".join(stringlist))

                    st.info("Training model...")
                    
                    training_status = st.empty()
                    training_status.write("Starting training...")
                    
                    # Use a Keras callback to update streamlit in intervals
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            logs = logs or {}
                            acc = logs.get('accuracy', 0)
                            val_acc = logs.get('val_accuracy', 0)
                            loss = logs.get('loss', 0)
                            val_loss = logs.get('val_loss', 0)
                            training_status.write(
                                f"Epoch {epoch+1}/{self.params['epochs']} - "
                                f"Loss: {loss:.4f}, Acc: {acc:.4f} - "
                                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                            )

                    history = model.fit(X_train, y_train, 
                                        epochs=20, 
                                        batch_size=32, 
                                        validation_data=(X_test, y_test), 
                                        callbacks=[StreamlitCallback()],
                                        verbose=0) 
                    
                    training_status.success("Training finished!")

                    # Evaluate model
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"Final Test Loss: {loss:.4f}")
                    st.write(f"Final Test Accuracy: {accuracy:.4f}")

                    # Plot training history (optional)
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.plot(history.history['accuracy'], label='Train Accuracy')
                        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        st.pyplot(fig)
                    except ImportError:
                        st.info("Matplotlib not found, skipping accuracy plot.")

                    # Save model
                    model.save(MODEL_FILE)
                    st.success(f"Model trained and saved as '{MODEL_FILE}'.")

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

# --- Mode: Recognize & Mark Attendance ---
elif mode == "Recognize & Mark Attendance":
    st.header("Recognize Face & Mark Attendance")

    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow/Keras is required for recognition. Please install it.")
        st.stop()

    # --- Load Model and Label Encoder ---
    model = None
    label_encoder = None
    try:
        if os.path.exists(MODEL_FILE):
            model = load_model(MODEL_FILE)
        else:
            st.warning(f"Model file '{MODEL_FILE}' not found. Please train the model first.")

        if os.path.exists(LE_PKL):
            with open(LE_PKL, 'rb') as f:
                label_encoder = pickle.load(f)
            st.info(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
        else:
            st.warning(f"Label encoder '{LE_PKL}' not found. Please train the model first.")

    except Exception as e:
        st.error(f"Error loading model or label encoder: {e}")
        model = None # Ensure model is None if loading failed

    # --- Attendance Log Display ---
    st.subheader("Today's Attendance")
    
    # We need a placeholder to update the dataframe
    attendance_placeholder = st.empty()
    
    def display_attendance():
        today_str = datetime.now().strftime('%Y-%m-%d')
        with attendance_lock:
            # Reload from disk to ensure consistency
            current_attendance_df = load_attendance()
            st.session_state.attendance_df = current_attendance_df
            
            df_to_display = current_attendance_df[
                pd.to_datetime(current_attendance_df['Timestamp']).dt.strftime('%Y-%m-%d') == today_str
            ]
        attendance_placeholder.dataframe(df_to_display, use_container_width=True)

    display_attendance() # Initial display
    
    # --- Result Queue ---
    # This queue will hold recognized names from the video thread
    result_queue = queue.Queue()

    # --- Video Transformer Class ---
    class VideoRecognizer(VideoTransformerBase):
        def __init__(self, model, le, queue_out):
            self.model = model
            self.label_encoder = le
            self.queue_out = queue_out
            self.confidence_threshold = 0.70 # 70% confidence

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            if self.model is None or self.label_encoder is None:
                return frame # Pass through if model not loaded
            
            img = frame.to_ndarray(format="bgr24")
            img_rgb = img # Keep color for display
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = []
            with classifier_lock:
                if not classifier.empty():
                    faces = classifier.detectMultiScale(img_gray, 1.3, 5)
                else:
                    print("Classifier not loaded in transform thread.")

            for (x, y, w, h) in faces:
                face_roi_gray = img_gray[y:y+h, x:x+w]
                
                # Ensure ROI is valid
                if face_roi_gray.size == 0:
                    continue
                    
                face_roi_resized = cv2.resize(face_roi_gray, IMG_SIZE)

                processed_face = preprocess_for_model(face_roi_resized)

                if processed_face is not None:
                    try:
                        # Predict
                        prediction = self.model.predict(processed_face, verbose=0)
                        confidence = np.max(prediction)
                        predicted_class_index = np.argmax(prediction)

                        if confidence > self.confidence_threshold:
                            predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
                            display_text = f"{predicted_label} ({confidence*100:.1f}%)"
                            color = (0, 255, 0) # Green

                            # Put the recognized name in the queue for the main thread
                            self.queue_out.put(predicted_label)
                        else:
                            display_text = f"Unknown ({confidence*100:.1f}%)"
                            color = (0, 0, 255) # Red

                        # Draw rectangle and text
                        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img_rgb, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"Error during prediction or drawing: {e}") # Log to console
            
            return av.VideoFrame.from_ndarray(img_rgb, format="bgr24")

    # --- Start Video Stream ---
    if model and label_encoder:
        st.info("Allow camera access in your browser. Recognition is active.")
        webrtc_ctx = webrtc_streamer(
            key="recognize",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: VideoRecognizer(model, label_encoder, result_queue),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Main loop to check queue and update attendance
        if webrtc_ctx.state.playing:
            while webrtc_ctx.state.playing:
                if st.session_state.attendance_needs_update:
                    display_attendance()
                    with attendance_lock:
                        st.session_state.attendance_needs_update = False
                
                try:
                    # Check for new names in the queue
                    name = result_queue.get(timeout=0.1) # Poll for 100ms
                    marked = mark_attendance(name)
                    
                    if marked:
                        # The flag will be set inside mark_attendance
                        pass

                except queue.Empty:
                    pass # No new name, just loop and check flag
                
                # A short sleep to prevent a busy-loop
                time.sleep(0.1) 
        else:
            # When stream stops, check for any final updates
            if st.session_state.attendance_needs_update:
                display_attendance()
                with attendance_lock:
                    st.session_state.attendance_needs_update = False
            
    else:
        st.error("Cannot start recognition. Model or Label Encoder not loaded.")
        st.write("Please go to 'Train Model' and train a model first.")

# --- Mode: View Attendance ---
elif mode == "View Attendance":
    st.header("Full Attendance Log")
    
    # Button to refresh data
    if st.button("Refresh Log"):
        st.session_state.attendance_df = load_attendance()
    
    st.dataframe(st.session_state.attendance_df, use_container_width=True)

    if st.button("Clear Full Attendance Log", key="clear_attendance_btn", type="primary"):
        if os.path.exists(ATTENDANCE_LOG):
            try:
                os.remove(ATTENDANCE_LOG)
                with attendance_lock:
                    st.session_state.attendance_df = load_attendance() # Reload empty df
                    st.session_state.last_marked = {} # Reset cooldowns
                st.success("Attendance log cleared.")
                time.sleep(1) # Give time to see message
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear log: {e}")
        else:
            st.info("Attendance log is already empty.")

else:
    st.error("Invalid mode selected.")


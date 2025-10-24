import streamlit as st
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
import time

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
ATTENDANCE_LOG = os.path.join(DATA_DIR, 'attendance.csv')
IMG_SIZE = (100, 100)
CAPTURE_COUNT = 100 # Number of images to capture per person

# Create necessary directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Load Haar Cascade Classifier
try:
    classifier = cv2.CascadeClassifier(CASCADE_PATH)
    if classifier.empty():
        st.error(f"Failed to load Haar Cascade from {CASCADE_PATH}. Make sure the file exists.")
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
        # Optional: Histogram Equalization (can sometimes improve contrast)
        # equalized = cv2.equalizeHist(resized)
        # return equalized
        return resized
    except cv2.error as e:
        st.warning(f"Image processing error: {e}. Skipping frame.")
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

def mark_attendance(name, attendance_df):
    """Adds an attendance entry if the person hasn't been marked recently."""
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')

    # Check if already marked today (simple check, could be more sophisticated)
    if name in attendance_df['Name'].values:
        last_entry_time_str = attendance_df[attendance_df['Name'] == name]['Timestamp'].iloc[-1]
        last_entry_date = datetime.strptime(last_entry_time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        if last_entry_date == today_str:
            # st.info(f"{name} already marked today at {last_entry_time_str}.")
            return attendance_df # No change

    # Add new entry using pd.concat
    new_entry = pd.DataFrame({'Name': [name], 'Timestamp': [timestamp_str]})
    attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
    st.success(f"Attendance marked for {name} at {timestamp_str}")
    save_attendance(attendance_df)
    return attendance_df


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Face Recognition Attendance")
st.title("Face Recognition Attendance System")

# --- Sidebar for Mode Selection ---
st.sidebar.title("Mode")
mode = st.sidebar.radio("Select Operation:", ["Home", "Collect Data", "Consolidate Data", "Train Model", "Recognize & Mark Attendance"])

if mode == "Home":
    st.header("Welcome!")
    st.write("""
        Select a mode from the sidebar to:
        1.  **Collect Data:** Register faces of new individuals.
        2.  **Consolidate Data:** Process collected images for training.
        3.  **Train Model:** Train the face recognition model (requires collected data).
        4.  **Recognize & Mark Attendance:** Start the camera to recognize faces and log attendance.
    """)
    st.info(f"Make sure `{CASCADE_PATH}` is in the same directory as this script.")
    if TENSORFLOW_AVAILABLE:
        st.success("TensorFlow/Keras found.")
    else:
        st.error("TensorFlow/Keras is required for model training and recognition.")

# --- Mode: Collect Data ---
elif mode == "Collect Data":
    st.header("Collect Face Data")
    name = st.text_input("Enter Person's Name:", key="collect_name").strip().lower()
    url_input = st.text_input("Enter IP Camera URL (or 0 for webcam):", "0", key="collect_url")
    start_capture = st.button("Start Capture", key="start_capture_btn")

    if 'capture_active' not in st.session_state:
        st.session_state.capture_active = False
    if 'captured_count' not in st.session_state:
        st.session_state.captured_count = 0
    if 'face_data_buffer' not in st.session_state:
        st.session_state.face_data_buffer = []

    if start_capture and not name:
        st.warning("Please enter a name.")
    elif start_capture and name:
        st.session_state.capture_active = True
        st.session_state.captured_count = 0
        st.session_state.face_data_buffer = []
        st.info(f"Starting capture for {name}. Press 'Stop Capture' below the video feed.")

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()

    if st.session_state.capture_active:
        stop_capture = st.button("Stop Capture", key="stop_capture_btn")
        if stop_capture:
            st.session_state.capture_active = False
            st.warning("Capture stopped.")
            # Save any remaining buffered data if needed (optional)
            if st.session_state.face_data_buffer:
                 st.write(f"Saving {len(st.session_state.face_data_buffer)} buffered images...")
                 for i, face_img in enumerate(st.session_state.face_data_buffer):
                     filename = os.path.join(IMG_DIR, f"{name}_{st.session_state.captured_count + i}.jpg")
                     cv2.imwrite(filename, face_img)
                 st.session_state.captured_count += len(st.session_state.face_data_buffer)
                 st.session_state.face_data_buffer = [] # Clear buffer
                 progress_bar.progress(1.0)
                 progress_text.text(f"Capture complete: {st.session_state.captured_count}/{CAPTURE_COUNT}")


        # Determine video source
        try:
            video_source = int(url_input) if url_input.isdigit() else url_input
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(f"Error: Could not open video source '{url_input}'. Check the URL or camera index.")
                st.session_state.capture_active = False
        except Exception as e:
            st.error(f"Error initializing video capture: {e}")
            st.session_state.capture_active = False

        while st.session_state.capture_active and 'cap' in locals() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame. Stopping capture.")
                st.session_state.capture_active = False
                break

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = classifier.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)

            capture_this_frame = False
            if len(faces) > 0:
                # Assuming only one face is needed per frame for collection
                x, y, w, h = faces[0]
                face_frame = frame[y:y+h, x:x+w]
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw on RGB frame

                if st.session_state.captured_count < CAPTURE_COUNT:
                     # Add to buffer instead of writing immediately
                    st.session_state.face_data_buffer.append(face_frame)
                    capture_this_frame = True
                    # Optional: Save in batches to reduce I/O frequency
                    if len(st.session_state.face_data_buffer) >= 10: # Save every 10 frames
                         st.write(f"Saving buffer ({len(st.session_state.face_data_buffer)} images)...")
                         for i, face_img in enumerate(st.session_state.face_data_buffer):
                             filename = os.path.join(IMG_DIR, f"{name}_{st.session_state.captured_count + i}.jpg")
                             cv2.imwrite(filename, face_img)
                         st.session_state.captured_count += len(st.session_state.face_data_buffer)
                         st.session_state.face_data_buffer = [] # Clear buffer

            # Update progress
            progress = st.session_state.captured_count / CAPTURE_COUNT
            progress_bar.progress(progress)
            progress_text.text(f"Captured: {st.session_state.captured_count}/{CAPTURE_COUNT}")

            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if st.session_state.captured_count >= CAPTURE_COUNT:
                st.success(f"Successfully captured {CAPTURE_COUNT} images for {name}.")
                st.session_state.capture_active = False

            # Add a small delay to prevent Streamlit from overwhelming the browser
            time.sleep(0.01)

        if 'cap' in locals():
            cap.release()
            cv2.destroyAllWindows() # Just in case

    elif not st.session_state.capture_active and st.session_state.captured_count > 0:
        st.info(f"Capture finished. {st.session_state.captured_count} images saved for {name}.")
        st.session_state.captured_count = 0 # Reset for next capture

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
                st.write(f"Unique labels found: {np.unique(labels_np)}")

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
                    le_path = os.path.join(DATA_DIR, 'label_encoder.p')
                    with open(le_path, 'wb') as f:
                        pickle.dump(le, f)
                    st.info(f"Label encoder saved to {le_path}")


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
                    model.summary(print_fn=st.text) # Display model summary in Streamlit

                    st.info("Training model...")
                    # Use st.empty to show progress dynamically (optional, requires callbacks)
                    training_status = st.empty()
                    training_status.write("Starting training...")
                    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0) # verbose=0 for less console output
                    training_status.success("Training finished!")

                    # Evaluate model
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"Test Loss: {loss:.4f}")
                    st.write(f"Test Accuracy: {accuracy:.4f}")

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
            st.info(f"Recognition model '{MODEL_FILE}' loaded.")
        else:
            st.warning(f"Model file '{MODEL_FILE}' not found. Please train the model first.")

        le_path = os.path.join(DATA_DIR, 'label_encoder.p')
        if os.path.exists(le_path):
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
            st.info(f"Label encoder loaded. Classes: {label_encoder.classes_}")
        else:
            st.warning("Label encoder not found. Please train the model first.")

    except Exception as e:
        st.error(f"Error loading model or label encoder: {e}")
        model = None # Ensure model is None if loading failed

    # --- Attendance Log ---
    attendance_df = load_attendance()
    st.subheader("Attendance Log")
    st.dataframe(attendance_df, use_container_width=True)

    # --- Real-time Recognition ---
    url_input_rec = st.text_input("Enter IP Camera URL (or 0 for webcam):", "0", key="rec_url")
    start_rec = st.button("Start Recognition", key="start_rec_btn", disabled=(model is None or label_encoder is None))

    if 'rec_active' not in st.session_state:
        st.session_state.rec_active = False

    if start_rec:
        st.session_state.rec_active = True
        st.info("Starting recognition. Press 'Stop Recognition' below the video feed.")

    rec_frame_placeholder = st.empty()

    if st.session_state.rec_active:
        stop_rec = st.button("Stop Recognition", key="stop_rec_btn")
        if stop_rec:
            st.session_state.rec_active = False
            st.warning("Recognition stopped.")

        try:
            video_source_rec = int(url_input_rec) if url_input_rec.isdigit() else url_input_rec
            cap_rec = cv2.VideoCapture(video_source_rec)
            if not cap_rec.isOpened():
                st.error(f"Error: Could not open video source '{url_input_rec}'.")
                st.session_state.rec_active = False
        except Exception as e:
            st.error(f"Error initializing video capture: {e}")
            st.session_state.rec_active = False

        while st.session_state.rec_active and 'cap_rec' in locals() and cap_rec.isOpened():
            ret, frame = cap_rec.read()
            if not ret:
                st.warning("Failed to grab frame. Stopping recognition.")
                st.session_state.rec_active = False
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For display

            faces = classifier.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi_gray = frame_gray[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi_gray, IMG_SIZE)

                # Preprocess for model
                processed_face = preprocess_for_model(face_roi_resized)

                if processed_face is not None and model is not None and label_encoder is not None:
                    # Predict
                    prediction = model.predict(processed_face, verbose=0) # verbose=0 hides prediction progress bar
                    confidence = np.max(prediction)
                    predicted_class_index = np.argmax(prediction)

                    # Set a confidence threshold (e.g., 70%)
                    confidence_threshold = 0.70

                    if confidence > confidence_threshold:
                        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
                        display_text = f"{predicted_label} ({confidence*100:.1f}%)"
                        color = (0, 255, 0) # Green for confident prediction

                        # Mark attendance (function handles duplicates)
                        attendance_df = mark_attendance(predicted_label, attendance_df)
                        # Refresh displayed dataframe - this causes a flicker, better to update less often or differently
                        # st.dataframe(attendance_df, use_container_width=True) # Re-display updated df

                    else:
                        predicted_label = "Unknown"
                        display_text = f"Unknown ({confidence*100:.1f}%)"
                        color = (0, 0, 255) # Red for unknown/low confidence

                    # Draw rectangle and text on the RGB frame
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame_rgb, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                else:
                    # Draw basic rectangle if processing failed
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)


            # Display the frame in Streamlit
            rec_frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Optional: Short delay
            time.sleep(0.01)

        if 'cap_rec' in locals():
            cap_rec.release()
            cv2.destroyAllWindows()

        # Update the dataframe display one last time after stopping
        if not st.session_state.rec_active:
             st.dataframe(attendance_df, use_container_width=True)


else:
    st.error("Invalid mode selected.")

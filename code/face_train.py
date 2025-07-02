import os
import cv2
import numpy as np
import pickle
import datetime

# --- Configuration Parameters ---
# Path to the Haar Cascade classifier XML file for face detection.
# cv2.data.haascades points to the 'data' folder within the OpenCV installation.
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'

# Root directory name containing face images.
# Ensure your images are structured like: IMAGE_DIR_NAME/person_name/image.jpg
IMAGE_DIR_NAME = "dataset"

# Output filenames for the trained model and label mapping.
TRAINER_FILE = "mytrainer.xml"
LABELS_FILE = "label.pickle"

# --- Logging Function ---
def log_message(level, message):
    """
    Custom logging function.
    :param level: Log level (e.g., "INFO", "WARNING", "ERROR")
    :param message: Log content
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# --- Main Script Start ---
log_message("INFO", "Face recognition training script started.")

# --- Initialize Variables ---
current_id = 0
label_ids = {}  # Dictionary to map person names to numerical IDs
x_train = []    # List to store cropped face images (NumPy arrays)
y_labels = []   # List to store corresponding IDs for each face image

# Get the absolute path of the current script and construct the dataset directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, IMAGE_DIR_NAME)

log_message("INFO", f"Checking dataset directory: '{image_dir}'")
# Check if the dataset directory exists.
if not os.path.exists(image_dir):
    log_message("ERROR", f"Dataset directory '{image_dir}' not found. Please ensure it exists and contains face images.")
    log_message("INFO", "Example directory structure:")
    log_message("INFO", f"{BASE_DIR}/")
    log_message("INFO", f"{os.path.basename(__file__)}")
    log_message("INFO", f"{IMAGE_DIR_NAME}/")
    log_message("INFO", f"person_A/")
    exit()
else:
    log_message("INFO", f"Dataset directory '{image_dir}' found.")

# Load the face detector.
log_message("INFO", f"Attempting to load Haar Cascade file: '{FACE_CASCADE_PATH}'")
classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if classifier.empty():
    log_message("ERROR", f"Failed to load Haar Cascade file. Please check the path or file integrity.")
    exit()
else:
    log_message("INFO", "Haar Cascade file loaded successfully.")

# Initialize the LBPH Face Recognizer.
recognizer = cv2.face.LBPHFaceRecognizer_create()
log_message("INFO", "LBPH Face Recognizer initialized.")

log_message("INFO", f"Starting to scan directory '{image_dir}' and collect face data...")
total_images_processed = 0
total_faces_extracted = 0

# Iterate through all files and folders within the dataset directory.
for root, dirs, files in os.walk(image_dir):
    # Extract the current directory name as the face label (e.g., 'zhangsan').
    label = os.path.basename(root)

    # Filter out empty directory names and hidden files (e.g., .DS_Store).
    if label == IMAGE_DIR_NAME or label.startswith('.'):
        continue

    # If this label (person's name) is new, assign a unique ID.
    if label not in label_ids:
        label_ids[label] = current_id
        current_id += 1

    id_ = label_ids[label] # Get the numerical ID for the current person's name.

    log_message("INFO", f"Processing label: '{label}' (ID: {id_})")
    images_in_current_label = 0
    faces_in_current_label = 0

    for file in files:
        # Process only common image formats.
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            total_images_processed += 1
            images_in_current_label += 1

            log_message("INFO", f"  - Reading image: '{path}'")
            # Read the image.
            image = cv2.imread(path)

            if image is None:
                log_message("WARNING", f"  - Failed to read image '{path}'. Skipping this file.")
                continue

            # Convert the image to grayscale (face detection often works better on grayscale).
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image to a NumPy array with uint8 data type.
            image_array = np.array(gray, "uint8")

            # Detect faces in the grayscale image.
            # scaleFactor: How much the image size is reduced at each image scale (1.2 means 20% reduction).
            # minNeighbors: Minimum number of neighboring rectangles to retain for a face (higher value reduces false positives).
            # minSize: Minimum possible object size (here, 50x50 pixels); smaller faces are ignored.
            faces = classifier.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 0:
                log_message("WARNING", f"  - No faces detected in image '{file}'.")
            else:
                log_message("INFO", f"  - Detected {len(faces)} face(s) in image '{file}'.")

            # Iterate through detected faces.
            for (x, y, w, h) in faces:
                # Crop the Region Of Interest (ROI) which is the face.
                roi = image_array[y:y+h, x:x+w]

                # Add the face ROI and its corresponding ID to the training dataset.
                x_train.append(roi)
                y_labels.append(id_)
                total_faces_extracted += 1
                faces_in_current_label += 1
                # log_message("DEBUG", f"    - Successfully extracted face ROI from '{file}'.") # Uncomment for more detailed debug info

    log_message("INFO", f"Label '{label}' processing completed. Processed {images_in_current_label} images, extracted {faces_in_current_label} face(s).")

log_message("INFO", "\nData collection completed.")
log_message("INFO", f"Total images processed: {total_images_processed}.")
log_message("INFO", f"Total training samples (faces) collected: {len(x_train)}.")
log_message("INFO", f"Total unique labels (individuals) identified: {len(label_ids)}.")

# Check if there's enough training data.
if len(x_train) < 2:
    log_message("ERROR", "Insufficient training samples. At least 2 samples are required to train the model.")
    log_message("ERROR", "Please ensure your 'dataset' directory contains images of multiple individuals with detectable faces.")
    log_message("INFO", "Script exiting, no model file generated.")
    exit()

# Save the label mapping to a file.
log_message("INFO", f"Saving label mapping to '{LABELS_FILE}'...")
try:
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_ids, f)
    log_message("INFO", f"Label mapping successfully saved to '{LABELS_FILE}'.")
except Exception as e:
    log_message("ERROR", f"Failed to save label mapping: {e}")

# --- Train the Face Recognizer ---
log_message("INFO", "Starting to train the face recognizer...")
try:
    # Convert y_labels to a NumPy array before training.
    recognizer.train(x_train, np.array(y_labels))
    log_message("INFO", "Face recognizer training completed.")
except cv2.error as e:
    log_message("ERROR", f"OpenCV training error: {e}")
    log_message("ERROR", "Please ensure training data (x_train and y_labels) meets requirements.")
except Exception as e:
    log_message("ERROR", f"An unexpected error occurred during training: {e}")

# Save the trained model.
log_message("INFO", f"Saving trained model to '{TRAINER_FILE}'...")
try:
    recognizer.save(TRAINER_FILE)
    log_message("INFO", f"Trained model successfully saved to '{TRAINER_FILE}'.")
except Exception as e:
    log_message("ERROR", f"Failed to save trained model: {e}")

log_message("INFO", "Face recognition training script finished.")
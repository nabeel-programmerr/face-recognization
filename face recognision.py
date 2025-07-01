import cv2
import numpy as np
import os

# Load the pre-trained face detector model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define paths
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directory where images are stored
data_dir = "C:/Users/MUHAMMED NABEEL/OneDrive/Desktop/Ai Project/Face recognisation  system/faces_dataset"

# Function to read images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        if person_name not in label_map:
            label_map[person_name] = label_count
            label_count += 1

        for image_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label_map[person_name])

    return images, labels, label_map

# Load training data
print("Loading training data...")
train_images, train_labels, label_map = load_images_from_folder(data_dir)

# Train the face recognizer
print("Training face recognizer...")
face_recognizer.train(train_images, np.array(train_labels))

# Start webcam for real-time face recognition
cap = cv2.VideoCapture(0)

print("Starting face recognition...")
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        
        # Recognize the face
        label, confidence = face_recognizer.predict(face_img)
        
        # Find the person's name from label
        for name, label_id in label_map.items():
            if label == label_id:
                person_name = name
                break
        
        # Draw rectangle around face and display name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{person_name} ({round(confidence, 2)})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

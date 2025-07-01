import cv2
import os

# Create folder for storing face images
person_name = input("Enter the name of the person: ")
os.makedirs(f"faces_dataset/{person_name}", exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces and save them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        img_count += 1
        img_path = f"faces_dataset/{person_name}/image_{img_count}.jpg"
        cv2.imwrite(img_path, face)
        print(f"Image {img_count} saved at {img_path}")

    cv2.imshow('Capturing Faces', frame)

    # Press 'q' to quit or 's' to save 50 images
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if img_count >= 50:
        print(f"Captured 50 images for {person_name}")
        break

cap.release()
cv2.destroyAllWindows()

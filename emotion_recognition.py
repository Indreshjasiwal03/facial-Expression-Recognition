import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion recognition model
model = load_model('models/emotion_model.h5')

# Define emotion labels (change based on your model)
emotion_labels = ["angry", "disgust",  "fear" , "happy", "neutral", "sad", "surprise"] 

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(face):
    """
    Predicts the emotion from the given face image with confidence score.
    """
    # Preprocess the face image
    face = cv2.resize(face, (48, 48))  # Resize the face image to 48x48
    face = face.astype("float32") / 255.0  # Normalize the image
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = np.expand_dims(face, axis=-1)  # Add channel dimension

    # Predict emotion probabilities
    predictions = model.predict(face)
    
    # Get the highest probability and corresponding emotion
    max_index = np.argmax(predictions)
    max_probability = predictions[0][max_index]
    predicted_emotion = emotion_labels[max_index]

    # Debugging: Print all probabilities
    print(f"Emotion probabilities: {predictions}")
    
    # Only return if confidence is above a threshold
    if max_probability > 0.5:# Adjust the threshold as needed
        return f"{predicted_emotion}"  #(Confidence: {max_probability:.2f})" (Confidence: {max_probability:.2f})
    else:
        return "Uncertain prediction"

# Example usage
# Assuming model and emotion_labels are defined
# face = cv2.imread("path_to_face_image")
# emotion = predict_emotion(face)
# print(emotion)

def start_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Detect faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face

            # Crop the face from the frame
            face = gray[y:y + h, x:x + w]
            predicted_emotion = predict_emotion(face)  # Get predicted emotion

            # Display emotion label on the frame
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)  # Show the frame with the predicted emotion

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit when 'q' is pressed
            break

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close OpenCV windows

if __name__ == "__main__":
    start_webcam()  # Start webcam feed

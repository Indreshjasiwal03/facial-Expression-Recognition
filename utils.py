import cv2
import numpy as np

def face_detect(image, use_dnn=False):
    """
    Detect faces in the input image using Haar cascades or DNN.
    
    Parameters:
        image (np.array): Input image.
        use_dnn (bool): Use Deep Learning-based detector if True.
    
    Returns:
        faces (list): List of detected face coordinates (x, y, w, h).
    """
    if use_dnn:
        # DNN-based face detection
        net = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades.replace('haarcascades/', 'deploy.prototxt'),
            cv2.data.haarcascades.replace('haarcascades/', 'res10_300x300_ssd_iter_140000.caffemodel')
        )
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Adjust confidence threshold as needed
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2 - x, y2 - y))
    else:
        # Haar cascade-based face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhance contrast
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

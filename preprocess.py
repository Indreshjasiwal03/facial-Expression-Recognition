import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(input_dir, output_dir='data/processed/', image_size=(48, 48)):
    """
    Preprocesses the FER2013 dataset for emotion recognition.

    Parameters:
        input_dir (str): Path to the FER2013 dataset folder.
        output_dir (str): Directory to save preprocessed data.
        image_size (tuple): Target size for resizing images.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define emotion labels and map them to integers
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    labels_dict = {label: idx for idx, label in enumerate(emotion_labels)}

    images = []
    labels = []

    # Iterate over train and test folders
    for phase in ['train', 'test']:
        phase_folder = os.path.join(input_dir, phase)

        for label in emotion_labels:
            folder_path = os.path.join(phase_folder, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found.")
                continue

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Read the image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not load image {image_path}. Skipping.")
                    continue

                # Enhance the image (contrast and noise reduction)
                image = cv2.equalizeHist(image)  # Improve contrast
                image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduce noise

                # Resize and normalize the image
                image = cv2.resize(image, image_size)
                image = image / 255.0  # Normalize pixel values to [0, 1]

                images.append(image)
                labels.append(labels_dict[label])

    # Convert lists to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    # Add channel dimension (needed for CNNs)
    images = np.expand_dims(images, axis=-1)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=len(emotion_labels))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Save the processed data as .npy files
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print(f"Data preprocessing complete. Files saved in {output_dir}")

# Run preprocessing
input_dir = 'fer2013'  # Path to the FER-2013 dataset folder
preprocess_data(input_dir)

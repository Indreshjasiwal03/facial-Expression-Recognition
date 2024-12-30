import cv2

def test_camera_with_info():
    """
    Test the webcam and display additional information.
    Press 'q' to quit the test.
    """
    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    print("Press 'q' to quit the camera test.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break

        # Display resolution information
        height, width = frame.shape[:2]
        text = f"Resolution: {width}x{height}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Test Camera", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_with_info()

# import cv2

# def list_cameras():
#     """
#     List all available camera indices.
#     """
#     index = 0
#     while True:
#         cap = cv2.VideoCapture(index)
#         if not cap.read()[0]:
#             break
#         else:
#             print(f"Camera found at index {index}")
#         cap.release()
#         index += 1

# def test_camera_with_info(camera_index=0):
#     """
#     Test the specified camera and display additional information.
#     Press 'q' to quit the test.
#     """
#     # Start webcam
#     cap = cv2.VideoCapture(camera_index)

#     if not cap.isOpened():
#         print(f"Error: Camera at index {camera_index} not found!")
#         return

#     print("Press 'q' to quit the camera test.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame!")
#             break

#         # Display resolution information
#         height, width = frame.shape[:2]
#         text = f"Resolution: {width}x{height}"
#         cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         cv2.imshow("Test Camera", frame)

#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     print("Listing available cameras...")
#     list_cameras()
    
#     # Replace 1 with the external camera index after checking with list_cameras()
#     print("Testing camera at index 1...")
#     test_camera_with_info(camera_index=1)

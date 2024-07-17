import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with other model weights like 'yolov8s.pt', 'yolov8m.pt', etc.

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

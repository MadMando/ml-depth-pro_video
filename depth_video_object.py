import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import depth_pro

# Set Torch backend configurations for optimization
torch.backends.cudnn.benchmark = True
torch.cuda.synchronize()

# Define device for model computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load depth model and preprocessing transform
depth_model, depth_transform = depth_pro.create_model_and_transforms()
depth_model.to(device)
depth_model.half()
depth_model.eval()

# Load pre-trained object detection model (e.g., YOLOv5 or other)
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
object_detection_model.to(device)
object_detection_model.eval()

def process_frame(frame):
    """
    Process a single video frame for depth prediction and object detection.

    Args:
        frame (numpy.ndarray): A single frame captured from a video feed.

    Returns:
        tuple: Depth map and detected objects with bounding boxes.
    """
    # Convert frame to PIL Image
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Depth prediction
    image_tensor = depth_transform(frame_image).to(device).half()
    with torch.no_grad():
        depth_prediction = depth_model.infer(image_tensor)
    depth = depth_prediction["depth"].to(device)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    # Object detection
    results = object_detection_model(frame)
    detections = results.pandas().xyxy[0]  # Extract detection results
    
    return depth_normalized.cpu().numpy(), detections

def live_depth_and_detection_feed():
    """
    Capture and process live video feed from the webcam to visualize depth maps and detect objects.
    """
    cap = cv2.VideoCapture(0)  # Open webcam (index 0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Process the frame to get depth and object detections
        depth_map, detections = process_frame(frame)
        
        # Visualize depth
        depth_visual = (depth_map * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        
        # Annotate objects on the frame
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Combine depth colormap and detection frame
        combined = np.hstack((frame, depth_colormap))
        
        # Display the result
        cv2.imshow("Depth and Object Detection (press q to exit)", combined)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_depth_and_detection_feed()

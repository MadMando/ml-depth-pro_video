import cv2
import torch
from PIL import Image
import numpy as np
import depth_pro
from ultralytics import YOLO


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

# Load pre-trained YOLOv8 model
object_detection_model = YOLO('yolov8l.pt') 
object_detection_model.to(device)
object_detection_model.eval()

def calculate_distance(depth_map, bbox, depth_scale=4.3):
    """
    Calculate the distance to an object using the depth map and bounding box.

    Args:
        depth_map (numpy.ndarray): The normalized depth map.
        bbox (tuple): The bounding box (xmin, ymin, xmax, ymax) of the detected object.
        depth_scale (float): Scaling factor to convert normalized depth to physical distance.

    Returns:
        tuple: The distance in feet and inches.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(depth_map.shape[1], xmax), min(depth_map.shape[0], ymax)
    
    # Extract depth values within the bounding box
    object_depth = depth_map[ymin:ymax, xmin:xmax]
    
    # Calculate average depth and scale it
    if object_depth.size > 0:
        distance_meters = np.mean(object_depth) * depth_scale
    else:
        distance_meters = 0.0  # Default to 0 if the bounding box is empty

    # Convert meters to feet
    distance_feet = distance_meters * 3.28084
    
    # Separate into feet and inches
    feet = int(distance_feet)
    inches = int((distance_feet - feet) * 12)
    
    return feet, inches

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
    results = object_detection_model(frame)  # YOLOv8 inference
    detections = results[0].boxes  # Extract bounding box results
    
    # Convert detections to pandas-like format
    detection_data = []
    for box in detections:
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf.cpu().numpy().item())  # Confidence score
        cls = int(box.cls.cpu().numpy().item())  # Class ID
        label = object_detection_model.names[cls]  # Class label
        detection_data.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "confidence": confidence, "name": label})
    
    return depth_normalized.cpu().numpy(), detection_data

def live_depth_and_detection_feed():
    """
    Capture and process live video feed from the webcam to visualize depth maps, detect objects, and calculate distances.
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
        for detection in detections:
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            label = f"{detection['name']} {detection['confidence']:.2f}"
            
            # Calculate distance using depth map
            feet, inches = calculate_distance(depth_map, (xmin, ymin, xmax, ymax))
            distance_text = f"Dist: {feet}ft {inches}in"
            
            # Draw bounding box and annotations
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, distance_text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Combine depth colormap and detection frame
        combined = np.hstack((frame, depth_colormap))
        
        # Display the result
        cv2.imshow("Depth, Object Detection, and Distance (press q to exit)", combined)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# def live_depth_and_detection_feed():
#     """
#     Capture and process live video feed from the webcam to visualize depth maps, detect objects, and calculate distances.
#     """
#     cap = cv2.VideoCapture(0)  # Open webcam (index 0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break
        
#         # Process the frame to get depth and object detections
#         depth_map, detections = process_frame(frame)
        
#         # Visualize depth
#         depth_visual = (depth_map * 255).astype(np.uint8)
#         depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        
#         # Overlay depth colormap with transparency
#         overlay = cv2.addWeighted(depth_colormap, 0.3, frame, 0.7, 0)  # 30% depth map, 70% original frame
        
#         # Annotate objects on the overlay
#         for detection in detections:
#             xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
#             label = f"{detection['name']} {detection['confidence']:.2f}"
            
#             # Calculate distance using depth map
#             feet, inches = calculate_distance(depth_map, (xmin, ymin, xmax, ymax))
#             distance_text = f"Dist: {feet}ft {inches}in"
            
#             # Draw bounding box and annotations
#             cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(overlay, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cv2.putText(overlay, distance_text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         # Display the overlay
#         cv2.imshow("Depth, Object Detection, and Distance (press q to exit)", overlay)
        
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    live_depth_and_detection_feed()

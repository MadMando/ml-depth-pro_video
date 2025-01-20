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

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.half()
model.eval()


def process_frame(frame):
    """
    Process a single video frame to predict depth.

    Args:
        frame (numpy.ndarray): A single frame captured from a video feed.

    Returns:
        numpy.ndarray: Normalized depth map as a NumPy array.
    """
    # Convert frame to PIL Image
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the transform and convert input to FP16
    image_tensor = transform(frame_image).to(device).half()
    
    # predioct depth
    with torch.no_grad():
        prediction = model.infer(image_tensor)
    
    # Get depth map
    depth = prediction["depth"].to(device)
    
    # Normalize depth for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth_normalized.cpu().numpy()


def live_depth_feed():
    """
    Capture and process live video feed from the webcam to visualize depth maps.
    """
    cap = cv2.VideoCapture(0)  # Open webcam (index 0)
    if not cap.isOpened():
        print("Error: no open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Process the frame to get depth
        depth_map = process_frame(frame)
        
        # Normalize and convert depth map to 8-bit for visualization
        depth_visual = (depth_map * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        
        # Display the depth map
        cv2.imshow("Depth Map press q to exit", depth_colormap)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_depth_feed()

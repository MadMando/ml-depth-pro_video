def live_depth_feed():
    """Capture and process live video feed from webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam (index 0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
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
        cv2.imshow("Depth Map", depth_colormap)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# In[9]:


live_depth_feed()


from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from depth import DepthEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Object detection with depth estimation')
    parser.add_argument('--video', type=str, default='/home/orin/Test/Test_Model/AriaEverydayActivities_1.0.0_loc5_script4_seq6_rec1_preview_rgb.mp4', help='Path to video file or camera index')
    parser.add_argument('--yolo-model', type=str, default='yolo11n.pt', help='Path to YOLO model')
    parser.add_argument('--depth-model', type=str, default='weights/depth_anything_vits14_406_dla0.trt', 
                        help='Path to depth model TRT engine')
    parser.add_argument('--input-size', type=int, default=406, help='Input size for depth model')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save results')
    return parser.parse_args()

def calculate_object_depth(depth_map, x1, y1, x2, y2):
    """
    Calculate the average depth of an object within its bounding box
    
    Args:
        depth_map: The depth map
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        float: Average depth value of the object
    """
    # Get image dimensions
    h, w = depth_map.shape
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    # Calculate depth in central region of bounding box (to avoid boundary effects)
    center_x1 = x1 + (x2 - x1) // 4
    center_y1 = y1 + (y2 - y1) // 4
    center_x2 = x2 - (x2 - x1) // 4
    center_y2 = y2 - (y2 - y1) // 4
    
    # Ensure central region has valid size
    if center_x2 <= center_x1 or center_y2 <= center_y1:
        # If bounding box is too small, use center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return depth_map[center_y, center_x]
    
    # Extract region and calculate average depth
    roi = depth_map[center_y1:center_y2, center_x1:center_x2]
    return np.mean(roi)

def main():
    args = parse_args()
    
    # Initialize YOLO model
    yolo_model = YOLO(args.yolo_model)
    
    # Initialize depth estimation engine
    depth_engine = DepthEngine(
        input_size=args.input_size,
        trt_engine_path=args.depth_model,
        save_path=args.save_path
    )
    
    # Get video properties
    cap = cv2.VideoCapture(args.video if args.video != '0' else 0)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video}")
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video dimensions: {frame_width}x{frame_height}")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO object detection
        results = yolo_model(frame)
        
        # Run depth estimation
        depth_map = depth_engine.infer(frame)
        
        # Visualize depth map
        vis_depth = depth_engine.visualize_depth(depth_map)
        
        # Convert to grayscale and invert (closer objects appear brighter but have smaller distance values)
        depth_gray = cv2.cvtColor(vis_depth, cv2.COLOR_BGR2GRAY)
        depth_resized = 255 - depth_gray  # Inverted for visualization (brighter = closer)
        
        # Normalize depth map for distance calculation (0-1 range)
        # Lower values (closer to 0) = closer objects, Higher values (closer to 1) = farther objects
        normalized_depth = depth_gray / 255.0
        
        # Process detection results and calculate distances
        for result in results:
            boxes = result.boxes
            for box in boxes:
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get class name and confidence
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = result.names[cls_id]
                    
                    # Calculate depth using the object region
                    object_depth = calculate_object_depth(normalized_depth, x1, y1, x2, y2)
                    
                    # Apply calibration formula from experimental data
                    # distance (cm) = 3.1002 * normalized_depth - 0.4657
                    distance_cm = 3.1002 * object_depth * 100 - 0.4657
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display object information with distance
                    label = f"{cls_name}: {conf:.2f}, Distance: {distance_cm:.1f}cm"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw a point at the central region used for depth calculation
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                    
                except Exception as e:
                    print(f"Error processing detection: {e}")
        
        # Display results
        cv2.imshow("Object Detection", frame)
        cv2.imshow("Depth Map", vis_depth)
        cv2.imshow("Grayscale Depth Map", depth_resized)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    depth_engine.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
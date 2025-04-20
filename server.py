#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path
import math # Added for distance calculation
import functools # For partial application

import cv2
import numpy as np
import websockets

USER_COLOR = (0, 255, 0) # Green
NORMAL_BOX_COLOR = (255, 0, 0) # Blue
THREAT_COLOR = (0, 0, 255) # Red
THREAT_THICKNESS = 3
NORMAL_THICKNESS = 2
BUBBLE = 350 # Proximity threshold in pixels

def preprocess_image(image, threshold=200):
    """
    Preprocess the image by applying an inverted binary threshold.
    Regions below the threshold become white (255), others black (0).
    
    Args:
        image: Input image (color or grayscale)
        threshold: Brightness threshold (0-255)
        
    Returns:
        Inverted binary thresholded image (BGR format)
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize the image to improve contrast
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply inverted binary threshold
    _, binary_inv = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert back to BGR for visualization and saving
    binary_inv_bgr = cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)
    
    return binary_inv_bgr


def detect_blobs(image):
    """
    Detect blobs in the image.
    
    Args:
        image: Input image (grayscale or color) for detection
        
    Returns:
        List of detected keypoints
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize the image to improve contrast
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Set up blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds for bright spots (people in thermal)
    params.minThreshold = 0
    params.maxThreshold = 100
    
    # Filter by Area
    # params.filterByArea = True
    # params.minArea = 50
    # params.maxArea = 1500
    
    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.5
    
    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.5
    
    # Filter by Inertia (measure of elongation)
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.2
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(normalized)
    
    return keypoints


def detect_contours(image):
    """
    Detect contours in the image.
    Handles both original and preprocessed binary images.
    
    Args:
        image: Input image (BGR format, potentially binary)
        
    Returns:
        List of detected contours
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if the image is already binary (only 0 and 255 values)
    # This indicates it was likely preprocessed
    unique_values = np.unique(gray)
    # Check for binary (0, 255) or inverted binary (0, 255) or just black/white
    is_binary = len(unique_values) <= 2 and (0 in unique_values or 255 in unique_values)

    if is_binary:
        # If already binary (normal or inverted), use it directly
        # For contour finding, we need white objects on black background
        # If it's our inverted preprocessed image (0=background, 255=object), use as is.
        # If it's somehow a normal binary (255=background, 0=object), it needs inversion.
        # Assuming input 'image' from preprocessing is always inverted binary (0 background)
        if 255 in unique_values: # Check if there are any white pixels (objects)
             thresh = gray
             #print("Using preprocessed binary image for contours.") # Reduced verbosity
        else: # Only black pixels? Empty image for contour finding.             
             thresh = np.zeros_like(gray)
             #print("Preprocessed image is all black, no contours to find.")
    else:
        # If not binary, apply normalization and thresholding (find bright spots)
        #print("Applying internal thresholding for contours (finding bright spots).")
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(normalized, 180, 255, cv2.THRESH_BINARY)
    
    # # Apply morphological operations to clean up the image
    # # Use a slightly larger kernel for potentially cleaner results on binary images
    # kernel_size = 5 if is_binary else 3
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours on the binary image (expecting white objects)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return all found contours without filtering
    return contours


def draw_detections(original_image, detections, use_contours):
    """
    Draw detection results (blobs or contours) onto the original image.
    
    Args:
        original_image: The unprocessed image to draw on
        detections: List of keypoints (for blobs) or contours
        use_contours: Boolean indicating if detections are contours
        
    Returns:
        Image with detections drawn on it
    """
    result_image = original_image.copy()
    
    if use_contours:
        # Draw all contours and their bounding boxes
        cv2.drawContours(result_image, detections, -1, (0, 255, 0), 2) # Draw all contours
        for contour in detections:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:  # Blob detection
        # Draw keypoints with rich information
        result_image = cv2.drawKeypoints(result_image, detections, np.array([]), 
                                         (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Draw bounding boxes around keypoints
        for kp in detections:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            radius = size // 2
            cv2.rectangle(result_image, (x - radius, y - radius), 
                         (x + radius, y + radius), (0, 0, 255), 2)
                         
    return result_image

def draw_boxes_around_white_regions(binary_image, min_area=10):
    """
    Finds white regions in a binary image (typically inverted preprocessed) 
    and returns their bounding boxes.

    Args:
        binary_image: The black and white image (BGR format expected, white=object).
        min_area: Minimum pixel area to consider a region valid (filters noise).

    Returns:
        List of bounding boxes [(x, y, w, h)]
    """
    
    # 1. Ensure grayscale
    if len(binary_image.shape) == 3:
        gray_binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_binary = binary_image.copy()

    # Image is already inverted (white objects on black background) - skip bitwise_not

    # 2. Find connected components and stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_binary, connectivity=8)

    detections = [] # Store bounding boxes found

    # 3. Iterate through components (skip label 0, which is the background)
    for i in range(1, num_labels):
        # 4. Extract stats
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 5. Filter small noise areas
        if area >= min_area:
            # 6. Store the box
            detections.append((x, y, w, h)) 
            
    #print(f"Found {len(detections)} white regions (originally black) above min_area {min_area}")
    return detections

# --- Helper Functions for Threat Detection ---

def bbox_center(bbox):
    """Calculates the center of a bounding box (x, y, w, h)."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)

def distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- End Helper Functions ---

def identify_and_draw_threats(original_img, all_bboxes, prev_user_bbox, is_first_frame):
    """
    Identifies the user, tracks them, detects threats, and draws boxes.

    Args:
        original_img: The image to draw on.
        all_bboxes: List of detected bounding boxes [(x, y, w, h)] for this frame.
        prev_user_bbox: The user's bounding box (x, y, w, h) from the previous frame.
        is_first_frame: Boolean flag indicating if this is the first frame.

    Returns:
        Tuple: (Image with boxes drawn, 
                identified user_bbox for this frame or None, 
                list of threat bboxes [(x, y, w, h)])
    """
    output_image = original_img.copy()
    current_user_bbox = None
    threat_bboxes = [] # List to store threat bboxes

    if not all_bboxes:
        return output_image, None, [] # No boxes detected

    # 1. Identify or Track User
    if is_first_frame or prev_user_bbox is None:
        # First frame or lost track: User is the largest box
        largest_area = -1
        for bbox in all_bboxes:
            x, y, w, h = bbox
            area = w * h
            if area > largest_area:
                largest_area = area
                current_user_bbox = bbox
    else:
        # Subsequent frames: Find box closest to previous user center
        min_dist = float('inf')
        prev_center = bbox_center(prev_user_bbox)
        tracked = False
        for bbox in all_bboxes:
            current_center = bbox_center(bbox)
            dist = distance(prev_center, current_center)
            if dist < min_dist:
                min_dist = dist
                current_user_bbox = bbox
                tracked = True
        if not tracked:
             # Fallback if no boxes were found (shouldn't happen if all_bboxes is not empty)
             # Or maybe add a max distance threshold for tracking?
             current_user_bbox = None # Consider user lost

    # 2. Draw Boxes (User, Threats, Normal)
    if current_user_bbox:
        user_center = bbox_center(current_user_bbox)
        # Draw user box first
        ux, uy, uw, uh = current_user_bbox
        cv2.rectangle(output_image, (ux, uy), (ux + uw, uy + uh), USER_COLOR, NORMAL_THICKNESS)

        # Draw other boxes and check for threats
        for bbox in all_bboxes:
            if bbox == current_user_bbox:
                continue # Skip the user box, already drawn

            other_center = bbox_center(bbox)
            dist_to_user = distance(user_center, other_center)
            
            x, y, w, h = bbox
            if dist_to_user <= BUBBLE:
                # Threat detected
                cv2.rectangle(output_image, (x, y), (x + w, y + h), THREAT_COLOR, THREAT_THICKNESS)
                threat_bboxes.append(bbox) # Add to threat list
                text_color = (255, 255, 255) # White
            else:
                # Normal box
                cv2.rectangle(output_image, (x, y), (x + w, y + h), NORMAL_BOX_COLOR, NORMAL_THICKNESS)
                text_color = (255, 255, 255) # White

            # Add distance text near the center
            text = f"{dist_to_user:.0f}px"
            font_scale = 0.4
            font_thickness = 1
            # Position text slightly above the center for better visibility
            text_pos = (int(other_center[0] - 5), int(other_center[1] - 5))
            cv2.putText(output_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, text_color, font_thickness, cv2.LINE_AA)

    else:
        # No user identified in this frame, draw all boxes normally
        for bbox in all_bboxes:
             x, y, w, h = bbox
             cv2.rectangle(output_image, (x, y), (x + w, y + h), NORMAL_BOX_COLOR, NORMAL_THICKNESS)

    # Return the annotated image, the user's bbox, and the list of threat bboxes
    return output_image, current_user_bbox, threat_bboxes

async def save_frame(frame_data, frame_number, save_dir, 
                   prev_user_bbox, is_first_frame, # Added for threat detection state
                   crop_coords=None, use_contours=False, preprocess=False, threshold=200):
    """
    Process and save a received frame. 
    If preprocess=True, finds regions below threshold, performs threat detection, 
    and draws boxes on original.
    
    Args:
        frame_data: Base64 encoded image data
        frame_number: Frame number from the video
        save_dir: Directory to save frames to
        prev_user_bbox: User's bounding box from the previous frame (for tracking).
        is_first_frame: Flag indicating if this is the first frame processed.
        crop_coords: Optional tuple (x1, y1, x2, y2) for cropping
        use_contours: If preprocess=False, use contour detection instead of blob detection.
        preprocess: Whether to apply inverted preprocessing and find dark regions.
        threshold: Brightness threshold for preprocessing.
        
    Returns:
        Tuple(filepath, current_user_bbox, threat_bboxes, img_to_save) or (None, None, [], None)
        - filepath: Path where the annotated image was saved.
        - current_user_bbox: Bbox identified as the user in this frame.
        - threat_bboxes: List of bboxes identified as threats.
        - img_to_save: The numpy array of the final annotated image (needed for sending).
    """
    img_to_save = None # Initialize here
    try:
        # Decode the base64 image data
        img_data = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"Error: Failed to decode image data for frame {frame_number}")
            return None, None, [], None
        
        # Apply cropping if coordinates are provided
        if crop_coords and len(crop_coords) == 4:
            x1, y1, x2, y2 = crop_coords
            # Check if the crop coordinates are valid
            height, width = original_img.shape[:2]
            if 0 <= x1 < x2 and 0 <= y1 < y2 and x2 <= width and y2 <= height:
                original_img = original_img[y1:y2, x1:x2]
            else:
                print(f"Warning: Invalid crop coordinates {crop_coords} for image of size {width}x{height}. Skipping crop.")
        
        # Create filename based on frame number
        filename = f"frame_{frame_number:06d}.jpg"
        filepath = save_dir / filename
        
        current_user_bbox = None # Specific to this frame
        detections = [] # General detections list
        threat_bboxes = [] # Specific to threat detection mode
        img_to_save = original_img.copy() # Start with the original image
        
        if preprocess:
            # Apply inverted preprocessing
            preprocessed_inv_img = preprocess_image(original_img, threshold)
            # Save the preprocessed image
            preprocessed_filename = f"frame_{frame_number:06d}_preprocessed_inv.jpg"
            preprocessed_filepath = save_dir / preprocessed_filename
            if not cv2.imwrite(str(preprocessed_filepath), preprocessed_inv_img):
                 print(f"Error: Failed to save preprocessed image {preprocessed_filepath}")
            # else:
                # print(f"Saved inverted preprocessed frame to {preprocessed_filepath}") # Reduced verbosity
            
            # Find white regions (originally black)
            all_bboxes = draw_boxes_around_white_regions(preprocessed_inv_img)
            
            # Identify user, track, detect threats, and draw boxes
            img_to_save, current_user_bbox, threat_bboxes = identify_and_draw_threats(
                original_img, all_bboxes, prev_user_bbox, is_first_frame
            )
            detections = all_bboxes # Store raw boxes as detections for this mode

        else:
            # Standard detection (blob or contour) on the original image
            detection_img = original_img # Use original for detection when not preprocessing
            if use_contours:
                detections = detect_contours(detection_img)
                # print(f"Detected {len(detections)} contours in frame {frame_number}") # Reduced verbosity
            else:
                detections = detect_blobs(detection_img)
                # print(f"Detected {len(detections)} blobs in frame {frame_number}") # Reduced verbosity
            
            # Draw standard detections onto the original image
            if detections:
                 img_to_save = draw_detections(original_img, detections, use_contours)
            
        # Save the final image (original + appropriate detections/boxes)
        if not cv2.imwrite(str(filepath), img_to_save):
            print(f"Error: Failed to save final image {filepath}")
            return None, None, [], None # Return None for all if save failed
        # else:
             # Add log for successful save of final image
             # print(f"Saved final annotated frame {frame_number} to {filepath}") # Reduced verbosity

        # Return the path, user bbox, threat list, and the image array itself
        return filepath, current_user_bbox, threat_bboxes, img_to_save
        
    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, [], None

async def process_client_frames(websocket, queue, save_dir, crop_coords=None, use_contours=False, preprocess=False, threshold=200):
    """Receive frames from the client, process them, send ACKs, and put threat data into the queue."""
    client_addr = websocket.remote_address
    print(f"Processing frames from client: {client_addr}")
    frames_processed = 0
    frames_failed = 0
    # State for threat detection/user tracking
    user_bbox = None 
    is_first_frame = True

    try:
        async for message in websocket:
            try:
                # Parse the JSON message
                data = json.loads(message)
                frame_number = data.get("frame_number")
                frame_data = data.get("image_data")
                
                if frame_number is None or frame_data is None:
                    print(f"Received invalid message format from {client_addr}")
                    await websocket.send(json.dumps({"status": "error", "frame_number": frame_number, "message": "Invalid message format"}))
                    continue
                
                # Process and Save the frame
                filepath, current_user_bbox, threat_bboxes, img_to_save = await save_frame(
                    frame_data, frame_number, save_dir, 
                    user_bbox, is_first_frame, # Pass state to save_frame
                    crop_coords, use_contours, preprocess, threshold
                )
                
                # Update tracking state for the next frame
                if filepath: # Only update if processing was successful
                    user_bbox = current_user_bbox # Update user_bbox for the next iteration
                    if is_first_frame:
                        is_first_frame = False # No longer the first frame
                
                # Send acknowledgment back to client
                if filepath:
                    ack_message = json.dumps({"status": "processed", "frame_number": frame_number})
                    frames_processed += 1

                    # --- Buffer Threat Data --- 
                    # Only buffer if preprocessing is enabled, a user is identified, and threats exist
                    if preprocess and queue and current_user_bbox and threat_bboxes:
                        try:
                            # Encode the annotated image to base64
                            _, buffer = cv2.imencode('.jpg', img_to_save)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Get centers
                            user_center = list(map(int, bbox_center(current_user_bbox)))
                            threat_centers = [list(map(int, bbox_center(tb))) for tb in threat_bboxes]

                            # Construct message
                            threat_data = {
                                "user_center": user_center,
                                "threat_center": threat_centers,
                                "img": img_base64,
                                "frame_number": frame_number # Include frame number for context
                            }
                            
                            # Put data onto the queue
                            await queue.put(threat_data)
                            print(f"Buffered threat data for frame {frame_number}.")
                        
                        # Remove websocket specific exceptions here, queue errors are different
                        # except websockets.exceptions.ConnectionClosed:
                        #     print(f"Frontend connection closed while trying to send threat data for frame {frame_number}.")
                        #     queue = None # Stop trying to send
                        except Exception as fe_send_err: # Rename? This is now queue error
                            print(f"Warning: Error buffering threat data for frame {frame_number}: {fe_send_err}")
                            # Queue errors (like full queue if maxsize set) might need handling
                    # --- End Buffer Threat Data ---

                else: # Processing failed
                    ack_message = json.dumps({"status": "error", "frame_number": frame_number, "message": "Processing failed"})
                    frames_failed += 1
                    print(f"Failed to process frame {frame_number} from {client_addr}")
                
                await websocket.send(ack_message)

            except json.JSONDecodeError:
                print(f"Received non-JSON message from {client_addr}")
                await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON"}))
            except Exception as e:
                # Log specific error during message handling but keep connection open
                print(f"Error handling message from {client_addr}: {e}")
                import traceback
                traceback.print_exc()
                # Send error ack if possible
                try:
                    frame_num_for_error = data.get("frame_number", -1) if 'data' in locals() else -1
                    await websocket.send(json.dumps({"status": "error", "frame_number": frame_num_for_error, "message": str(e)}))
                except Exception as send_err:
                    print(f"Failed to send error acknowledgment to {client_addr}: {send_err}")
                
    except websockets.exceptions.ConnectionClosedOK:
        print(f"Connection closed normally by client: {client_addr}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error for {client_addr}: {e}")
    except Exception as e:
        print(f"Unexpected error in frame processing loop for {client_addr}: {e}")
        import traceback
        traceback.print_exc()        
    finally:
        print(f"Finished processing frames for {client_addr}. Processed: {frames_processed}, Failed: {frames_failed}")


async def main():
    parser = argparse.ArgumentParser(description="Connect to client and process video frames.")
    parser.add_argument("--client-host", type=str, default="localhost", help="Client host address to connect to (default: localhost)")
    parser.add_argument("--client-port", type=int, default=8765, help="Client port to connect to (default: 8765)")
    parser.add_argument("--save-dir", type=str, default="./files/frames", 
                        help="Directory to save frames (default: ./files/frames)")
    parser.add_argument("--crop-x1", type=int, default=108, help="Left coordinate for cropping (default: 108)")
    parser.add_argument("--crop-y1", type=int, default=359, help="Top coordinate for cropping (default: 359)")
    parser.add_argument("--crop-x2", type=int, default=984, help="Right coordinate for cropping (default: 984)")
    parser.add_argument("--crop-y2", type=int, default=1527, help="Bottom coordinate for cropping (default: 1527)")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping")
    parser.add_argument("--use-contours", action="store_true", default=True,
                        help="Use contour detection instead of the default blob detection (ignored if --preprocess is used).")
    parser.add_argument("--preprocess", action="store_true", default=True,
                        help="Apply inverted thresholding and find dark regions, drawing boxes on original image.")
    parser.add_argument("--threshold", type=int, default=150,
                        help="Brightness threshold for preprocessing (0-255, default: 150)")
    parser.add_argument("--frontend-host", type=str, default="0.0.0.0",
                        help="Host address for this server to listen on for frontend connections (default: 0.0.0.0)")
    parser.add_argument("--frontend-port", type=int, default=8766, # Different default from client
                        help="Port for this server to listen on for frontend connections (default: 8766)")
    
    args = parser.parse_args()
        
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    crop_coords = None
    if not args.no_crop:
        crop_coords = (args.crop_x1, args.crop_y1, args.crop_x2, args.crop_y2)
        print(f"Will crop images to: ({args.crop_x1}, {args.crop_y1}) to ({args.crop_x2}, {args.crop_y2})")
    
    if args.preprocess:
        print(f"Preprocessing enabled: Finding dark regions below threshold {args.threshold}.")
    else:
        detection_method = "contour" if args.use_contours else "blob"
        print(f"Standard detection enabled using {detection_method} detection.")

    client_url = f"ws://{args.client_host}:{args.client_port}"
    print(f"Attempting to connect to client at {client_url}...")
    print(f"Frames will be saved to {save_dir.absolute()}")
    print(f"Will listen for frontend connections on ws://{args.frontend_host}:{args.frontend_port}")

    threat_buffer_queue = asyncio.Queue()

    # Start the frontend server task in the background
    handler_with_queue = functools.partial(frontend_handler, queue=threat_buffer_queue)
    frontend_server = await websockets.serve(
        handler_with_queue, 
        args.frontend_host, 
        args.frontend_port
    )
    frontend_task = asyncio.create_task(frontend_server.serve_forever(), name="FrontendServerTask")
    print("Frontend server started.")

    # Run the client connection task and wait for it to complete (or fail)
    print("Starting client connection handler...")
    try:
        await connect_and_process_client(
            client_url, threat_buffer_queue, save_dir, crop_coords, 
            args.use_contours, args.preprocess, args.threshold
        )
        print("Client connection handler finished normally.")
    except Exception as client_err:
        print(f"Client connection handler exited with error: {client_err}")
        # Decide if we need to propagate the error or just log it

    # --- Graceful Shutdown --- 
    print("Client connection task finished. Starting graceful shutdown...")

    # 1. Wait for the queue to be empty
    if not threat_buffer_queue.empty():
        print(f"Waiting for {threat_buffer_queue.qsize()} items in the queue to be processed...")
        await threat_buffer_queue.join() # Waits until all task_done() calls are made
        print("Threat buffer queue is now empty.")
    else:
        print("Threat buffer queue was already empty.")

    # 2. Stop the frontend server task
    print("Stopping frontend server task...")
    if not frontend_task.done():
        frontend_task.cancel()
        try:
            await frontend_task # Wait for the task to acknowledge cancellation
        except asyncio.CancelledError:
            print("Frontend server task successfully cancelled.")
        except Exception as e:
            print(f"Error occurred while awaiting frontend task cancellation: {e}")
    else:
         print("Frontend server task was already done.")
         # Check for exceptions if it finished unexpectedly
         if frontend_task.exception():
             print(f"Frontend task finished with exception: {frontend_task.exception()}")

    # Close the server socket itself (optional but good practice)
    frontend_server.close()
    await frontend_server.wait_closed()
    print("Frontend server socket closed.")
    print("Shutdown complete.")


async def connect_and_process_client(client_url, queue, save_dir, crop_coords, use_contours, preprocess, threshold):
    """Continuously tries to connect to the client and process frames."""
    while True: # Keep trying to connect to the client
        try:
            async with websockets.connect(
                client_url,
                ping_interval=10,
                ping_timeout=10,
                open_timeout=5 # Shorter connection timeout
            ) as websocket:
                print(f"Connected to client at {client_url}")
                
                # Pass the queue instead of a frontend_websocket object
                await process_client_frames(
                    websocket, # Client connection
                    queue,    # Threat buffer queue
                    save_dir, crop_coords, 
                    use_contours, preprocess, threshold
                )
                print("Finished processing client frames session. Waiting for client disconnect/reconnect...")
                # Let the connection close gracefully or wait for next attempt
                break
                # await websocket.wait_closed() # Wait until connection is closed

        except (ConnectionRefusedError, asyncio.TimeoutError, websockets.exceptions.InvalidURI, OSError) as e:
            print(f"Client connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosed:
             print("Client connection closed unexpectedly. Retrying in 5 seconds...")
             await asyncio.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred during client connection/processing loop: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying client connection in 10 seconds...")
            await asyncio.sleep(10)

# --- Frontend Handler --- 
async def frontend_handler(websocket, path, queue):
    """Handles incoming connections from the frontend app."""
    client_addr = websocket.remote_address
    print(f"Frontend connected: {client_addr}")
    try:
        # Continuously check the queue and send data to this frontend client
        while True:
            threat_data = await queue.get() # Wait for an item from the buffer
            try:
                await websocket.send(json.dumps(threat_data))
                print(f"Sent buffered threat data frame {threat_data.get('frame_number', 'N/A')} to frontend {client_addr}")
                queue.task_done() # Acknowledge processing
            except websockets.exceptions.ConnectionClosed:
                print(f"Frontend connection {client_addr} closed while trying to send. Requeuing data.")
                # Basic requeue: Put the item back at the end. Might cause infinite loop if connection is broken.
                # A more robust solution might involve dead-letter queues or discarding.
                await queue.put(threat_data) 
                break # Exit inner loop for this client
            except Exception as send_err:
                print(f"Error sending to frontend {client_addr}: {send_err}. Data may be lost for this client.")
                # Decide if we should requeue or discard
                queue.task_done() # Mark as done even if send failed for this client to avoid blocking others

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Frontend {client_addr} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Frontend {client_addr} disconnected with error: {e}")
    except Exception as e:
        print(f"Error in frontend handler for {client_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Frontend connection closed for {client_addr}.")
# --- End Frontend Handler ---


if __name__ == "__main__":
    # Wrap the main logic in asyncio.run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
    except Exception as e:
        print(f"\nCritical error in main execution: {e}")
        import traceback
        traceback.print_exc() 
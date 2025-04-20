#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import websockets


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
    _, binary_inv = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY_INV)
    
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
             print("Using preprocessed binary image for contours.")
        else: # Only black pixels? Empty image for contour finding.             
             thresh = np.zeros_like(gray)
             print("Preprocessed image is all black, no contours to find.")
    else:
        # If not binary, apply normalization and thresholding (find bright spots)
        print("Applying internal thresholding for contours (finding bright spots).")
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

def draw_boxes_around_white_regions(target_image_to_draw_on, binary_image, min_area=10):
    """
    Finds white regions in a binary image (typically inverted preprocessed) 
    and draws bounding boxes around them on a target image.

    Args:
        target_image_to_draw_on: The image (e.g., original) where boxes will be drawn.
        binary_image: The black and white image (BGR format expected, white=object).
        min_area: Minimum pixel area to consider a region valid (filters noise).

    Returns:
        The target image with bounding boxes drawn.
    """
    
    # 1. Ensure grayscale
    if len(binary_image.shape) == 3:
        gray_binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_binary = binary_image.copy()

    # Image is already inverted (white objects on black background) - skip bitwise_not

    # 2. Find connected components and stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_binary, connectivity=8)

    output_image = target_image_to_draw_on.copy()
    box_color = (255, 0, 0) # Blue color for boxes
    thickness = 2
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
            # 6. Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, thickness)
            detections.append((x, y, w, h)) # Store the box
            
    print(f"Found {len(detections)} white regions (originally black) above min_area {min_area}")
    return output_image, detections

async def save_frame(frame_data, frame_number, save_dir, crop_coords=None, use_contours=False, preprocess=False, threshold=200):
    """
    Process and save a received frame. 
    If preprocess=True, finds regions below threshold and draws boxes on original.
    If preprocess=False, runs blob/contour detection and draws on original.
    
    Args:
        frame_data: Base64 encoded image data
        frame_number: Frame number from the video
        save_dir: Directory to save frames to
        crop_coords: Optional tuple (x1, y1, x2, y2) for cropping
        use_contours: If preprocess=False, use contour detection instead of blob detection.
        preprocess: Whether to apply inverted preprocessing and find dark regions.
        threshold: Brightness threshold for preprocessing.
        
    Returns:
        Tuple(filepath, detections) or None if processing failed. Detections are
        contours/keypoints if preprocess=False, bounding boxes if preprocess=True.
    """
    try:
        # Decode the base64 image data
        img_data = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"Error: Failed to decode image data for frame {frame_number}")
            return None, []
        
        # Apply cropping if coordinates are provided
        if crop_coords and len(crop_coords) == 4:
            x1, y1, x2, y2 = crop_coords
            # Check if the crop coordinates are valid
            height, width = original_img.shape[:2]
            if 0 <= x1 < x2 and 0 <= y1 < y2 and x2 <= width and y2 <= height:
                original_img = original_img[y1:y2, x1:x2]
            else:
                print(f"Warning: Invalid crop coordinates for image of size {width}x{height}. Skipping crop.")
        
        # Create filename based on frame number
        filename = f"frame_{frame_number:06d}.jpg"
        filepath = save_dir / filename
        
        detections = []
        img_to_save = original_img.copy() # Start with the original image
        
        if preprocess:
            # Apply inverted preprocessing
            preprocessed_inv_img = preprocess_image(original_img, threshold)
            # Save the preprocessed image
            preprocessed_filename = f"frame_{frame_number:06d}_preprocessed_inv.jpg"
            preprocessed_filepath = save_dir / preprocessed_filename
            if not cv2.imwrite(str(preprocessed_filepath), preprocessed_inv_img):
                 print(f"Error: Failed to save preprocessed image {preprocessed_filepath}")
            else:
                print(f"Saved inverted preprocessed frame to {preprocessed_filepath}")
            
            # Find white regions (originally black) and draw boxes on original image
            img_to_save, detections = draw_boxes_around_white_regions(original_img, preprocessed_inv_img)

        else:
            # Standard detection (blob or contour) on the original image
            detection_img = original_img # Use original for detection when not preprocessing
            if use_contours:
                detections = detect_contours(detection_img)
                print(f"Detected {len(detections)} contours in frame {frame_number}")
            else:
                detections = detect_blobs(detection_img)
                print(f"Detected {len(detections)} blobs in frame {frame_number}")
            
            # Draw standard detections onto the original image
            img_to_save = draw_detections(original_img, detections, use_contours)
            
        # Save the final image (original + appropriate detections/boxes)
        if not cv2.imwrite(str(filepath), img_to_save):
            print(f"Error: Failed to save final image {filepath}")
            return None, detections # Return detections even if save failed
        else:
             # Add log for successful save of final image
             print(f"Saved final annotated frame {frame_number} to {filepath}")

        return filepath, detections
        
    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        return None, []


async def handle_connection(websocket, save_dir, crop_coords=None, use_contours=False, preprocess=False, threshold=200):
    """Handle incoming WebSocket connections."""
    client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"Client connected: {client_info}")
    
    try:
        async for message in websocket:
            try:
                # Parse the JSON message
                data = json.loads(message)
                frame_number = data.get("frame_number")
                frame_data = data.get("image_data")
                
                if frame_number is None or frame_data is None:
                    print(f"Received invalid message format from {client_info}")
                    await websocket.send(json.dumps({"status": "error", "frame_number": frame_number, "message": "Invalid message format"}))
                    continue
                
                # Process and Save the frame
                filepath, detections = await save_frame(
                    frame_data, frame_number, save_dir, 
                    crop_coords, use_contours, preprocess, threshold
                )
                
                # Send acknowledgment back to client
                if filepath:
                    ack_message = json.dumps({"status": "processed", "frame_number": frame_number})
                    # Removed print statement for saved frame here, handled in save_frame
                else:
                    ack_message = json.dumps({"status": "error", "frame_number": frame_number, "message": "Processing failed"})
                    print(f"Failed to process frame {frame_number}")
                
                await websocket.send(ack_message)
                # Removed print statement for sent ACK here for cleaner logs

            except json.JSONDecodeError:
                print(f"Received non-JSON message from {client_info}")
                await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON"}))
            except Exception as e:
                # Log specific error during message handling but keep connection open
                print(f"Error handling message from {client_info}: {e}")
                # Send error ack if possible
                try:
                    frame_num_for_error = data.get("frame_number", -1) if 'data' in locals() else -1
                    await websocket.send(json.dumps({"status": "error", "frame_number": frame_num_for_error, "message": str(e)}))
                except Exception as send_err:
                    print(f"Failed to send error acknowledgment to {client_info}: {send_err}")
                
    except websockets.exceptions.ConnectionClosedOK:
        print(f"Connection closed normally with {client_info}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error for {client_info}: {e}")
    except Exception as e:
        print(f"Unexpected error in handle_connection for {client_info}: {e}")
    finally:
        print(f"Connection handler finished for {client_info}")


async def main():
    parser = argparse.ArgumentParser(description="WebSocket server to receive and save video frames")
    parser.add_argument("--host", type=str, default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--save-dir", type=str, default="./files/frames", 
                        help="Directory to save frames (default: ./files/frames)")
    parser.add_argument("--crop-x1", type=int, default=108, help="Left coordinate for cropping (default: 108)")
    parser.add_argument("--crop-y1", type=int, default=359, help="Top coordinate for cropping (default: 359)")
    parser.add_argument("--crop-x2", type=int, default=984, help="Right coordinate for cropping (default: 984)")
    parser.add_argument("--crop-y2", type=int, default=1527, help="Bottom coordinate for cropping (default: 1527)")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping")
    # Removed --detect flag
    parser.add_argument("--use-contours", action="store_true", 
                        help="Use contour detection instead of the default blob detection (ignored if --preprocess is used).")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply inverted thresholding and find dark regions, drawing boxes on original image.")
    parser.add_argument("--threshold", type=int, default=200,
                        help="Brightness threshold for preprocessing (0-255, default: 200)")
    
    args = parser.parse_args()
        
    # Ensure save directory exists
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup crop coordinates
    crop_coords = None
    if not args.no_crop:
        crop_coords = (args.crop_x1, args.crop_y1, args.crop_x2, args.crop_y2)
        print(f"Images will be cropped to: ({args.crop_x1}, {args.crop_y1}) to ({args.crop_x2}, {args.crop_y2})")
    
    # Start the WebSocket server
    print(f"Starting WebSocket server on {args.host}:{args.port}")
    print(f"Frames will be saved to {save_dir.absolute()}")
    
    if args.preprocess:
        print(f"Preprocessing enabled: Finding dark regions below threshold {args.threshold} using inverted binary image.")
    else:
        # Log the detection method being used when not preprocessing
        detection_method = "contour" if args.use_contours else "blob"
        print(f"Standard detection enabled using {detection_method} detection (finding bright spots/blobs).")
    
    async with websockets.serve(
        lambda ws: handle_connection(
            ws, save_dir, crop_coords, 
            args.use_contours, args.preprocess, args.threshold
        ), 
        args.host, 
        args.port,
        # Increase queue size and message size limits if needed
        # max_size=2**24, # Example: 16MB message limit
        # max_queue=128   # Example: Buffer up to 128 messages
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main()) 
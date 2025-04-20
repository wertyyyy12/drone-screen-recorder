#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import websockets


def detect_blobs(image):
    """
    Detect blobs in the image that could be people.
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Tuple of (image with keypoints drawn, keypoints)
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
    params.minThreshold = 180
    params.maxThreshold = 255
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1500
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Filter by Inertia (measure of elongation)
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(normalized)
    
    # Draw detected keypoints with rich information
    result_image = cv2.drawKeypoints(image, keypoints, np.array([]), 
                                     (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Draw bounding boxes around keypoints for better visibility
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        radius = size // 2
        cv2.rectangle(result_image, (x - radius, y - radius), 
                     (x + radius, y + radius), (0, 0, 255), 2)
    
    return result_image, keypoints


def detect_contours(image):
    """
    Detect people in thermal image using contours.
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Tuple of (image with contours drawn, contours)
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize the image to improve contrast
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply threshold to isolate potential people (bright spots in thermal)
    _, thresh = cv2.threshold(normalized, 180, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (to match human size)
    person_contours = []
    result_image = image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 1500:  # Area range for people
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio - people viewed from above typically have aspect ratio close to 1
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:
                person_contours.append(contour)
                
                # Draw contour and bounding box
                cv2.drawContours(result_image, [contour], 0, (0, 255, 0), 2)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return result_image, person_contours


async def save_frame(frame_data, frame_number, save_dir, crop_coords=None, detect_people=False, use_contours=False):
    """
    Save a received frame to the specified directory.
    
    Args:
        frame_data: Base64 encoded image data
        frame_number: Frame number from the video
        save_dir: Directory to save frames to
        crop_coords: Optional tuple (x1, y1, x2, y2) for cropping
        detect_people: Whether to apply blob detection for people
        use_contours: Whether to use contour detection instead of blob detection
    """
    # Decode the base64 image data
    img_data = base64.b64decode(frame_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Apply cropping if coordinates are provided
    if crop_coords and len(crop_coords) == 4:
        x1, y1, x2, y2 = crop_coords
        # Check if the crop coordinates are valid
        height, width = img.shape[:2]
        if 0 <= x1 < x2 and 0 <= y1 < y2 and x2 <= width and y2 <= height:
            img = img[y1:y2, x1:x2]
        else:
            print(f"Warning: Invalid crop coordinates for image of size {width}x{height}")
    
    # Apply detection if enabled
    detections = []
    if detect_people:
        if use_contours:
            img, detections = detect_contours(img)
            print(f"Detected {len(detections)} potential people using contours in frame {frame_number}")
        else:
            img, detections = detect_blobs(img)
            print(f"Detected {len(detections)} potential people using blobs in frame {frame_number}")
    
    # Create filename based on frame number
    filename = f"frame_{frame_number:06d}.jpg"
    filepath = save_dir / filename
    
    # Save the image
    cv2.imwrite(str(filepath), img)
    return filepath, detections


async def handle_connection(websocket, save_dir, crop_coords=None, detect_people=False, use_contours=False):
    """Handle incoming WebSocket connections."""
    client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"Client connected: {client_info}")
    
    try:
        async for message in websocket:
            # Parse the JSON message
            data = json.loads(message)
            frame_number = data.get("frame_number")
            frame_data = data.get("image_data")
            
            if not frame_data:
                print(f"Received message without image data from {client_info}")
                continue
            
            # Save the frame
            filepath, detections = await save_frame(
                frame_data, frame_number, save_dir, 
                crop_coords, detect_people, use_contours
            )
            print(f"Saved frame {frame_number} to {filepath}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed with {client_info}")
    except Exception as e:
        print(f"Error handling connection from {client_info}: {e}")


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
    parser.add_argument("--detect-people", action="store_true", help="Enable people detection")
    parser.add_argument("--use-contours", action="store_true", 
                        help="Use contour detection instead of blob detection (requires --detect-people)")
    
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
    
    if args.detect_people:
        if args.use_contours:
            print("People detection enabled using contour detection")
        else:
            print("People detection enabled using blob detection")
    
    async with websockets.serve(
        lambda ws: handle_connection(ws, save_dir, crop_coords, args.detect_people, args.use_contours), 
        args.host, 
        args.port
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main()) 
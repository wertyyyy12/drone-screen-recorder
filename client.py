#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import websockets


async def send_frames(websocket, video_path, crop_top, crop_bottom, frame_interval=25):
    """
    Extract frames from the video file and send them over the WebSocket connection.
    Only sends every Nth frame (determined by frame_interval).
    Crops the top and bottom of each frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if frame_count % frame_interval == 0:
                # Apply vertical cropping
                height = frame.shape[0]
                if crop_top + crop_bottom >= height:
                    raise ValueError("Invalid crop values: would result in empty frame")
                
                cropped_frame = frame[crop_top:height-crop_bottom, :]
                
                # Convert frame to JPEG format
                _, buffer = cv2.imencode('.jpg', cropped_frame)
                # Convert to base64 for sending
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare message with frame data and metadata
                message = {
                    "frame_number": frame_count,
                    "image_data": frame_data
                }
                
                # Send frame over WebSocket
                await websocket.send(json.dumps(message))
                print(f"Sent frame {frame_count}")
            
            frame_count += 1
            
    finally:
        cap.release()
        print(f"Processed {frame_count} frames, sent {frame_count // frame_interval} frames")


async def main():
    parser = argparse.ArgumentParser(description="Send video frames over WebSocket")
    parser.add_argument("video_path", type=str, help="Path to the MP4 file")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--crop-top", type=int, default=0, help="Number of pixels to crop from the top")
    parser.add_argument("--crop-bottom", type=int, default=0, help="Number of pixels to crop from the bottom")
    parser.add_argument("--frame-interval", type=int, default=25, help="Send every Nth frame")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    websocket_url = f"ws://{args.host}:{args.port}"
    print(f"Connecting to WebSocket server at {websocket_url}")
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print(f"Connected to {websocket_url}")
            await send_frames(
                websocket, 
                video_path,
                args.crop_top,
                args.crop_bottom,
                args.frame_interval
            )
    except websockets.exceptions.ConnectionClosedError:
        print(f"Failed to connect to WebSocket server at {websocket_url}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 
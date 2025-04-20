#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import websockets


async def send_frames(websocket, video_path, crop_top, crop_bottom, frame_interval=25, target_frames=None):
    """
    Extract frames from the video file, send them over the WebSocket connection,
    and wait for acknowledgment before sending the next frame.
    Sends every Nth frame OR specific frames if target_frames is provided.
    Crops the top and bottom of each frame.
    
    Args:
        websocket: The WebSocket connection object.
        video_path: Path to the video file.
        crop_top: Pixels to crop from the top.
        crop_bottom: Pixels to crop from the bottom.
        frame_interval: Interval for sending frames if target_frames is None.
        target_frames: Optional set of specific frame numbers to send.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    sent_frame_counter = 0
    processed_target_frames = set()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Determine if this frame should be processed
            should_process = False
            if target_frames is not None:
                if frame_count in target_frames:
                    should_process = True
                    processed_target_frames.add(frame_count) 
            elif frame_count % frame_interval == 0:
                should_process = True
                
            if should_process:
                # Apply vertical cropping
                height = frame.shape[0]
                if crop_top + crop_bottom >= height:
                    print(f"Error: Invalid crop values ({crop_top}, {crop_bottom}) for frame height {height}. Stopping.")
                    break
                
                cropped_frame = frame[crop_top:height-crop_bottom, :]
                
                # Convert frame to JPEG format
                ret_encode, buffer = cv2.imencode('.jpg', cropped_frame)
                if not ret_encode:
                    print(f"Warning: Failed to encode frame {frame_count}")
                    frame_count += 1
                    continue
                    
                # Convert to base64 for sending
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare message with frame data and metadata
                message = {
                    "frame_number": frame_count,
                    "image_data": frame_data
                }
                
                # Send frame over WebSocket
                try:
                    await websocket.send(json.dumps(message))
                    print(f"Sent frame {frame_count}")
                    sent_frame_counter += 1
                    
                    # Wait for acknowledgment
                    try:
                        ack_message = await asyncio.wait_for(websocket.recv(), timeout=10.0) # 10 second timeout
                        ack_data = json.loads(ack_message)
                        
                        if ack_data.get("status") == "processed" and ack_data.get("frame_number") == frame_count:
                            print(f"Received acknowledgment for frame {frame_count}")
                        elif ack_data.get("status") == "error":
                            print(f"Server error for frame {frame_count}: {ack_data.get('message', 'Unknown error')}")
                            # Decide whether to continue or stop
                            # break # Example: Stop on server error
                        else:
                            print(f"Warning: Received unexpected acknowledgment: {ack_data}")
                            # Decide how to handle this - maybe retry or stop?
                            # break # Example: Stop on unexpected ACK
                            
                    except asyncio.TimeoutError:
                        print(f"Error: Timeout waiting for acknowledgment for frame {frame_count}. Stopping.")
                        break # Stop sending frames on timeout
                    except json.JSONDecodeError:
                        print("Error: Received invalid JSON acknowledgment. Stopping.")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        print("Error: Connection closed while waiting for acknowledgment. Stopping.")
                        break
                    except Exception as e:
                        print(f"Error receiving/processing acknowledgment for frame {frame_count}: {e}. Stopping.")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print(f"Error: Connection closed before sending frame {frame_count}. Stopping.")
                    break
                except Exception as e:
                    print(f"Error sending frame {frame_count}: {e}. Stopping.")
                    break
            
            frame_count += 1
            
            # If we are only sending specific frames, check if we have sent all requested frames
            if target_frames is not None and len(processed_target_frames) == len(target_frames):
                 print(f"All requested frames ({len(target_frames)}) have been processed. Stopping.")
                 break
            
    finally:
        cap.release()
        total_frames_read = frame_count # frame_count increments one last time before exit
        if target_frames is not None:
            print(f"Finished sending specific frames. Read {total_frames_read} frames, sent {sent_frame_counter}/{len(target_frames)} requested frames.")
            missed_frames = target_frames - processed_target_frames
            if missed_frames:
                print(f"Warning: Did not find/process the following requested frames: {sorted(list(missed_frames))}")
        else:
             print(f"Finished sending interval frames. Read {total_frames_read} frames, sent {sent_frame_counter} frames (interval: {frame_interval}).")


async def main():
    parser = argparse.ArgumentParser(description="Send video frames over WebSocket")
    parser.add_argument("video_path", type=str, help="Path to the MP4 file")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--crop-top", type=int, default=0, help="Number of pixels to crop from the top")
    parser.add_argument("--crop-bottom", type=int, default=0, help="Number of pixels to crop from the bottom")
    parser.add_argument("--frame-interval", type=int, default=25, help="Send every Nth frame (ignored if --frames is used)")
    parser.add_argument("--frames", type=str, default=None, 
                        help="Comma-separated list of specific frame numbers to send (e.g., '100,200,300'). Overrides --frame-interval.")
    
    args = parser.parse_args()
    
    target_frames = None
    if args.frames:
        try:
            target_frames = set(int(f.strip()) for f in args.frames.split(',') if f.strip().isdigit())
            if not target_frames:
                 print("Error: No valid frame numbers provided in --frames argument.")
                 return
            print(f"Attempting to send specific frames: {sorted(list(target_frames))}")
        except ValueError:
            print("Error: Invalid format for --frames argument. Please use comma-separated integers.")
            return
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    websocket_url = f"ws://{args.host}:{args.port}"
    print(f"Connecting to WebSocket server at {websocket_url}")
    
    try:
        # Set higher connection timeout and ping interval/timeout
        async with websockets.connect(
            websocket_url, 
            open_timeout=10,
            ping_interval=20,
            ping_timeout=20
        ) as websocket:
            print(f"Connected to {websocket_url}")
            await send_frames(
                websocket, 
                video_path,
                args.crop_top,
                args.crop_bottom,
                args.frame_interval,
                target_frames # Pass the set of target frames
            )
    except websockets.exceptions.InvalidURI:
         print(f"Error: Invalid WebSocket URI: {websocket_url}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        print(f"Error: Connection refused by server at {websocket_url}")
    except asyncio.TimeoutError:
        print(f"Error: Connection timed out to {websocket_url}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 
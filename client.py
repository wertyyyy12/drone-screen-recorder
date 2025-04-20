#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import websockets
from websockets.connection import State  # Import State enum


async def send_frames(websocket, video_path, crop_top, crop_bottom, frame_interval=25, target_frames=None):
    """
    Extract frames from the video file, send them over the established WebSocket connection,
    and wait for acknowledgment before sending the next frame.
    Sends every Nth frame OR specific frames if target_frames is provided.
    Crops the top and bottom of each frame.
    
    Args:
        websocket: The established WebSocket connection object.
        video_path: Path to the video file.
        crop_top: Pixels to crop from the top.
        crop_bottom: Pixels to crop from the bottom.
        frame_interval: Interval for sending frames if target_frames is None.
        target_frames: Optional set of specific frame numbers to send.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}. Exiting handler.")
        # Send error message to server?
        return # Exit this handler if video fails
    
    frame_count = 0
    sent_frame_counter = 0
    processed_target_frames = set()
    
    try:
        while cap.isOpened():
            # Check if the server connection is still open before reading frame
            # Use websocket.state for checking connection status
            if websocket.state == State.CLOSED or websocket.state == State.CLOSING:
                print("Server connection closed or closing. Stopping frame sending.")
                break
                
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
                ret_encode, buffer = cv2.imencode('.jpg', cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                # print how big the jpeg image is
                print(f"JPEG image size: {len(buffer)} bytes")
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
                    # Check connection again before sending
                    if websocket.state != State.OPEN:
                         print(f"Connection not open before sending frame {frame_count}. Stopping.")
                         break
                         
                    await websocket.send(json.dumps(message))
                    await asyncio.sleep(args.wait_time)
                    sent_frame_counter += 1
                    
                    # Wait for acknowledgment
                    try:
                        # Increased timeout slightly
                        ack_message = await asyncio.wait_for(websocket.recv(), timeout=20.0) 
                        ack_data = json.loads(ack_message)
                        
                            
                    except asyncio.TimeoutError:
                        print(f"Error: Timeout waiting for acknowledgment for frame {frame_count}. Stopping.")
                        break # Stop sending frames on timeout
                    except json.JSONDecodeError:
                        print(f"Error: Received invalid JSON acknowledgment: {ack_message}. Stopping.")
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
        
        # Try to close the connection gracefully from the client side after finishing
        # Use websocket.state for checking
        if websocket.state == State.OPEN:
            print("Closing connection from client side.")
            await websocket.close()

# Define global arguments to be accessible in the handler
args = None
target_frames_set = None

async def connection_handler(websocket):
    """Handles a single incoming connection from the server."""
    global args, target_frames_set
    server_addr = websocket.remote_address
    print(f"Connection established with server: {server_addr}")
    try:
        # Start sending frames using the established connection and global args
        await send_frames(
            websocket, 
            args.video_path,
            args.crop_top,
            args.crop_bottom,
            args.frame_interval,
            target_frames_set # Use the parsed set of target frames
        )
    except Exception as e:
        print(f"Error during send_frames for connection {server_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Finished handling connection from {server_addr}")
        # Connection should be closed by send_frames or if an error occurred
        # Use websocket.state for checking
        if websocket.state == State.OPEN:
             print("Attempting final close from handler.")
             await websocket.close()


async def main():
    global args, target_frames_set
    parser = argparse.ArgumentParser(description="Listen for server connection and send video frames.")
    parser.add_argument("video_path", type=str, help="Path to the MP4 file")
    parser.add_argument("--listen-host", type=str, default="0.0.0.0", help="Host address to listen on (default: 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--crop-top", type=int, default=0, help="Number of pixels to crop from the top")
    parser.add_argument("--crop-bottom", type=int, default=0, help="Number of pixels to crop from the bottom")
    parser.add_argument("--frame-interval", type=int, default=25, help="Send every Nth frame (ignored if --frames is used)")
    parser.add_argument("--frames", type=str, default=None, 
                        help="Comma-separated list of specific frame numbers to send (e.g., '100,200,300'). Overrides --frame-interval.")
    parser.add_argument("--wait-time", type=float, default=1.0, help="Wait time between frames in seconds")
    
    args = parser.parse_args()
    
    target_frames_set = None
    if args.frames:
        try:
            target_frames_set = set(int(f.strip()) for f in args.frames.split(',') if f.strip().isdigit())
            if not target_frames_set:
                 print("Error: No valid frame numbers provided in --frames argument.")
                 return
            print(f"Will attempt to send specific frames: {sorted(list(target_frames_set))}")
        except ValueError:
            print("Error: Invalid format for --frames argument. Please use comma-separated integers.")
            return
    
    video_path_obj = Path(args.video_path)
    if not video_path_obj.exists():
        print(f"Error: Video file not found: {args.video_path}")
        return
    args.video_path = video_path_obj # Ensure it's a Path object

    print(f"Client waiting for connection from server on {args.listen_host}:{args.listen_port}")
    
    # Serve connections - connection_handler will be called for each incoming connection
    # This will run until the handler finishes (or forever if no connection)
    # Consider adding logic to stop the server after one successful handling if desired.
    stop_event = asyncio.Event() # For potential future graceful shutdown
    try:
        async with websockets.serve(connection_handler, args.listen_host, args.listen_port):
            await stop_event.wait() # Keep server running indefinitely (or until handler finishes/error)
    except OSError as e:
         print(f"Error starting client listener: {e}. Port likely in use.")
    except Exception as e:
        print(f"An unexpected error occurred running the client listener: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 

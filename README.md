# Video Frame Processing System

A system for extracting frames from MP4 files, sending them over WebSockets, and saving them on a server.

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Server

Start the WebSocket server. By default, it uses blob detection on original frames.

```
python server.py
```

Use contour detection instead of blob detection:
```
python server.py --use-contours
```

Use preprocessing to find dark regions (below threshold) and draw boxes:
```
python server.py --preprocess --threshold=200
```

Combine options:
```
python server.py --host=0.0.0.0 --port=8765 --save-dir=./output_frames --preprocess --threshold=180
```

Server parameters:
- `--host`: Server host address (default: localhost)
- `--port`: Server port number (default: 8765)
- `--save-dir`: Directory to save frames (default: ./files/frames)
- `--crop-x1`: Left coordinate for cropping (default: 108)
- `--crop-y1`: Top coordinate for cropping (default: 359)
- `--crop-x2`: Right coordinate for cropping (default: 984)
- `--crop-y2`: Bottom coordinate for cropping (default: 1527)
- `--no-crop`: Disable cropping entirely
- `--use-contours`: Use contour detection instead of the default blob detection (ignored if `--preprocess` is used).
- `--preprocess`: Apply inverted thresholding to find dark regions. Outputs bounding boxes for these regions drawn on the original image. Overrides `--use-contours`.
- `--threshold`: Brightness threshold for preprocessing (0-255, default: 200). Pixels below this become white in the inverted preprocessed image.

### Image Processing Modes

The server operates in one of two main modes:

1.  **Standard Detection Mode (Default)**:
    *   Runs either blob detection (default) or contour detection (`--use-contours`) on the original (or cropped) thermal image.
    *   Aims to find bright spots/shapes directly.
    *   Saves the original image with detections (keypoints/contours and bounding boxes) drawn on it (`frame_XXXXXX.jpg`).

2.  **Preprocessing Mode (`--preprocess`)**:
    *   Applies an *inverted* binary threshold (`--threshold`). Pixels **below** the threshold become white, others black.
    *   Saves this inverted binary image as `frame_XXXXXX_preprocessed_inv.jpg`.
    *   Finds connected white components (corresponding to the originally *dark* regions) in the inverted image using `cv2.connectedComponentsWithStats`.
    *   Draws blue bounding boxes around these detected regions on the *original* image.
    *   Saves the original image with these blue bounding boxes as the main output (`frame_XXXXXX.jpg`).
    *   The `--use-contours` flag is ignored in this mode.

### Client

Basic client usage:
```
python client.py video.mp4
```

With specific frames:
```
python client.py video.mp4 --frames=100,500,1250
```

With all options:
```
python client.py video.mp4 --host=localhost --port=8765 --crop-top=100 --crop-bottom=100 --frame-interval=25
```

Client parameters:
- `video_path`: Path to the MP4 file (required)
- `--host`: WebSocket server host (default: localhost)
- `--port`: WebSocket server port (default: 8765)
- `--crop-top`: Number of pixels to crop from the top of each frame (default: 0)
- `--crop-bottom`: Number of pixels to crop from the bottom of each frame (default: 0)
- `--frame-interval`: Send every Nth frame (ignored if --frames is used)
- `--frames`: Comma-separated list of specific frame numbers to send (e.g., '100,200,300'). Overrides --frame-interval. 
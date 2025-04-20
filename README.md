# Video Frame Processing System

A system where a client sends frames from an MP4 file over WebSockets to a server for processing and saving.

**Connection Flow:**

1.  The `client.py` script starts first, listening for incoming connections.
2.  The `server.py` script then connects to the waiting client.
3.  The client sends video frames to the server.
4.  The server processes the frames, saves results, and sends acknowledgments back.

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

**Important:** Start the `client.py` script *before* starting the `server.py` script.

### Client (Listens for Server Connection)

Start the client listener. It needs the video file path.

```
python client.py video.mp4
```

Specify listen host/port (if not default):
```
python client.py video.mp4 --listen-host=0.0.0.0 --listen-port=9000
```

Send specific frames:
```
python client.py video.mp4 --frames=100,500,1250
```

Client parameters:
- `video_path`: Path to the MP4 file (required).
- `--listen-host`: Host address for the client to listen on (default: 0.0.0.0).
- `--listen-port`: Port for the client to listen on (default: 8765).
- `--crop-top`: Number of pixels to crop from the top of each frame (default: 0).
- `--crop-bottom`: Number of pixels to crop from the bottom of each frame (default: 0).
- `--frame-interval`: Send every Nth frame (ignored if --frames is used).
- `--frames`: Comma-separated list of specific frame numbers to send. Overrides --frame-interval.

### Server (Connects to Client, Processes Frames)

Start the server, telling it where the client is listening.

```
python server.py --client-host=localhost --client-port=8765
```

Use contour detection instead of blob detection:
```
python server.py --client-host=localhost --use-contours
```

Use preprocessing to find dark regions (below threshold) and draw boxes:
```
python server.py --client-host=localhost --preprocess --threshold=200
```

Combine options:
```
python server.py --client-host=192.168.1.100 --save-dir=./output_frames --preprocess --threshold=180
```

Server parameters:
- `--client-host`: Host address of the listening client (required or use default: localhost).
- `--client-port`: Port the client is listening on (default: 8765).
- `--save-dir`: Directory to save processed frames (default: ./files/frames).
- `--crop-x1`: Left coordinate for cropping (default: 108).
- `--crop-y1`: Top coordinate for cropping (default: 359).
- `--crop-x2`: Right coordinate for cropping (default: 984).
- `--crop-y2`: Bottom coordinate for cropping (default: 1527).
- `--no-crop`: Disable cropping entirely.
- `--use-contours`: Use contour detection instead of the default blob detection (ignored if `--preprocess` is used).
- `--preprocess`: Apply inverted thresholding to find dark regions. Outputs bounding boxes for these regions drawn on the original image. Overrides `--use-contours`.
- `--threshold`: Brightness threshold for preprocessing (0-255, default: 200). Pixels below this become white in the inverted preprocessed image.

### Image Processing Modes (Server-Side)

The server operates in one of two main modes:

1.  **Standard Detection Mode (Default)**:
    *   Runs either blob detection (default) or contour detection (`--use-contours`) on the frames received from the client.
    *   Aims to find bright spots/shapes directly.
    *   Saves the original frame with detections (keypoints/contours and bounding boxes) drawn on it (`frame_XXXXXX.jpg`).

2.  **Preprocessing Mode (`--preprocess`)**:
    *   Applies an *inverted* binary threshold (`--threshold`) to received frames. Pixels **below** the threshold become white, others black.
    *   Saves this inverted binary image as `frame_XXXXXX_preprocessed_inv.jpg`.
    *   Finds connected white components (corresponding to the originally *dark* regions) in the inverted image.
    *   Draws blue bounding boxes around these detected regions on the *original* frame.
    *   Saves the original frame with these blue bounding boxes as the main output (`frame_XXXXXX.jpg`).
    *   The `--use-contours` flag is ignored in this mode. 
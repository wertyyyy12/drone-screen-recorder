# Video Frame Processing System

A system for extracting frames from MP4 files, sending them over WebSockets, and saving them on a server.

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Server

Start the WebSocket server to receive and save frames:

```
python server.py
```

With options:
```
python server.py --host=0.0.0.0 --port=8765 --save-dir=./files/frames --crop-x1=108 --crop-y1=359 --crop-x2=984 --crop-y2=1527 --detect-people
```

Or with contour detection:
```
python server.py --detect-people --use-contours
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
- `--detect-people`: Enable people detection in thermal images
- `--use-contours`: Use contour-based detection instead of blob detection (requires --detect-people)

### People Detection

The server can process thermal images to detect potential people using two different methods:

#### Blob Detection (Default)

When enabled with only `--detect-people` flag, the server will:
1. Apply specialized blob detection to identify heat signatures that likely represent people
2. Draw green circles around detected people (keypoints)
3. Draw red bounding boxes around each detection for better visibility

#### Contour Detection

When enabled with both `--detect-people` and `--use-contours` flags, the server will:
1. Apply thresholding and contour detection to identify heat signatures
2. Filter contours by area and aspect ratio to find potential people
3. Draw green contours around the detected shapes
4. Draw red bounding boxes around each detected person

This feature is optimized for thermal imagery taken from above, where people appear as warm shapes against a cooler background.

### Client

Basic client usage:
```
python client.py video.mp4
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
- `--frame-interval`: Send every Nth frame (default: 25) 
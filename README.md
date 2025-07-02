# Weapon Detection API using YOLOv8

This project implements a weapon detection system using the YOLOv8 object detection model. It provides a Flask API to detect weapons (specifically knives and pistols) in images, videos, and live camera frames.

## Features

*   **Object Detection:** Utilizes YOLOv8 for real-time weapon detection.
*   **Supported Media:**
    *   Image detection (`/detect/image`): Upload an image file (PNG, JPG, JPEG, GIF).
    *   Video detection (`/detect/video`): Upload a video file (MP4, AVI, MOV, MKV).
    *   Live frame detection (`/detect/live/frame`): Send individual frames for detection (e.g., from a live camera feed).
*   **Detected Classes:** Currently trained to detect:
    *   Knives
    *   Pistols
*   **Annotated Output:** Returns JSON responses with detection details (bounding boxes, confidence scores, counts) and saves annotated media (images/videos with drawn boxes and labels).
*   **Flask API:** Easy-to-use API endpoints for integration into other applications.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    **Important:** The `requirements.txt` file in this repository has encoding issues (extra null characters). Before running `pip install`, you **must** manually edit `requirements.txt` to remove these null characters, or the installation will likely fail. Each line should be a standard package specifier (e.g., `package_name==version`).
    Once fixed, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset Configuration (`data.yaml`):**
    The `data.yaml` file contains paths to the training and validation datasets. These are currently absolute paths.
    *   **For running inference with the pre-trained model:** This step might not be strictly necessary if you only use the `/detect/*` endpoints and the model is already trained.
    *   **For training a new model:** You **MUST** update the `train` and `val` paths in `data.yaml` to point to the correct locations of your image datasets on your local machine. Alternatively, modify them to be relative paths if your datasets are within the project structure.
    ```yaml
    # Example of what to update in data.yaml:
    train: path/to/your/train/images
    val: path/to/your/valid/images

    nc: 2 # Number of classes
    names: ["knife", "pistol"] # Class names
    ```

5.  **Pre-trained Model:**
    The application expects a pre-trained model weights file at `runs/detect/model/weights/last.pt`.
    *   If you have this file from a previous training run, ensure it's in the correct location.
    *   If you don't have this file, you'll need to train a model first (see "Training a Custom Model" section).

## Running the Application

1.  **Start the Flask Server:**
    Once the setup is complete and you have the pre-trained model weights (`last.pt`) in the correct directory (`runs/detect/model/weights/`), you can run the application:
    ```bash
    python app.py
    ```
    The server will typically start on `http://0.0.0.0:5000/`.

2.  **API Endpoints:**

    *   **`/detect/image`**
        *   **Method:** `POST`
        *   **Description:** Detects weapons in an uploaded image.
        *   **Form Data:** `image` (file) - The image file (e.g., `.jpg`, `.png`).
        *   **Example (using cURL):**
            ```bash
            curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/detect/image
            ```
        *   **Response:** JSON containing detection counts, path to the annotated image, and details of individual detections.

    *   **`/detect/video`**
        *   **Method:** `POST`
        *   **Description:** Detects weapons in an uploaded video.
        *   **Form Data:** `video` (file) - The video file (e.g., `.mp4`, `.avi`).
        *   **Example (using cURL):**
            ```bash
            curl -X POST -F "video=@/path/to/your/video.mp4" http://localhost:5000/detect/video
            ```
        *   **Response:** JSON containing total detection counts for the video and the path to the annotated video.

    *   **`/detect/live/frame`**
        *   **Method:** `POST`
        *   **Description:** Detects weapons in a single image frame (e.g., from a live camera stream).
        *   **Form Data:** `frame` (file) - The image frame file.
        *   **Example (using cURL):**
            ```bash
            curl -X POST -F "frame=@/path/to/your/frame.jpg" http://localhost:5000/detect/live/frame
            ```
        *   **Response:** JSON containing detection counts for the frame, path to the saved annotated frame, and details of individual detections.

    **Note on File Paths in Response:** The API response includes `annotated_file_path`. This path points to where the annotated media is saved *on the server*. If you are accessing the API remotely, you'll need to set up a way to retrieve these files (e.g., by serving the `detect/results` directory).

## Training a Custom Model

If you want to train the model on your own dataset or fine-tune the existing one:

1.  **Prepare Your Dataset:**
    *   Organize your images into `train/images` and `valid/images` directories.
    *   Create corresponding label files in `train/labels` and `valid/labels`.
    *   YOLO expects label files in `.txt` format, with one file per image. Each line in the text file should represent one bounding box: `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`.
    *   Coordinates should be normalized (between 0 and 1) relative to the image size.
    *   Refer to the [Ultralytics YOLO documentation](https://docs.ultralytics.com/datasets/) for detailed dataset preparation guidelines.

2.  **Update `data.yaml`:**
    Modify the `data.yaml` file to point to your dataset locations and specify your class names and number of classes (`nc`).
    ```yaml
    train: path/to/your/train/images  # Update this
    val: path/to/your/valid/images    # Update this

    nc: 2  # Number of classes - update if you have different classes
    names: ["class1", "class2"] # List of your class names - update
    ```
    For this project, the default classes are `["knife", "pistol"]`.

3.  **Run the Training Script:**
    Execute the `train_yolo.py` script:
    ```bash
    python train_yolo.py
    ```
    *   This script uses `yolov8n.pt` as the base model. You can change this in `train_yolo.py` if needed.
    *   Training arguments like epochs, image size (`imgsz`), batch size, etc., can also be modified in `train_yolo.py`.
    *   The trained model weights (including `last.pt` and `best.pt`) and other training artifacts will be saved in a new directory under `runs/detect/` (e.g., `runs/detect/guns_knives_model/weights/`).

4.  **Use the Trained Model:**
    After training, copy the desired weights file (e.g., `runs/detect/guns_knives_model/weights/last.pt` or `best.pt`) to `runs/detect/model/weights/last.pt` for the `app.py` to use it by default. Alternatively, you can update the `model_path` variable in `app.py` to point directly to your new model file.

## Project Structure

```
.
├── app.py                  # Main Flask application with API endpoints
├── train_yolo.py           # Script for training the YOLOv8 model
├── data.yaml               # YOLO dataset configuration (paths, classes)
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # Base YOLOv8 model weights (used for training)
├── runs/                   # Directory where YOLO training outputs are stored
│   └── detect/
│       └── model/          # Default directory for the trained model
│           └── weights/
│               ├── last.pt # Weights from the last epoch of training
│               └── best.pt # Weights with the best validation performance
├── detect/                 # Directory for media to be detected and results
│   ├── results/            # Stores annotated images, videos, and live frames
│   │   ├── images/
│   │   ├── videos/
│   │   └── live_frames/
│   └── ... (sample images/videos for testing)
├── train/                  # Training dataset
│   ├── images/
│   └── labels/
├── valid/                  # Validation dataset
│   ├── images/
│   └── labels/
├── test/                   # Test dataset (optional, for evaluation)
│   ├── images/
│   └── labels/
└── uploads/                # Temporary storage for uploaded videos before processing
```

## Dependencies

The main dependencies for this project are:

*   **Ultralytics YOLO:** For the object detection model and training.
*   **OpenCV (cv2):** For image and video processing.
*   **Flask:** For creating the web API.
*   **PyTorch:** As the backend for YOLOv8.
*   **NumPy:** For numerical operations.

A full list of Python packages and their versions can be found in `requirements.txt`. Make sure to install them as described in the "Setup and Installation" section.
Remember to manually fix the encoding of `requirements.txt` by removing null characters if you encounter issues during `pip install`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

(You can add more specific contribution guidelines here if needed, e.g., coding standards, testing procedures.)

## License

This project is currently unlicensed. You can add an open-source license file (e.g., MIT, Apache 2.0) if you wish to share it under specific terms.

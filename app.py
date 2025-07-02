from ultralytics import YOLO
import cv2
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import torch.nn.modules.container as container
import ultralytics.nn.modules.conv as ultralytics_conv
import torch.nn.modules.conv as torch_conv
import torch.nn.modules.batchnorm as batchnorm_modules
import torch.nn.modules.activation as activation_modules
import ultralytics.nn.modules.block as ultralytics_block
import torch.nn.modules.pooling as pooling_modules
import torch.nn.modules.upsampling as upsampling_modules
import ultralytics.nn.modules.head as ultralytics_head

import os
# from flask import Flask, request, jsonify, Response, 
from flask import Flask, request, jsonify, Response, render_template_string, send_from_directory, send_file

from flask_cors import CORS, cross_origin
import numpy as np
import io
import uuid
from werkzeug.utils import secure_filename
import base64

# --- GLOBALS & MODEL LOADING ---

# Add all necessary safe globals (from previous troubleshooting)
add_safe_globals([
    DetectionModel,
    container.Sequential,
    ultralytics_conv.Conv,
    torch_conv.Conv2d,
    batchnorm_modules.BatchNorm2d,
    activation_modules.SiLU,
    ultralytics_block.C2f,
    container.ModuleList,
    ultralytics_block.Bottleneck,
    ultralytics_block.SPPF,
    pooling_modules.MaxPool2d,
    upsampling_modules.Upsample,
    ultralytics_conv.Concat,
    ultralytics_head.Detect,
    ultralytics_block.DFL
])

model_path = "runs/detect/model/weights/last.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

print(f"Loading YOLO model from: {model_path}...")
model = YOLO(model_path)
print("YOLO model loaded successfully.")

gun_labels = ["pistol", "gun", "revolver"]
knife_labels = ["knife", "blade", "dagger"]

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = 'uploads' 
RESULTS_FOLDER = 'detect/results' 

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(RESULTS_FOLDER + '/images', exist_ok=True), os.makedirs(RESULTS_FOLDER + '/videos', exist_ok=True), os.makedirs(RESULTS_FOLDER + '/live_frames', exist_ok=True)


ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def _draw_detections_and_counts_on_frame(
    frame_to_annotate,
    detection_results, 
    knives_count,
    guns_count,
    total_weapons_count,
    model_names_map,
    gun_labels_list,
    knife_labels_list
):
    for result in detection_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = model_names_map.get(cls, f"class_{cls}") 

            cv2.rectangle(frame_to_annotate, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # Green box
            cv2.putText(frame_to_annotate, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
            
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255) # Red color for counts

    cv2.putText(frame_to_annotate, f"Total Weapons: {total_weapons_count}", (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_to_annotate, f"Knives: {knives_count}", (10, 70), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_to_annotate, f"Guns: {guns_count}", (10, 110), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return frame_to_annotate

@app.route("/image/<filename>")
def get_image(filename):
    path = f"detect/results/images/{filename}"
    return send_file(path, mimetype="image/png")

@app.route("/video/<filename>")
def get_video(filename):
    return send_from_directory(
        directory=os.path.join(os.getcwd(), RESULTS_FOLDER, 'videos'),
        path=filename,
        mimetype="video/mp4"
    )

@app.route('/detect/image', methods=['POST'])
def detect_image_endpoint():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": f"Unsupported image file type: {file.filename.rsplit('.', 1)[1].lower()}"}), 400

    try:
        np_image = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({"error": "Could not decode image"}), 400

        print(f"Received image: {file.filename} with dimensions: {original_image.shape[1]}x{original_image.shape[0]}")

        # Perform detection
        inference_img_size = 640
        results = model(original_image, imgsz=inference_img_size)

        knives_count = 0
        guns_count = 0
        individual_detections_data = [] 
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, f"class_{cls}")

                if label in knife_labels:
                    knives_count += 1
                elif label in gun_labels:
                    guns_count += 1
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].item()
                individual_detections_data.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2, y2]
                })

        weapon_count = knives_count + guns_count

        annotated_image = original_image.copy() 
        annotated_image = _draw_detections_and_counts_on_frame(
            annotated_image,
            results, 
            knives_count,
            guns_count,
            weapon_count,
            model.names,
            gun_labels,
            knife_labels
        )

        print(f"Image Detection Complete: Total Weapons={weapon_count}, Knives={knives_count}, Guns={guns_count}")

        unique_filename = f"{uuid.uuid4()}.png"
        image_save_path = RESULTS_FOLDER + '/images/' + unique_filename
        cv2.imwrite(image_save_path, annotated_image)
        print(f"Annotated image saved to: {image_save_path}")

        response_data = {
            "media_type": "image",
            "total_weapons": weapon_count,
            "knives_detected": knives_count,
            "guns_detected": guns_count,
            "unique_filename": unique_filename,
            "detections": individual_detections_data 
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"An error occurred during image detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect/video', methods=['POST'])
def detect_video_endpoint():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": f"Unsupported video file type: {file.filename.rsplit('.', 1)[1].lower()}"}), 400

    try:
        filename = secure_filename(file.filename)
        temp_video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 500

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_video_filename = f"{uuid.uuid4()}.mp4"
        output_video_path = RESULTS_FOLDER + '/videos/' + output_video_filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        print(f"Processing video: {total_frames} frames at {fps} FPS...")

        video_knives_total_count = 0
        video_guns_total_count = 0
        video_weapons_total_count = 0 

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            processed_frames += 1
            if processed_frames % (fps * 5) == 0: 
                print(f"Processed {processed_frames}/{total_frames} frames...")

            inference_img_size = 640 
            frame_results = model(frame, imgsz=inference_img_size)

            current_frame_knives = 0
            current_frame_guns = 0
            
            for result in frame_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    label = model.names.get(cls, f"class_{cls}")

                    if label in knife_labels:
                        current_frame_knives += 1
                    elif label in gun_labels:
                        current_frame_guns += 1
            
            video_knives_total_count += current_frame_knives
            video_guns_total_count += current_frame_guns
            
            current_frame_weapons = current_frame_knives + current_frame_guns
            video_weapons_total_count += current_frame_weapons 
            
            annotated_frame = _draw_detections_and_counts_on_frame(
                frame, 
                frame_results, 
                current_frame_knives, 
                current_frame_guns,
                current_frame_weapons, 
                model.names,
                gun_labels,
                knife_labels
            )

            out.write(annotated_frame) 

        cap.release()
        out.release()
        os.remove(temp_video_path) 
        print(f"Video Detection Complete. Annotated video saved to: {output_video_path}")

        response_data = {
            "media_type": "video",
            "total_knives_detected_in_video": video_knives_total_count,
            "total_guns_detected_in_video": video_guns_total_count,
            "total_weapons_detected_in_video": video_weapons_total_count,
            "unique_filename": output_video_filename,
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"An error occurred during video detection: {e}")
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({"error": str(e)}), 500

@app.route('/detect/live/frame', methods=['POST'])
def detect_live_frame():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame file provided"}), 400

    file = request.files['frame']
    if file.filename == '':
        return jsonify({"error": "No selected frame file"}), 400

    try:
        np_frame = np.frombuffer(file.read(), np.uint8)
        current_frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        if current_frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        inference_img_size = 640
        results = model(current_frame, imgsz=inference_img_size, verbose=False)

        knives_count = 0
        guns_count = 0
        detections_for_client = [] 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                label = model.names.get(cls, f"class_{cls}")

                if label in knife_labels:
                    knives_count += 1
                elif label in gun_labels:
                    guns_count += 1
                
                detections_for_client.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2, y2]
                })
        
        weapon_count = knives_count + guns_count

        annotated_frame = _draw_detections_and_counts_on_frame(
            current_frame.copy(),
            results,
            knives_count,
            guns_count,
            weapon_count,
            model.names,
            gun_labels,
            knife_labels
        )
        
        # --- NEW: SAVE THE ANNOTATED LIVE FRAME ---
        unique_filename = f"{uuid.uuid4()}.png"
        live_frame_save_path = os.path.join(RESULTS_FOLDER, 'live_frames', unique_filename)
        cv2.imwrite(live_frame_save_path, annotated_frame)
        print(f"Annotated live frame saved to: {live_frame_save_path}")

        # --- Encode the annotated frame to Base64 string for response ---
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            raise Exception("Could not encode annotated frame to JPEG.")
        
        response_data = {
            "total_weapons": weapon_count,
            "knives_detected": knives_count,
            "guns_detected": guns_count,
            "annotated_file_path": live_frame_save_path, 
            "detections": detections_for_client
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"An error occurred during live frame detection: {e}")
        return jsonify({"error": str(e)}), 500


# To run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
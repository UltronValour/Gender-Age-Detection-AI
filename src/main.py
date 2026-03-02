import argparse
import cv2
import os
import sys
import time
import csv
import math
from datetime import datetime
from collections import Counter

# Import custom modules
from utils import load_models, detect_faces, align_face
from inference import predict_age_gender

"""
Usage:
- Webcam mode (default):
    python src/main.py
- Image mode:
    python src/main.py --image path/to/image.jpg
- Folder mode:
    python src/main.py --folder path/to/images/
"""

def main():
    parser = argparse.ArgumentParser(description="Gender and Age Detection using OpenCV DNN")
    parser.add_argument("--image", type=str, help="Path to the input image. If omitted, webcam is used.")
    parser.add_argument("--folder", type=str, help="Path to a folder of images to process in batch.")
    args = parser.parse_args()

    # Load models from the relative models/ directory
    # The models directory is expected to be one level up from src/
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    print("Loading models...")
    try:
        face_net, age_net, gender_net = load_models(model_dir)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print(f"Ensure the 'models/' directory exists and contains the required pre-trained files.")
        sys.exit(1)

    # Automatically create directories for logging and saving outputs
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # CSV logging purpose: Track historical data and enable offline analysis
    # Initialize the CSV file with a header if it does not yet exist
    csv_file = os.path.join("logs", "predictions.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gender", "gender_conf", "age_bucket", "age_conf"])

    # Process based on input mode (Image or Webcam)
    if args.image:
        print(f"Running in Image mode: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not read image at {args.image}")
            sys.exit(1)
        
        process_frame(frame, face_net, age_net, gender_net)
        
        # Save output result containing the overlaid labels & bounds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", f"result_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Result saved to: {output_path}")
        
        # Display the output
        cv2.imshow("Gender and Age Prediction", frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.folder:
        print(f"Running in Folder mode: {args.folder}")
        if not os.path.isdir(args.folder):
            print(f"Error: {args.folder} is not a valid directory.")
            sys.exit(1)
            
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        processed_count = 0
        
        for filename in os.listdir(args.folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue
                
            img_path = os.path.join(args.folder, filename)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Warning: Could not read image {filename}. Skipping...")
                continue
                
            print(f"Processing {filename}...")
            process_frame(frame, face_net, age_net, gender_net)
            
            # Save the annotated output
            output_path = os.path.join("outputs", filename)
            cv2.imwrite(output_path, frame)
            processed_count += 1
            
        print(f"Folder processing complete. {processed_count} images saved to 'outputs/'")
        
    else:
        print("Running in Webcam mode. Press 'q' or close the window to exit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)
            
        frame_count = 0
        cached_results = []
        prev_time = time.time()
        
        # Dictionary to maintain face ID contexts
        # format: ID -> {"centroid": (x, y), "history": [(gender, age), ...], "last_seen": frame_count}
        trackers = {}
        next_face_id = 0
        
        # Latency tracking variables
        total_latency_ms = 0
        inference_count = 0
        avg_latency = 0
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break
            
            # FPS calculation using the time module
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Resize frame width to 750 for better face detail while keeping performance acceptable
            h, w = frame.shape[:2]
            new_w = 750
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))
            
            # FPS optimization: Process predictions only on every 2nd frame
            # This improves performance by reducing heavy compute overhead per frame
            if frame_count % 2 == 0:
                bboxes = detect_faces(face_net, frame)
                cached_results = []
                
                # We will update active trackers during inference cycles
                current_frame_faces = []
                
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    face_img = frame[max(0, y1):min(y2, frame.shape[0]-1), max(0, x1):min(x2, frame.shape[1]-1)]
                    
                    # Skip prediction if the face ROI is too small (<50 pixels)
                    # Small faces lack details causing inaccurate age/gender predictions
                    if face_img.size == 0 or face_img.shape[0] < 50 or face_img.shape[1] < 50:
                        continue
                        
                    # Conditionally apply face alignment to larger faces for better CNN accuracy
                    # Alignment makes eyes horizontal, which helps age/gender models trained on cropped aligned faces
                    if face_img.shape[0] >= 80 and face_img.shape[1] >= 80:
                        face_img = align_face(face_img)
                        
                    # Improve low-light robustness by brightening the face ROI before prediction
                    # This helps Caffe predict better in typical indoor room lighting
                    face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=15)
                        
                    # Predict age and gender and measure latency
                    start_time = time.time()
                    gender, gender_conf, age, age_conf = predict_age_gender(face_img, age_net, gender_net)
                    end_time = time.time()
                    
                    # Update running average latency
                    latency_ms = (end_time - start_time) * 1000
                    total_latency_ms += latency_ms
                    inference_count += 1
                    avg_latency = total_latency_ms / inference_count
                    
                    # Compute centroid for simple tracking
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    matched_id = None
                    min_dist = float('inf')
                    
                    # Find closest existing face (threshold: 70 pixels)
                    for fid, data in trackers.items():
                        dcx, dcy = data["centroid"]
                        dist = math.hypot(cx - dcx, cy - dcy)
                        if dist < 70 and dist < min_dist:
                            min_dist = dist
                            matched_id = fid
                            
                    if matched_id is None:
                        # Create new ID
                        matched_id = next_face_id
                        next_face_id += 1
                        trackers[matched_id] = {"centroid": (cx, cy), "history": [], "last_seen": frame_count}
                        
                    # Update tracker
                    trackers[matched_id]["centroid"] = (cx, cy)
                    trackers[matched_id]["last_seen"] = frame_count
                    trackers[matched_id]["history"].append((gender, age, gender_conf, age_conf))
                    
                    # Store last 5 predictions
                    if len(trackers[matched_id]["history"]) > 5:
                        trackers[matched_id]["history"].pop(0)
                        
                    # Compute majority vote over the history buffer
                    # Temporal smoothing improves real-time stability by preventing intermittent false predictions
                    # from causing the text label to flicker back and forth between frames.
                    history = trackers[matched_id]["history"]
                    genders = [item[0] for item in history]
                    ages = [item[1] for item in history]
                    
                    smooth_gender = Counter(genders).most_common(1)[0][0]
                    smooth_age = Counter(ages).most_common(1)[0][0]
                    
                    # Use the latest confidence for display
                    label = f"{smooth_gender} ({gender_conf:.2f}), {smooth_age} ({age_conf:.2f})"
                    cached_results.append((bbox, label))
                    
                    # Append row cleanly avoiding negative impact
                    with open(os.path.join("logs", "predictions.csv"), mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), smooth_gender, f"{gender_conf:.4f}", smooth_age, f"{age_conf:.4f}"])
                        
                # Cleanup dormant face IDs (disappeared for > 10 frames)
                trackers = {fid: data for fid, data in trackers.items() if frame_count - data["last_seen"] <= 10}
            
            # Ensure bounding boxes and labels persist smoothly between frames
            # Even on skipped frames (counter % 2 != 0), we still draw previous predictions
            for bbox, label in cached_results:
                x1, y1, x2, y2 = bbox
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Prepare label text and draw it
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Display FPS properly in the top-left corner
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Faces: {len(cached_results)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Avg latency: {int(avg_latency)} ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display results
            cv2.imshow("Gender and Age Prediction", frame)
            
            # Break loop on 'q' or when the window is closed using 'X'
            try:
                # getWindowProperty throws an exception or returns < 1 if closed depending on OS
                if cv2.getWindowProperty("Gender and Age Prediction", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # increment counter each frame
            frame_count += 1
                
        cap.release()
        cv2.destroyAllWindows()

def process_frame(frame, face_net, age_net, gender_net):
    """
    Detect faces in the frame, safely extract them, run inference,
    and draw the bounding boxes and labels on the frame in place.
    """
    # Detect faces
    bboxes = detect_faces(face_net, frame)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # Safe extraction: Get the face Region of Interest (ROI)
        # We already ensured coordinates were clipped and x2>x1, y2>y1 in detect_faces
        face_img = frame[max(0, y1):min(y2, frame.shape[0]-1), max(0, x1):min(x2, frame.shape[1]-1)]
        
        # Skip prediction if the face ROI is too small
        if face_img.size == 0 or face_img.shape[0] < 50 or face_img.shape[1] < 50:
            continue
            
        # Conditionally apply face alignment to larger faces for better CNN accuracy
        if face_img.shape[0] >= 80 and face_img.shape[1] >= 80:
            face_img = align_face(face_img)
            
        # Predict age and gender
        gender, gender_conf, age, age_conf = predict_age_gender(face_img, age_net, gender_net)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text and draw it
        label = f"{gender} ({gender_conf:.2f}), {age} ({age_conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Automatically append to log output
        with open(os.path.join("logs", "predictions.csv"), mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gender, f"{gender_conf:.4f}", age, f"{age_conf:.4f}"])

if __name__ == "__main__":
    main()

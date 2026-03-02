import cv2
import os
import math
import numpy as np

def load_models(model_dir="models"):
    """
    Load the pre-trained face, age, and gender networks using OpenCV DNN.
    Assumes relative paths to the model_dir.
    """
    # Face detection models
    face_proto = os.path.join(model_dir, "opencv_face_detector.pbtxt")
    face_model = os.path.join(model_dir, "opencv_face_detector_uint8.pb")
    
    # Age prediction models
    age_proto = os.path.join(model_dir, "age_deploy.prototxt")
    age_model = os.path.join(model_dir, "age_net.caffemodel")
    
    # Gender prediction models
    gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
    gender_model = os.path.join(model_dir, "gender_net.caffemodel")
    
    # Load networks
    face_net = cv2.dnn.readNet(face_model, face_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    
    return face_net, age_net, gender_net

def detect_faces(face_net, frame, conf_threshold=0.7):
    """
    Detect faces in the frame using the face_net.
    Returns the bounding boxes for detected faces, safely clipped to the frame size.
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    # Create a blob from the frame (standard size for opencv face detector: 300x300)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    # Forward pass
    face_net.setInput(blob)
    detections = face_net.forward()
    
    bboxes = []
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Extract bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            
            # Ensure coordinates are safely within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width - 1, x2)
            y2 = min(frame_height - 1, y2)
            
            # Only add valid bounding boxes (width and height > 0)
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
            
    return bboxes

def align_face(face_img):
    """
    Aligns the face based on eye coordinates detected by MediaPipe Face Mesh.
    Face alignment improves age prediction because CNN models are typically 
    trained on front-facing, aligned faces. 
    """
    try:
        import mediapipe as mp
        try:
            import mediapipe.python.solutions.face_mesh as mp_face_mesh
        except AttributeError:
            mp_face_mesh = mp.solutions.face_mesh
            
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True) as face_mesh:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = face_img.shape[:2]
            
            # Use refine_landmarks=True eye centers (468 for left, 473 for right)
            left_eye = (int(landmarks[468].x * w), int(landmarks[468].y * h))
            right_eye = (int(landmarks[473].x * w), int(landmarks[473].y * h))
            
            # Compute rotational angle to make eyes horizontal
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Center of eyes
            center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_face = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_CUBIC)
            
            return aligned_face
    except Exception:
        # Do not crash if MediaPipe fails; return original face image
        pass
        
    return face_img

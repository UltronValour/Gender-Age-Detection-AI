import cv2

# Define constants for predictions
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

# Mean values required for the model blob
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def predict_age_gender(face_img, age_net, gender_net):
    """
    Predict age and gender from a cropped face image.
    Uses models loaded by cv2.dnn.
    Returns: gender_label, gender_confidence, age_label, age_confidence
    Confidence scores are useful to gauge prediction reliability.
    """
    # Create a blob of size 227x227 as expected by age and gender Caffe models
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
    
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]
    gender_conf = gender_preds[0][gender_idx]
    
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age = AGE_BUCKETS[age_idx]
    age_conf = age_preds[0][age_idx]
    
    return gender, gender_conf, age, age_conf

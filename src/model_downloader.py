import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """
    Downloads the required model files at runtime if they are not present 
    in the local 'models/' directory. This is crucial for Streamlit Cloud deployment
    where large model files cannot be stored in the github repository.
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Dictionary of filename to raw download URL
    # Using established raw github endpoints and mirrors for these standard models
    MODEL_URLS = {
        "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb",
        "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
        "age_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",
        "gender_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
        "gender_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel"
    }

    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            logger.info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        else:
            logger.debug(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_models()

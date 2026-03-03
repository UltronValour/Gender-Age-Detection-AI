import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import time

# Add src to path so we can import our modules cleanly
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.model_downloader import download_models
download_models()

from utils import load_models, detect_faces, align_face
from inference import predict_age_gender

# Initialize models (cache them so they are only loaded once)
@st.cache_resource
def get_models():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return load_models(model_dir)

def clear_logs():
    csv_file = os.path.join("logs", "predictions.csv")
    if os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            f.write("timestamp,gender,gender_conf,age_bucket,age_conf\n")

def main():
    # 1. PAGE CONFIG
    st.set_page_config(page_title="Gender & Age Detection AI", page_icon="📊", layout="wide")
    
    # 8. CUSTOM CSS STYLING
    st.markdown("""
        <style>
        /* Base typography and background */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Clean up Streamlit defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Constrain layout to prevent extreme stretching on ultra-wide screens */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px !important; 
        }
        
        /* Header styling with gradient */
        h1.main-title {
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.8rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 0rem !important;
            padding-bottom: 0.5rem !important;
        }
        
        .subtitle-text {
            text-align: center;
            font-size: 1.35rem;
            color: #64748B;
            margin-bottom: 3.5rem;
            font-weight: 500;
        }
        
        /* Style native Streamlit Metrics as beautiful Cards */
        div[data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(128, 128, 128, 0.2);
            text-align: center;
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #3B82F6 !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Prettify DataFrame default rendering */
        div[data-testid="stDataFrame"] > div {
            border-radius: 12px !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            overflow: hidden !important;
        }
        
        /* Mobile specific CSS to fix squished layouts */
        @media screen and (max-width: 768px) {
            h1.main-title {
                font-size: 2.5rem !important;
            }
            .subtitle-text {
                font-size: 1.1rem !important;
                margin-bottom: 2rem !important;
            }
            div[data-testid="metric-container"] {
                padding: 1rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.6rem !important;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.7rem !important;
                white-space: nowrap !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Gender & Age Detection AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Real-time Computer Vision Analytics</p>", unsafe_allow_html=True)
    
    # Load Models setup
    try:
        face_net, age_net, gender_net = get_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Initial paths and folders (9. SAFETY)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    csv_file = os.path.join("logs", "predictions.csv")
    last_output_image = os.path.join("outputs", "last_stream_capture.jpg")
    
    if not os.path.exists(csv_file):
        clear_logs()

    # 2. SIDEBAR CONTROLS
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Image uploader options
        st.subheader("Image Source")
        source_mode = st.radio("Select input mode:", ["Upload Image", "Live Camera"], label_visibility="collapsed")
        
        frame = None
        is_live_camera = False
        
        if source_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif source_mode == "Live Camera":
            is_live_camera = True
            camera_image = st.camera_input("Take a picture via Webcam")
            if camera_image is not None:
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.markdown("---")
        st.subheader("Inference Settings")
        enable_alignment = st.checkbox("Enable Face Alignment (MediaPipe)", value=True)
        enable_smoothing = st.checkbox("Enable Temporal Smoothing", value=False)
        conf_threshold = st.slider("Face Conf. Threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
        
        st.markdown("---")
        st.subheader("ℹ️ Model Info")
        st.markdown(f"""
        <div style='background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 8px; font-size: 0.9em; margin-bottom: 15px; border: 1px solid rgba(128, 128, 128, 0.2);'>
            <b>Model:</b> Adience Caffe<br>
            <b>Face Detector:</b> OpenCV DNN<br>
            <b>Alignment:</b> {'ON' if enable_alignment else 'OFF'}<br>
            <b>Smoothing:</b> {'ON' if enable_smoothing else 'OFF'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Data Management")
        if st.button("🗑️ Clear Local Logs", use_container_width=True):
            clear_logs()
            st.rerun()

    # Read CSV logs safely 
    df_logs = pd.DataFrame()
    try:
        # Explicitly define columns since previous data might be missing headers
        col_names = ["timestamp", "gender", "gender_conf", "age_bucket", "age_conf"]
        df_temp = pd.read_csv(csv_file, header=None)
        
        # If the first row is the header string, skip it
        if not df_temp.empty and str(df_temp.iloc[0, 0]).startswith("timestamp"):
            df_temp = df_temp.iloc[1:].reset_index(drop=True)
            
        df_temp.columns = col_names
        df_logs = df_temp
    except Exception:
        pass

    # Process Inference if an image is provided
    faces_data = []
    latency_ms = 0.0
    fps_val = 0.0
    
    if frame is not None:
        # 7. LOADING SPINNER
        with st.spinner("Running inference..."):
            inference_start = time.time()
            
            # Run Face detection with variable threshold
            bboxes = detect_faces(face_net, frame, conf_threshold=conf_threshold)
            
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                face_img = frame[max(0, y1):min(y2, frame.shape[0]-1), max(0, x1):min(x2, frame.shape[1]-1)]
                
                # Minimum size threshold for processing
                if face_img.size == 0 or face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue
                    
                # Alignment check
                if enable_alignment and face_img.shape[0] >= 80 and face_img.shape[1] >= 80:
                    face_img = align_face(face_img)
                    
                # Low-light Pre-processing
                face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=15)
                
                # Age Gender prediction
                gender, gender_conf, age, age_conf = predict_age_gender(face_img, age_net, gender_net)
                
                # Annotate Frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{gender} ({gender_conf:.2f}), {age} ({age_conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                faces_data.append({
                    "Gender": gender,
                    "Gender Conf": f"{gender_conf:.2f}",
                    "Age Window": age,
                    "Age Conf": f"{age_conf:.2f}"
                })
                
                # Log writing
                try:
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{gender},{gender_conf:.4f},{age},{age_conf:.4f}\n")
                except Exception:
                    pass
            
            # Post-inference execution metrics
            exec_time = time.time() - inference_start
            latency_ms = exec_time * 1000
            if exec_time > 0:
                fps_val = 1.0 / exec_time
                
            # Auto-save image to output buffer
            try:
                cv2.imwrite(last_output_image, frame)
            except Exception:
                pass
                
            # Hot reload the logs dataframe safely
            try:
                col_names = ["timestamp", "gender", "gender_conf", "age_bucket", "age_conf"]
                df_temp = pd.read_csv(csv_file, header=None)
                if not df_temp.empty and str(df_temp.iloc[0, 0]).startswith("timestamp"):
                    df_temp = df_temp.iloc[1:].reset_index(drop=True)
                df_temp.columns = col_names
                df_logs = df_temp
            except Exception:
                pass

    # 3. TOP METRIC CARDS
    st.markdown("### 📈 Live Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    
    total_faces = len(df_logs) if not df_logs.empty else 0
    m1.metric("Faces Processed", total_faces)
    
    avg_conf_str = "N/A"
    if not df_logs.empty and 'gender_conf' in df_logs.columns and 'age_conf' in df_logs.columns:
        valid_conf = df_logs[['gender_conf', 'age_conf']].dropna()
        if not valid_conf.empty:
            avg_all = valid_conf.values.mean()
            avg_conf_str = f"{avg_all:.1%}"
    m2.metric("Average Confidence", avg_conf_str)
    
    latency_str = f"{latency_ms:.1f} ms" if latency_ms > 0 else "N/A"
    m3.metric("Average Latency", latency_str)
    
    fps_str = "N/A"
    if frame is not None:
        # User defined: FPS (if webcam active, else show N/A)
        # Note: streamlit camera_input executes per picture, so "webcam active" is true if they just took a pic
        if is_live_camera:
            fps_str = f"{fps_val:.1f}"
    m4.metric("FPS (Backend)", fps_str)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. MAIN LAYOUT (TOP ROW)
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📷 Detection Result")
        
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            if faces_data:
                st.success(f"✅ Successfully detected {len(faces_data)} face(s)")
                st.dataframe(pd.DataFrame(faces_data), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No face detected in the image")
        else:
            st.info("👈 Please select an image source from the sidebar to begin.")

    with col2:
        st.markdown("### � Recent Activity Logs")
        
        # Recent logs table
        if df_logs is not None and not df_logs.empty:
            st.dataframe(df_logs.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No records in the database.")
            
    st.markdown("---")
    st.markdown("### 📊 Live Demographics")
    
    # 5. CHARTS LAYOUT (BOTTOM ROW)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Gender Distribution**")
        if df_logs is not None and not df_logs.empty and 'gender' in df_logs.columns:
            gender_counts = df_logs['gender'].value_counts()
            st.bar_chart(gender_counts, use_container_width=True)
        else:
            st.info("Awaiting enough data for Gender chart.")
            
    with chart_col2:
        st.markdown("**Age Demographics**")
        if df_logs is not None and not df_logs.empty and 'age_bucket' in df_logs.columns:
            age_counts = df_logs['age_bucket'].value_counts()
            st.bar_chart(age_counts, use_container_width=True)
        else:
            st.info("Awaiting enough data for Age chart.")

    # 6. DOWNLOAD BUTTONS
    with st.sidebar:
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        # Download CSV
        if not df_logs.empty:
            csv_data = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Logs",
                data=csv_data,
                file_name=f"predictions_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("Download CSV Logs", disabled=True, use_container_width=True)
            
        # Download Image
        if os.path.exists(last_output_image):
            with open(last_output_image, "rb") as file:
                img_bytes = file.read()
            st.download_button(
                label="Download Last Image",
                data=img_bytes,
                file_name=f"annotated_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.button("Download Last Image", disabled=True, use_container_width=True)

    # 7. FOOTER
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 2rem;'>"
        "Built with OpenCV • MediaPipe • Streamlit • Caffe DNN"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

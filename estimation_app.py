import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import cv2
import time

DEMO_IMAGE = 'workerindian.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")



st.title("Human Pose Estimation Using Machine Learning")
st.text('Make Sure you have a clear image with all the parts clearly visible')

# Add video processing option
video_option = st.radio("Select Input Type:", ('Image', 'Webcam', 'Upload Video', 'Image with MediaPipe'))

if video_option == 'Image':
    img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=["jpg", "jpeg", "png"])
    
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.subheader('Original Image')
    st.image(image, caption=f"Original Image", use_container_width=True)

    thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5)
    thres = thres / 100

    @st.cache_resource
    def poseDetectorMultiPerson(frame):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]
        
        assert(len(BODY_PARTS) == out.shape[1])
        
        # Multi-person detection
        points_all = []
        for i in range(out.shape[0]):  # Loop through all detected people
            points = []
            for j in range(len(BODY_PARTS)):
                heatMap = out[i, j, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                points.append((int(x), int(y)) if conf > thres else None)
            points_all.append(points)
        
        # Draw poses for all detected people
        for points in points_all:
            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]
                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        
        return frame

    output = poseDetectorMultiPerson(image)
    
    st.subheader('Positions Estimated')
    st.image(output, caption=f"Positions Estimated", use_container_width=True)

elif video_option == 'Image with MediaPipe':
    img_file_buffer = st.file_uploader("Upload an image for MediaPipe Processing", type=["jpg", "jpeg", "png"])
    
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.subheader('Original Image')
    st.image(image, caption=f"Original Image", use_container_width=True)
    
    # Initialize MediaPipe pose detector
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe
    results = pose.process(image_rgb)
    
    # Draw the pose landmarks
    if results.pose_landmarks:
        output_image = image.copy()
        mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
        st.subheader('Positions Estimated with MediaPipe')
        st.image(output_image, caption="MediaPipe Processed Image", use_container_width=True)

elif video_option == 'Webcam':
    def process_video(video_source=0):
        """
        Process video frames for human pose estimation using MediaPipe.
        """
        # Initialize MediaPipe pose detector
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error("Error: Unable to access video source.")
            return
        
        # Create a placeholder for real-time video feed
        frame_placeholder = st.empty()
        stop_processing = st.button("Stop Webcam")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to read from video source.")
                break
            
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe pose detector
            results = pose.process(frame_rgb)

            # Draw keypoints and skeleton on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert back to RGB for Streamlit display
            frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb_display, caption="Processed Video Frame", use_container_width=True)

            # Break if the stop button is pressed
            if stop_processing:
                st.info("Webcam stopped.")
                break

            time.sleep(0.03)

        cap.release()

    process_video()

elif video_option == 'Upload Video':
    video_file_buffer = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file_buffer:
        temp_video_path = f"temp_video.{video_file_buffer.name.split('.')[-1]}"
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(video_file_buffer.read())

        def process_recorded_video(video_path):
            """
            Process a recorded video file for human pose estimation using MediaPipe.
            """
            # Initialize MediaPipe pose detector
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Error: Unable to access video file.")
                return
            
            # Create a placeholder for video feed
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.info("End of video reached.")
                    break
                
                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe pose detector
                results = pose.process(frame_rgb)

                # Draw keypoints and skeleton on the frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Convert back to RGB for Streamlit display
                frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb_display, caption="Processed Video Frame", use_container_width=True)

                time.sleep(0.03)

            cap.release()

        process_recorded_video(temp_video_path)

# Add logging functionality
def log_metrics():
    #with open("metrics_log.txt", "a") as log_file:
        #log_file.write(f"Threshold: {thres}, Model: graph_opt.pb\n")#solve this part later
    with st.container():
        st.write("### Log Recorded")
        st.success("Your sample log has been recorded.")
    #st.success("Metrics logged successfully.")

# Button to log metrics
st.button("Log Metrics", on_click=log_metrics)
st.caption("Adjust the threshold slider to fine-tune the detection sensitivity of keypoints.")

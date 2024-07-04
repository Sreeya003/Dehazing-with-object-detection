import cv2
import numpy as np
import subprocess
import streamlit as st
import os

def dark_channel(im, size):
    if len(im.shape) == 2:  # If the image is single-channel
        dc = im.copy()
    else:
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(dc, kernel)

def atmospheric_light(im, dc):
    flat = dc.reshape(dc.shape[0] * dc.shape[1])
    flat = np.argsort(flat)
    idx = flat[int(len(flat) * 0.95)]
    return im[idx // im.shape[1], idx % im.shape[1]]

def transmission_estimate(im, al, size, omega, t0):
    epsilon = 1e-6  # Small value to avoid division by zero
    al = np.maximum(al, epsilon)  # Ensure atmospheric light is not zero

    t_b = 1 - omega * dark_channel(im[:, :, 0] / al[0], size)
    t_g = 1 - omega * dark_channel(im[:, :, 1] / al[1], size)
    t_r = 1 - omega * dark_channel(im[:, :, 2] / al[2], size)
    
    # Clip the values to avoid invalid transmissions
    t_b = np.clip(t_b, 0, 1)
    t_g = np.clip(t_g, 0, 1)
    t_r = np.clip(t_r, 0, 1)
    
    return cv2.min(cv2.min(t_b, t_g), t_r)

def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * im + mean_b

def recover(im, t, al, t0):
    res = np.empty_like(im)
    for ind in range(3):
        res[:, :, ind] = ((im[:, :, ind] - al[ind]) / np.maximum(t, t0)) + al[ind]
    return res

def dehaze(image, omega=0.95, win_size=15, eps=0.001, t0=0.1):
    dc = dark_channel(image, win_size)
    al = atmospheric_light(image, dc)
    transmission = transmission_estimate(image, al, win_size, omega, t0)
    refined_t = guided_filter(dark_channel(image, 15), transmission, 60, eps)
    result = recover(image.astype(np.float64), refined_t, al, t0)
    return np.uint8(np.clip(result, 0, 255))

def dehaze_video(input_video_path, output_video_path, omega=0.95, win_size=15, eps=0.001, t0=0.1):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    st_frame = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply dehazing to the frame
        dehazed_frame = dehaze(frame, omega, win_size, eps, t0)

       
        # Write the dehazed frame to the output video
        out.write(dehazed_frame)

        frame_count += 1
        st_frame.text(f"Processed frame {frame_count}/{num_frames}")

    # Release resources
    cap.release()
    out.release()

    print(f"All frames processed. Dehazed video saved at: {output_video_path}")
    return output_video_path

def main():
    st.title("Video Dehazing App")
    session_state = st.session_state
    uploaded_file = st.file_uploader("Upload a video", type=['mp4'])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        input_video_path = f"./{uploaded_file.name}"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_video_dir = './haze/results/dehazed'
        os.makedirs(output_video_dir, exist_ok=True)

        output_video_path = os.path.join(output_video_dir, 'dehazed_video.mp4')
        if not session_state.get("dehazed", False):  # Check if the video has been dehazed
            dehaze_video(input_video_path, output_video_path)
            session_state.dehazed = True  # Update the session state

        # Display the dehazed video
        if os.path.exists(output_video_path):
            st.write("Dehazed Video:")
            video_file = open(output_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
        else:
            st.write("Error: Dehazed video not found.")

        # Button to launch another Streamlit app (a.py)
        if st.button("Launch App Object detection"):
            subprocess.Popen(["streamlit", "run", "a.py"])
            st.write("App A is now running.")

if __name__ == "__main__":
    main()

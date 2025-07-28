import cv2
import os

def extract_frames(video_path, scene_timestamps, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, (start, _) in enumerate(scene_timestamps):
        frame_no = int(start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"scene_{i:03d}.jpg")
            cv2.imwrite(out_path, frame)
    cap.release()

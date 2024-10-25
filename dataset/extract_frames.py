import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_folder, interval):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"fps: {fps}")
    frame_interval = interval * fps

    with tqdm(total=total_frames, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            saved_frame_cnt = int(frame_count // frame_interval)
            if frame_count % frame_interval == 0:
                frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{saved_frame_cnt:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            frame_count += 1
            pbar.update(1)

    cap.release()
    return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos and generate captions.")
    parser.add_argument("--video_folder", default="data/raw_videos", help="Path to the video folder")
    parser.add_argument("--output_folder", default="data/frames_output", help="Folder to save extracted frames")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame extraction interval in seconds")
    args = parser.parse_args()

    # Collect all video files in the specified folder
    video_files = [os.path.join(args.video_folder, f) for f in os.listdir(args.video_folder) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]

    for video_path in video_files:
        print(f"Processing video: {video_path}")
        frames = extract_frames(video_path, args.output_folder, args.interval)

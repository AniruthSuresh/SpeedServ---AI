import cv2

def read_video(path):
    """
    Reads frames from a video file and returns a list of frames.
    """

    cap = cv2.VideoCapture(path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Could not open video at {path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Saves a sequence of frames as a video file.
    """
    if not output_video_frames:
        print("Error: No frames to save.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    if not out.isOpened():
        print(f"Error: Could not open output video writer for {output_video_path}")
        return

    for frame in output_video_frames:
        out.write(frame)

    out.release()

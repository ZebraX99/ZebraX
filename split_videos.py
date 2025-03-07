import cv2 as cv
import os

def split_video(video_path, output_folder):
    """
    Split a long video into about 20s clips and save them as videos
    :param video_path:
    :param output_folder:
    :return:
    """
    # Get video name
    video_name = os.path.basename(video_path).split(".")[0]

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video duration: {duration} seconds")

    # Split video into clips
    clip_duration = 20  # seconds
    clip_frame_count = int(clip_duration * fps)
    clip_count = frame_count // clip_frame_count
    print(f"Splitting video into {clip_count} clips...")

    for i in range(clip_count):
        clip_time_interval = (i * clip_duration, (i + 1) * clip_duration)
        clip_name = os.path.join(output_folder, f"{video_name}_clip_{clip_time_interval[0]}_{clip_time_interval[1]}.mp4")
        out = cv.VideoWriter(clip_name, cv.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), isColor=False)
        # fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Using MJPG encoder
        # out = cv.VideoWriter(clip_name, fourcc, fps, ((int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))), isColor=False)
        for j in range(clip_frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            # Change to gray
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # cv.imshow("frame", frame)
            # cv.waitKey(0)

            out.write(frame)

        out.release()
        print(f"Saved clip {clip_name}")

    cap.release()
    print("Done.")

if __name__ == '__main__':
    videos_folder = "andy"
    output_folder = "clip_videos"
    for video in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, video)
        split_video(video_path, output_folder)

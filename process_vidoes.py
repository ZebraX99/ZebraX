import os
import tqdm
import cv2 as cv
import numpy as np

def gamma_trans(input, gamma=2, eps=0 ):
    return 255. * (((input + eps)/255.) ** gamma)

def process_videos(videos_folder, output_folder, downsample_rate=2, target_size=(1280, 720), skip_frames_interval=2):
    """
     downsample videos to target_size and change to gray
    :param videos_folder:
    :param output_folder:
    :return:
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_name in os.listdir(videos_folder):
        # check mp4 file
        if not video_name.endswith(".mp4"):
            continue

        video_path = os.path.join(videos_folder, video_name)
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"Video duration: {duration} seconds")

        # Downsample and resize video
        width, height = target_size

        out = cv.VideoWriter(os.path.join(output_folder, video_name), cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

        print(f"Processing video {video_name}...")
        skip_count = 0
        for i in tqdm.tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break

            # # Skip frames
            # if skip_count == skip_frames_interval:
            #     skip_count = 0
            #     continue

            frame = cv.resize(frame, (width, height))
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # change to 0-1
            gray_frame = gray_frame / 255.0

            gray_frame = gamma_trans(gray_frame, gamma=0.865)
            gray_frame = np.clip(gray_frame*255, 0, 255).astype(np.uint8)

            # cv.imshow("frame", gray_frame)
            # cv.waitKey()

            out.write(gray_frame)

            # skip_count = skip_count + 1

        out.release()
        print(f"Saved processed video {video_name}")

        cap.release()
        print("Done.")


if __name__ == '__main__':
    videos_folder = "videos/unhealthy"
    output_folder = "processed_videos"
    target_size = (640, 360)
    process_videos(videos_folder, output_folder, target_size=target_size)
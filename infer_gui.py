import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import cv2
from PIL import Image, ImageTk

def select_folder():
    # Open folder selection dialog
    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        entry_path.delete(0, tk.END)  # Clear the input box
        entry_path.insert(0, folder_path)  # Insert the selected folder path

def run_script():
    # Get the folder path from the input box
    folder_path = entry_path.get()
    if not folder_path:
        messagebox.showwarning("Warning", "Please select a folder first!")
        return

    if not os.path.exists(folder_path):
        messagebox.showerror("Error", "The path of the folder does not exist!")
        return

    # Concatenate the path of the Python file to run
    script_path = 'infer.py'

    # Try to run infer.py and capture the output
    try:
        process = subprocess.Popen(
            ["python", script_path, folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        # Display the output results in the text box
        text_output.delete(1.0, tk.END)  # Clear the text box
        if stdout:
            text_output.insert(tk.END, f"Output results:\n{stdout}\n")
        if stderr:
            text_output.insert(tk.END, f"Error information:\n{stderr}\n")

        # Remove session name in the folder path
        folder_name = folder_path.split('/')[-1]
        video_name = folder_name.replace('session_', '')
        video_name = video_name + '_tracked.avi'
        video_file = os.path.join(folder_path, video_name)

        # If the script runs successfully, show a prompt
        if process.returncode == 0:
            # Assume the generated video file name is "output.mp4", adjust as needed
            if os.path.exists(video_file):
                play_video(video_file)
            else:
                text_output.insert(tk.END, "Generated video file output.mp4 not found\n")

        # If the script runs successfully, show a prompt
        if process.returncode == 0:
            messagebox.showinfo("Success", "infer.py ran successfully!")
        else:
            messagebox.showerror("Error", "An error occurred while running infer.py, please check the output window!")
    except Exception as e:
        messagebox.showerror("Error", f"An unknown error occurred:\n{e}")

video_capture = None  # 用于存储 cv2.VideoCapture 对象
video_running = False  # 用于控制视频更新循环

def play_video(video_path):
    # Open a new window for video playback
    # video_window = tk.Toplevel(root)
    # video_window.title("Video Playback")
    # video_window.geometry("800x600")
    #
    # Create a label to display video frames
    # video_label = tk.Label(video_window)
    # video_label.pack(fill="both", expand=True)

    global video_capture, video_running

    # 如果之前有运行中的视频，先释放资源
    if video_capture:
        video_capture.release()
        video_capture = None

    # Use OpenCV to open the video
    cap = cv2.VideoCapture(video_path)

    video_capture = cv2.VideoCapture(video_path)
    video_running = True  # 标记视频正在播放

    def update_frame():
        global video_running
        if not video_running:  # 如果视频已停止，则退出更新循环
            return

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image object
            img = Image.fromarray(frame)
            # Convert to ImageTk object
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            # Update frame every 30 milliseconds
            video_label.after(30, update_frame)
        else:
            # Release resources and close window after video playback
            video_running = False  # 视频播放结束
            cap.release()
            # video_window.destroy()

    update_frame()

def reset_output():
    global video_capture, video_running

    # 清空视频播放区域
    video_label.configure(image="")
    video_label.imgtk = None

    # 停止视频播放
    video_running = False  # 停止更新帧
    if video_capture:
        video_capture.release()  # 释放视频资源
        video_capture = None

    # 清空文本输出框
    text_output.delete(1.0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Run Python File")

# Set default window size
root.geometry("800x600")  # Width 800, height 600
root.minsize(800, 600)    # Set minimum window size

# Folder path input box and button
frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Folder Path:")
label.grid(row=0, column=0, padx=5, pady=5)

entry_path = tk.Entry(frame, width=40)
entry_path.grid(row=0, column=1, padx=5, pady=5)

btn_select = tk.Button(frame, text="Select Folder", command=select_folder)
btn_select.grid(row=0, column=2, padx=5, pady=5)

# Run button
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

btn_run = tk.Button(root, text="Run infer.py", command=run_script, width=20, bg="green", fg="white")
btn_run.pack(pady=10)

btn_reset = tk.Button(btn_frame, text="Reset", command=reset_output, width=20, bg="red", fg="white")
btn_reset.pack(side="left", padx=10)

# Main content area, divided into left and right parts
content_frame = tk.Frame(root)
content_frame.pack(fill="both", expand=True)

# Left: for video playback
video_frame = tk.Frame(content_frame, bg="black", width=500, height=600)
video_frame.pack(side="left", fill="both", expand=True)

video_label = tk.Label(video_frame, bg="white")
video_label.pack(fill="both", expand=True)

# Right: for displaying output results
output_frame = tk.Frame(content_frame, width=500, height=600)
output_frame.pack(side="right", fill="both", expand=True)

text_output = tk.Text(output_frame, wrap="word")
text_output.pack(side="left", fill="both", expand=True)

# Add scrollbar
scrollbar = tk.Scrollbar(output_frame, command=text_output.yview)
text_output.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Main loop
root.mainloop()
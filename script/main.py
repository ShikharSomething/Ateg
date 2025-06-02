import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from threading import Thread
from detect_kills import detect_kills
from extract_clips import extract_kill_clips
from sync_and_generate_video import generate_final_montage
import subprocess
import sys


class VideoEditorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ClipSyncAI")
        self.root.geometry("480x500")  # Set a fixed size or make it resizable
        self.root.iconbitmap("icon.ico")  # Optional: Add an icon for the window

        # Apply a style
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")  # You can change the theme to "alt", "clam", etc.

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Create frames to organize content
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        self.video_frame = ttk.LabelFrame(self.main_frame, text="Video Settings", padding="10")
        self.video_frame.pack(fill="x", pady=10)

        self.music_frame = ttk.LabelFrame(self.main_frame, text="Music Settings", padding="10")
        self.music_frame.pack(fill="x", pady=10)

        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output Settings", padding="10")
        self.output_frame.pack(fill="x", pady=10)

        self.status_frame = ttk.Frame(self.main_frame, padding="10")
        self.status_frame.pack(fill="x", pady=10)

        # Video File Selection
        self.video_path_label = ttk.Label(self.video_frame, text="Select Video File:")
        self.video_path_label.grid(row=0, column=0, sticky="w", pady=5)
        self.video_path_button = ttk.Button(self.video_frame, text="Browse", command=self.browse_video)
        self.video_path_button.grid(row=0, column=1, padx=5, pady=5)

        # Music File Selection
        self.music_path_label = ttk.Label(self.music_frame, text="Select Music File:")
        self.music_path_label.grid(row=0, column=0, sticky="w", pady=5)
        self.music_path_button = ttk.Button(self.music_frame, text="Browse", command=self.browse_music)
        self.music_path_button.grid(row=0, column=1, padx=5, pady=5)

        # Output File Location Selection
        self.output_path_label = ttk.Label(self.output_frame, text="Select Output File Location:")
        self.output_path_label.grid(row=0, column=0, sticky="w", pady=5)
        self.output_path_button = ttk.Button(self.output_frame, text="Browse", command=self.browse_save_location)
        self.output_path_button.grid(row=0, column=1, padx=5, pady=5)

        # Status Label
        self.status_label = ttk.Label(self.status_frame, text="Status: Ready to start", anchor='w')
        self.status_label.grid(row=0, column=0, sticky="w", pady=5)

        # Process Buttons
        self.detect_button = ttk.Button(self.status_frame, text="Detect Kills", command=self.detect_kills_step)
        self.detect_button.grid(row=1, column=0, padx=10, pady=5)

        self.extract_button = ttk.Button(self.status_frame, text="Extract Kill Clips", command=self.extract_clips_step)
        self.extract_button.grid(row=1, column=1, padx=10, pady=5)

        self.generate_button = ttk.Button(self.status_frame, text="Generate Final Video",
                                          command=self.generate_video_step)
        self.generate_button.grid(row=2, column=0, padx=10, pady=5)

        # Open Kill Clips Folder Button
        self.open_clips_button = ttk.Button(self.status_frame, text="Kill Clips", command=self.open_kill_clips_folder)
        self.open_clips_button.grid(row=2, column=1, padx=10, pady=5)

        # Initialize paths
        self.video_path = None
        self.music_path = None
        self.timestamps_file = None  # We'll set this dynamically after video selection
        self.clip_folder = "kill_clips"
        self.output_path = None

        # Check if kill timestamps exist
        self.check_timestamps_available()

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if self.video_path:
            self.video_path_label.config(text=f"Video File: {self.video_path}")
            # Set timestamps file name dynamically based on video file name
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            self.timestamps_file = f"{video_name}_kill_timestamps.txt"
            self.check_timestamps_available()

    def browse_music(self):
        self.music_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if self.music_path:
            self.music_path_label.config(text=f"Music File: {self.music_path}")

    def browse_save_location(self):
        """Open a save file dialog to choose location and file name for final video."""
        self.output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Files", "*.mp4")])
        if self.output_path:
            self.output_path_label.config(text=f"Output File: {self.output_path}")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def check_timestamps_available(self):
        """Check if the timestamps file exists and update status."""
        if self.timestamps_file and os.path.exists(self.timestamps_file):
            self.update_status(f"Kill timestamps found: {self.timestamps_file}")
        else:
            self.update_status("No kill timestamps found. Please run 'Detect Kills'.")

    def reset_files_and_folders(self):
        """Resets the clips folder but keeps the timestamps file intact."""
        # Delete the existing clips folder if it exists
        if os.path.exists(self.clip_folder):
            for file in os.listdir(self.clip_folder):
                file_path = os.path.join(self.clip_folder, file)
                os.remove(file_path)
            os.rmdir(self.clip_folder)

        # Create a fresh clips folder
        os.makedirs(self.clip_folder, exist_ok=True)

    def detect_kills_step(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return
        self.update_status("Detecting kills...")
        self.detect_button.config(state=tk.DISABLED)
        self.extract_button.config(state=tk.NORMAL)

        # Run kill detection in a separate thread
        def run_detection():
            try:
                detect_kills(self.video_path, self.timestamps_file)
                self.update_status("Kills detected successfully!")
                self.check_timestamps_available()
            except Exception as e:
                self.update_status(f"Error: {e}")
            finally:
                self.detect_button.config(state=tk.NORMAL)

        Thread(target=run_detection).start()

    def extract_clips_step(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return
        if not os.path.exists(self.timestamps_file):
            messagebox.showerror("Error", "No kill timestamps found. Run 'Detect Kills' first.")

        # Delete existing kill clips folder to reset the clips
        self.reset_files_and_folders()

        self.update_status("Extracting kill clips...")
        self.extract_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)

        # Run clip extraction in a separate thread
        def run_extraction():
            try:
                clip_paths = extract_kill_clips(self.video_path, self.timestamps_file)
                self.update_status(f"Extracted {len(clip_paths)} kill clips successfully!")
            except Exception as e:
                self.update_status(f"Error: {e}")
            finally:
                self.extract_button.config(state=tk.NORMAL)

        Thread(target=run_extraction).start()

    def generate_video_step(self):
        if not self.video_path or not self.music_path:
            messagebox.showerror("Error", "Please select both video and music files.")
            return
        if not self.output_path:
            messagebox.showerror("Error", "Please select a location to save the final video.")
            return

        self.update_status("Generating final video...")
        self.generate_button.config(state=tk.DISABLED)

        # Run video generation in a separate thread
        def run_generation():
            try:
                generate_final_montage(self.clip_folder, self.music_path, self.output_path)
                self.update_status("Final video generated successfully!")
            except Exception as e:
                self.update_status(f"Error: {e}")
            finally:
                self.generate_button.config(state=tk.NORMAL)

        Thread(target=run_generation).start()

    def open_kill_clips_folder(self):
        """Open the kill clips folder using the default file explorer."""
        if os.path.exists(self.clip_folder):
            if sys.platform == "win32":  # Windows
                subprocess.run(["explorer", self.clip_folder])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", self.clip_folder])
            elif sys.platform == "linux":  # Linux
                subprocess.run(["xdg-open", self.clip_folder])
        else:
            messagebox.showerror("Error", "Kill clips folder does not exist.")


# Run the GUI
def main():
    root = tk.Tk()
    app = VideoEditorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
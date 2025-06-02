import os
from moviepy.editor import VideoFileClip


def extract_kill_clips(video_path: str,
                       timestamps_file: str,
                       buffer_duration: float = 0.5,
                       max_gap: float = 0.5):
    """
    Extracts short clips (including audio) from `video_path` based on kill timestamps.
    Each group of timestamps that are within `max_gap` seconds of each other becomes one clip,
    with an extra `buffer_duration` added before the first timestamp and after the last timestamp.
    """
    # 1. Prepare output folder
    output_dir = "kill_clips"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Read all timestamps (in seconds) from the file
    with open(timestamps_file, "r") as f:
        # Each line should be a float, e.g. "12.34\n"
        timestamps = [float(line.strip()) for line in f if line.strip()]

    if not timestamps:
        print("No timestamps found, exiting.")
        return []

    timestamps.sort()

    # 3. Group timestamps that are within max_gap of each other
    grouped_timestamps = []
    current_group = [timestamps[0]]

    for t in timestamps[1:]:
        if t - current_group[-1] <= max_gap:
            current_group.append(t)
        else:
            grouped_timestamps.append(current_group)
            current_group = [t]
    grouped_timestamps.append(current_group)

    # 4. Load the full video once
    video = VideoFileClip(video_path)
    video_duration = video.duration  # in seconds

    clip_paths = []
    for idx, group in enumerate(grouped_timestamps):
        start_time = max(min(group) - buffer_duration, 0)
        end_time = min(max(group), video_duration)

        # MoviePy’s subclip uses (t_start, t_end) in seconds
        subclip = video.subclip(start_time, end_time)

        output_clip_path = os.path.join(output_dir, f"kill{idx + 1}.mp4")
        # Write out the clip with both video + audio
        # You can tweak bitrate/fps/etc. if needed; this is a reasonable default
        subclip.write_videofile(
            output_clip_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=video.fps,
            verbose=False,
            logger=None
        )

        print(f"Saved clip #{idx + 1}: {output_clip_path}")
        clip_paths.append(output_clip_path)

    video.reader.close()
    video.audio.reader.close_proc()
    return clip_paths


if __name__ == "__main__":
    video_file = "valorant.mp4"  # replace with your actual video file path
    timestamps_txt = "valorant_kill_timestamps.txt"  # replace with your timestamps file
    extract_kill_clips(
        video_file,
        timestamps_txt,
        buffer_duration=0.5,
        max_gap=0.5
    )
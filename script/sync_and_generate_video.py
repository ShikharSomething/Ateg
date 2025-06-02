import os
import subprocess
import pathlib
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, \
    concatenate_audioclips


def apply_audio_mixing_to_clip(video_clip, music_audio, start_time_in_music):
    """
    Apply audio mixing to a single clip:
    - First 0.5 seconds: 2x clip audio + 50% music
    - Rest: mute clip audio + normal music
    """
    intro_duration = 0.5
    clip_duration = video_clip.duration

    # Handle music duration - loop or extend if needed
    music_duration = music_audio.duration

    if start_time_in_music >= music_duration:
        # If we've exceeded music length, loop the music
        loop_start = start_time_in_music % music_duration
        music_for_clip = music_audio.subclip(loop_start, min(loop_start + clip_duration, music_duration))

        # If clip extends beyond one loop, we need to loop the music
        if clip_duration > (music_duration - loop_start):
            remaining_duration = clip_duration - (music_duration - loop_start)
            # Create a looped version of the music
            loops_needed = int(remaining_duration / music_duration) + 1
            extended_music = concatenate_audioclips([music_audio] * (loops_needed + 1))
            music_for_clip = extended_music.subclip(loop_start, loop_start + clip_duration)
    else:
        # Normal case - extract music segment
        music_end_time = min(start_time_in_music + clip_duration, music_duration)
        music_for_clip = music_audio.subclip(start_time_in_music, music_end_time)

        # If clip is longer than remaining music, loop the music
        if start_time_in_music + clip_duration > music_duration:
            remaining_duration = (start_time_in_music + clip_duration) - music_duration
            loops_needed = int(remaining_duration / music_duration) + 1
            extended_music = concatenate_audioclips([music_audio] * (loops_needed + 1))
            music_for_clip = extended_music.subclip(start_time_in_music, start_time_in_music + clip_duration)

    if video_clip.audio is not None:
        if clip_duration > intro_duration:
            # Split clip audio into intro and rest
            intro_clip_audio = video_clip.audio.subclip(0, intro_duration).volumex(4)  # 2x amplification

            # Split music into intro and rest
            intro_music = music_for_clip.subclip(0, intro_duration).volumex(0.1)  # 50% music
            rest_music = music_for_clip.subclip(intro_duration).volumex(0.7)  # Normal music

            # Combine intro sections
            intro_composite = CompositeAudioClip([intro_clip_audio, intro_music])

            # Concatenate intro and rest
            final_audio = concatenate_audioclips([intro_composite, rest_music])
        else:
            # If clip is shorter than 0.5 seconds, just use intro logic for whole clip
            final_audio = CompositeAudioClip([
                video_clip.audio.volumex(2.5),
                music_for_clip.volumex(0.5)
            ])
    else:
        # If no clip audio, just use music
        final_audio = music_for_clip.volumex(0.8)

    return video_clip.set_audio(final_audio)


def generate_final_montage(clips_folder: str, music_path: str, output_path: str):
    """
    Create a montage from existing clip files using vid_transition.py directly.

    Args:
        clips_folder: Folder containing the kill clips
        music_path: Path to the music file
        output_path: Path for the output video
    """
    # Get list of all clip files sorted by name
    clip_files = sorted([os.path.join(clips_folder, f) for f in os.listdir(clips_folder)
                         if f.endswith(".mp4")])

    if len(clip_files) < 2:
        print(f"Error: Need at least 2 video clips to create transitions. Found {len(clip_files)} clips.")
        return

    print(f"Found {len(clip_files)} clips to process")

    # Temporary folder for transition outputs
    work_dir = pathlib.Path("temp_transitions")
    work_dir.mkdir(exist_ok=True)

    # Load music once
    music_audio = AudioFileClip(music_path)
    print(f"Music duration: {music_audio.duration:.2f} seconds")

    # List to store all processed clips for final concatenation
    processed_clips = []
    current_music_time = 0.0

    # Load first clip and get FPS for frame calculations
    first_clip = VideoFileClip(clip_files[0])
    fps = first_clip.fps if first_clip.fps else 30  # Default to 30 FPS if not available
    frames_to_drop = 8
    time_to_drop = frames_to_drop / fps  # Convert frames to time in seconds

    # Process first clip (trimmed - remove LAST 8 frames to prepare for transition)
    first_clip_trimmed = first_clip.subclip(0, first_clip.duration - time_to_drop)
    first_clip_with_audio = apply_audio_mixing_to_clip(first_clip_trimmed, music_audio, current_music_time)
    processed_clips.append(first_clip_with_audio)
    current_music_time += first_clip_trimmed.duration

    # Process each pair of consecutive clips with vid_transition.py
    for i in range(len(clip_files) - 1):
        clip1 = clip_files[i]
        clip2 = clip_files[i + 1]

        transition_output = work_dir / f"transition_{i}_{i + 1}_merged.mp4"

        print(f"Creating transition between clip {i + 1} and clip {i + 2}...")

        # Select a transition type
        transition_types = ['rotation', 'zoom_in', 'zoom_out', 'translation', 'translation_inv']
        transition_type = transition_types[i % len(transition_types)]

        # Call vid_transition.py directly
        cmd = [
            "python", "vid_transition.py",
            "-i", clip1, clip2,
            "--animation", transition_type,
            "--num_frames", "8",
            "--max_brightness", "3",
            "--merge", "true",
            "--output", str(work_dir / f"transition_{i}_{i + 1}")
        ]

        try:
            subprocess.run(cmd, check=True)

            if transition_output.exists():
                # Add the transition with audio mixing
                transition_clip = VideoFileClip(str(transition_output))
                transition_with_audio = apply_audio_mixing_to_clip(transition_clip, music_audio, current_music_time)
                processed_clips.append(transition_with_audio)
                current_music_time += transition_clip.duration

                # Load the next clip
                next_clip = VideoFileClip(clip_files[i + 1])

                # For ALL clips after the first one: remove FIRST 8 frames (already used in transition)
                # For clips that aren't the last one: also remove LAST 8 frames (for next transition)
                if i < len(clip_files) - 2:
                    # Intermediate clip: remove first 8 frames AND last 8 frames
                    next_clip_trimmed = next_clip.subclip(time_to_drop, next_clip.duration - time_to_drop)
                else:
                    # Last clip: only remove first 8 frames (no more transitions after this)
                    next_clip_trimmed = next_clip.subclip(time_to_drop)

                next_clip_with_audio = apply_audio_mixing_to_clip(next_clip_trimmed, music_audio, current_music_time)
                processed_clips.append(next_clip_with_audio)
                current_music_time += next_clip_trimmed.duration



            else:
                print(f"Warning: Transition file {transition_output} was not created")
                # Fallback: add next clip with proper trimming
                next_clip = VideoFileClip(clip_files[i + 1])
                if i < len(clip_files) - 2:
                    next_clip_trimmed = next_clip.subclip(time_to_drop, next_clip.duration - time_to_drop)
                else:
                    next_clip_trimmed = next_clip.subclip(time_to_drop)

                next_clip_with_audio = apply_audio_mixing_to_clip(next_clip_trimmed, music_audio, current_music_time)
                processed_clips.append(next_clip_with_audio)
                current_music_time += next_clip_trimmed.duration


        except subprocess.CalledProcessError as e:
            print(f"Error creating transition: {e}")
            # Fallback: add next clip with proper trimming
            next_clip = VideoFileClip(clip_files[i + 1])
            if i < len(clip_files) - 2:
                next_clip_trimmed = next_clip.subclip(time_to_drop, next_clip.duration - time_to_drop)
            else:
                next_clip_trimmed = next_clip.subclip(time_to_drop)

            next_clip_with_audio = apply_audio_mixing_to_clip(next_clip_trimmed, music_audio, current_music_time)
            processed_clips.append(next_clip_with_audio)
            current_music_time += next_clip_trimmed.duration

    # Close the original first clip

    # Concatenate all processed clips
    print("Concatenating clips and transitions...")
    final_video = concatenate_videoclips(processed_clips, method="compose")

    # Write the final video
    print("Writing final montage...")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Clean up
    for clip in processed_clips:
        clip.close()
    music_audio.close()
    final_video.close()

    print(f"Final montage saved at: {output_path}")
    print(f"Montage resolution: {final_video.size}, Duration: {final_video.duration:.2f} seconds")

    # Clean up temporary files if needed
    # import shutil
    # shutil.rmtree(work_dir)


if __name__ == "__main__":
    # Example usage
    generate_final_montage(
        clips_folder="kill_clips",
        music_path="music.mp3",  # Replace with your music file
        output_path="final_montage.mp4"
    )
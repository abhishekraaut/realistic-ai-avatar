import os
import sys

try:
    from moviepy import VideoFileClip
except ImportError:
    print("moviepy not installed. Please install moviepy.")
    sys.exit(1)

def extract_audio(video_path, output_path, fps=24000):
    print(f"Extracting audio from {video_path}...")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        print("No audio track found in video!")
        sys.exit(1)
        
    clip.audio.write_audiofile(output_path, fps=fps, nbytes=2, buffersize=2000, codec='pcm_s16le')
    print(f"Saved audio to {output_path} at {fps}Hz.")

if __name__ == "__main__":
    video_path = r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets\how-i-ai.mp4"
    output_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "audio.wav")
    
    extract_audio(video_path, output_path)

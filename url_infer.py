import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
import tempfile
import os
import subprocess
from urllib.parse import urlparse


def is_url(string):
    """Check if the string is a URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_youtube_url(url):
    """Check if the URL is a YouTube URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc for domain in youtube_domains)


def download_video_from_url(url, temp_dir):
    """Download video from URL to a temporary file"""
    if is_youtube_url(url):
        try:
            temp_file = os.path.join(temp_dir, "youtube_video.mp4")
            
            # List available formats
            list_formats_command = [
                'yt-dlp',
                '--list-formats',
                url
            ]
            print("Listing available formats...")
            subprocess.run(list_formats_command)
            
            # Use yt-dlp command to download the video
            command = [
                'yt-dlp',
                '-f', '230',  # Replace '230' with the desired format IDe '230' with the desired format IDe '230' with the desired format ID
                '-o', temp_file,
                url
            ]
            
            print("Running yt-dlp to download video...")
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                print(f"Error downloading YouTube video: {process.stderr}")
                return None
                
            return temp_file
        except Exception as e:
            print(f"Error downloading YouTube video: {e}")
            return None
    else:
        # Handle direct video URLs
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filename = os.path.basename(urlparse(url).path) or "downloaded_video.mp4"
                temp_file = os.path.join(temp_dir, filename)
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                return temp_file
            else:
                print(f"Failed to download video: HTTP status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None


def run_inference_on_video(model_path, video_path, conf_threshold=0.25, save_path=None):
    """Run inference on a video file"""
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer if save_path is provided
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        start_time = time.time()
        results = model(frame, conf=conf_threshold)
        inference_time = time.time() - start_time
        
        total_time += inference_time
        frame_count += 1
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(annotated_frame, fps_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save or display the frame
        if save_path:
            writer.write(annotated_frame)
        else:
            cv2.imshow("Gun Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if save_path:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")


def main():
    # Set fixed paths directly in the script
    model_path = "/home/vmukti/Downloads/gun/Weapon_detection/Model/gun.pt"  # Path to your model
    video_url = "https://www.youtube.com/watch?v=W274l2WOcsE"  # URL of the video
    output_path = "results/output.mp4"  # Where to save the output
    confidence = 0.5  # Confidence threshold
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a temporary directory to store downloaded videos
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading video from {video_url}...")
        video_path = download_video_from_url(video_url, temp_dir)
        
        if video_path:
            print(f"Video downloaded to {video_path}")
            print("Running inference...")
            run_inference_on_video(model_path, video_path, conf_threshold=confidence, save_path=output_path)
        else:
            print("Failed to download video. Please check the URL and try again.")


if __name__ == "__main__":
    main()
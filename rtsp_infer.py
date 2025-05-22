import argparse
import os
from pathlib import Path
import cv2
import time
from ultralytics import YOLO

def run_inference_on_stream(model_path, stream_source, conf_threshold=0.25, save_path=None):
    """Run inference on any video stream (webcam, RTSP, etc.), with TCP + timeouts + auto-reconnect."""
    model = YOLO(model_path)

    # Force TCP (more reliable) and set OpenCV timeouts
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    open_timeout = 5000   # ms
    read_timeout = 2000   # ms

    def make_capture():
        cap = cv2.VideoCapture(stream_source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout)
        return cap

    cap = make_capture()
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream {stream_source}")
        return

    # Try to read properties (may return 0 on RTSP)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps    = cap.get(cv2.CAP_PROP_FPS)               or 25.0

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    print(f"[INFO] Streaming at {width}×{height} @ {fps:.1f} FPS")
    reconnect_delay = 1.0  # seconds between reconnect attempts

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed – attempting reconnect...")
            cap.release()
            time.sleep(reconnect_delay)
            cap = make_capture()
            if not cap.isOpened():
                print(f"[ERROR] Reconnect to {stream_source} failed. Giving up.")
                break
            else:
                print("[INFO] Reconnected successfully.")
            continue

        # Inference
        t0 = time.time()
        results = model(frame, conf=conf_threshold)
        inf_t = time.time() - t0

        annotated = results[0].plot()
        cv2.putText(
            annotated,
            f"FPS: {1/inf_t:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("RTSP Gun Detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Gun Detection (RTSP + reconnect)")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to your .pt model"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="RTSP URL (or local file/webcam)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Where to save video output"
    )

    args = parser.parse_args()
    src = args.source.lower()

    if src == "webcam":
        run_inference_on_stream(
            args.model, 0, conf_threshold=args.conf, save_path=args.output
        )
    elif src.startswith("rtsp://"):
        run_inference_on_stream(
            args.model, args.source, conf_threshold=args.conf, save_path=args.output
        )
    else:
        # fallback to image / file handling
        ext = Path(src).suffix.lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            from pathlib import Path as _P
            run_inference_on_video = globals()["run_inference_on_video"]
            run_inference_on_video(
                args.model, args.source, conf_threshold=args.conf, save_path=args.output
            )
        else:
            run_inference_on_image = globals()["run_inference_on_image"]
            run_inference_on_image(
                args.model, args.source, conf_threshold=args.conf, save_path=args.output
            )


if __name__ == "__main__":
    main()


# rtsp://admin:@ATPL-908604-AIPTZ.torqueverse.dev:8604/ch0_0.264
# rtsp://admin:@ATPL-909659-AIPTZ.torqueverse.dev:9659/ch0_0.264
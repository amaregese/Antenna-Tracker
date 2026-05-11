import sys
import time
import argparse
import cv2

import config
from vision.camera import initialize_capture
from vision.detector import Detector
from hardware.tracker import AntennaTracker
from ui.visualizer import TrackerVisualizer


def main():
    parser = argparse.ArgumentParser(description="AI Antenna Tracker")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera device index")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Detection confidence")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="YOLO model path")
    parser.add_argument("--port", type=str, default="COM3", help="Tracker serial port")
    parser.add_argument("--baud", type=int, default=57600, help="Serial baud rate")
    parser.add_argument("--pan-channel", type=int, default=1, help="Pan servo channel")
    parser.add_argument("--tilt-channel", type=int, default=2, help="Tilt servo channel")
    args = parser.parse_args()

    if args.conf_threshold is not None:
        config.CONFIDENCE_THRESHOLD = args.conf_threshold

    print("=" * 50)
    print("AI ANTENNA TRACKER")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Confidence: {config.CONFIDENCE_THRESHOLD}")
    print(f"Tracker port: {args.port} (baud={args.baud})")
    print(f"Pan ch:{args.pan_channel} Tilt ch:{args.tilt_channel}")
    print("=" * 50)

    cap = initialize_capture(args.camera_index)
    if cap is None:
        print("Failed to open camera")
        sys.exit(1)

    detector = Detector(args.model)
    tracker = AntennaTracker(args.port, args.baud, args.pan_channel, args.tilt_channel)
    visualizer = TrackerVisualizer()

    cv2.namedWindow(config.WINDOW_NAME)
    click_x, click_y = -1, -1

    def mouse_cb(event, x, y, flags, param):
        nonlocal click_x, click_y
        if event == cv2.EVENT_LBUTTONDOWN:
            click_x, click_y = x, y

    cv2.setMouseCallback(config.WINDOW_NAME, mouse_cb)

    print("\nControls: CLICK object to track | 'q' quit | 'c' clear selection")

    try:
        while True:
            detections, fps, frame = detector.process_frame(cap)
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            if click_x != -1 and click_y != -1:
                clicked = detector.find_object_at(detections, click_x, click_y)
                if clicked:
                    detector.select_object(clicked)
                else:
                    print("No object at click position")
                click_x, click_y = -1, -1

            selected = detector.get_selected()

            for obj in detections:
                is_sel = obj is selected
                color = config.COLOR_SELECTED if is_sel else config.COLOR_DEFAULT
                thickness = 2 if is_sel else 1
                cv2.rectangle(frame, (obj.Left, obj.Top), (obj.Right, obj.Bottom), color, thickness)
                label = f"{obj.class_name} ({obj.confidence:.0f}%)" if is_sel else obj.class_name
                cv2.putText(frame, label, (obj.Left, obj.Top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                if is_sel:
                    cv2.line(frame, (w // 2, h // 2), obj.Center, (0, 255, 0), 1)
                    cv2.circle(frame, obj.Center, 7, (0, 255, 0), -1)

            cv2.circle(frame, (w // 2, h // 2), 6, (255, 255, 255), -1)

            pan, tilt = 0.0, 0.0
            obj_norm = None
            is_tracking = False
            vconf = 0.0
            vname = None

            if selected is not None:
                x_delta = (selected.Center[0] - w / 2) / w
                y_delta = (selected.Center[1] - h / 2) / h
                pan = max(-config.MAX_PAN, min(config.MAX_PAN, x_delta * config.MAX_PAN * config.GAIN_PAN))
                tilt = max(-config.MAX_TILT, min(config.MAX_TILT, y_delta * config.MAX_TILT * config.GAIN_TILT))
                tracker.send_pan(pan)
                tracker.send_tilt(tilt)

                obj_norm = (x_delta * 2, y_delta * 2)
                loss = detector.get_tracking_status()
                is_tracking = not loss
                vconf = detector.get_tracking_confidence()
                vname = selected.class_name

                status_color = (0, 0, 255) if loss else (0, 255, 0)
                status_text = f"LOST {selected.class_name}" if loss else f"TRACKING {selected.class_name} ({vconf:.0f}%)"
                cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                cv2.putText(frame, f"Pan: {pan:.1f} Tilt: {tilt:.1f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Click object to track", (w // 2 - 100, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Detected: {len(detections)}", (w // 2 - 80, h - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(config.WINDOW_NAME, frame)
            visualizer.update(pan, tilt, obj_norm, is_tracking, vconf, fps, vname)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                detector.clear_selection()
                print("Selection cleared")

    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import config


def _find_working_camera():
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        skip_keywords = ["ndi", "virtual", "obs"]
        for i, name in enumerate(devices):
            lower = name.lower()
            if any(kw in lower for kw in skip_keywords):
                continue
            if "integrated" in lower or "camera" in lower:
                print(f"  Auto-detected camera: [{i}] {name}")
                return i
        for i, name in enumerate(devices):
            lower = name.lower()
            if not any(kw in lower for kw in skip_keywords):
                print(f"  Auto-detected camera: [{i}] {name}")
                return i
    except ImportError:
        print("  (pygrabber not available, using default camera index 0)")
    except Exception:
        pass
    return 0


def initialize_capture(camera_index=None):
    if camera_index is None:
        camera_index = _find_working_camera()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Camera not found at index {camera_index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DEFAULT_HEIGHT)
    print(f"Camera opened (index {camera_index})")
    return cap


def read_frame(cap):
    if cap is None or not cap.isOpened():
        return False, None
    return cap.read()

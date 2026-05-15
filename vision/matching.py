import math

import cv2
import numpy as np


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def get_roi(frame, obj):
    if frame is None or obj is None or obj.area <= 0:
        return None
    roi = frame[obj.Top:obj.Bottom, obj.Left:obj.Right]
    if roi.size == 0:
        return None
    return roi


def compute_color_histogram(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_color_histogram_2d(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [64, 64], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_color_histogram_3d(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_color_signature(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    n_clusters = min(3, len(pixels))
    if n_clusters < 1:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten()).astype(np.float32)
    weights = (counts / counts.sum()).tolist()
    order = np.argsort(-np.array(weights))
    return {
        "centers": centers[order].tolist(),
        "weights": [weights[i] for i in order],
    }


def compare_color_signatures(sig1, sig2):
    if sig1 is None or sig2 is None:
        return 0.0
    centers1 = np.array(sig1["centers"])
    centers2 = np.array(sig2["centers"])
    if len(centers1) == 0 or len(centers2) == 0:
        return 0.0
    weights1 = np.array(sig1["weights"])
    score = 0.0
    hsv_ranges = np.array([180.0, 256.0, 256.0])
    for i, c1 in enumerate(centers1):
        dists = np.sum(np.abs(centers2 - c1) / hsv_ranges, axis=1)
        best_sim = max(0.0, 1.0 - np.min(dists) * 2.5)
        score += best_sim * weights1[i]
    return float(min(1.0, score))


def get_object_contour(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def match_shapes_contour(contour1, contour2):
    if contour1 is None or contour2 is None:
        return 0.0
    dist = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)
    return float(max(0.0, 1.0 - dist * 0.5))


def compute_region_histograms(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None, None, None, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    mid_y = h // 2
    mid_x = w // 2
    top = hsv[:mid_y, :]
    bottom = hsv[mid_y:, :]
    left = hsv[:, :mid_x]
    right = hsv[:, mid_x:]
    hist_top = cv2.calcHist([top], [0], None, [64], [0, 180])
    hist_bottom = cv2.calcHist([bottom], [0], None, [64], [0, 180])
    hist_left = cv2.calcHist([left], [0], None, [64], [0, 180])
    hist_right = cv2.calcHist([right], [0], None, [64], [0, 180])
    for h in (hist_top, hist_bottom, hist_left, hist_right):
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return hist_top, hist_bottom, hist_left, hist_right


def compute_hu_moments(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def compute_contour_features(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / max(hull_area, 1)
    extent = contour_area / max(obj.area, 1)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * math.pi * contour_area) / max(perimeter * perimeter, 1)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
    complexity = len(approx)
    return {
        "solidity": solidity,
        "extent": extent,
        "circularity": circularity,
        "complexity": complexity,
        "contour_area": contour_area,
    }


def compute_mean_std_hsv(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    mean_hsv = tuple(float(v) for v in np.mean(pixels, axis=0))
    std_hsv = tuple(float(v) for v in np.std(pixels, axis=0))
    return mean_hsv, std_hsv


def compute_edge_density(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def compute_gradient_magnitude_stats(frame, obj):
    roi = get_roi(frame, obj)
    if roi is None:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(magnitude)), float(np.std(magnitude))


def extract_features(frame, obj):
    mean_hsv, std_hsv = compute_mean_std_hsv(frame, obj)
    hist_top, hist_bottom, hist_left, hist_right = compute_region_histograms(frame, obj)
    contour_feats = compute_contour_features(frame, obj)
    grad_mean, grad_std = compute_gradient_magnitude_stats(frame, obj)
    return {
        "class_name": obj.class_name,
        "area": obj.area,
        "aspect_ratio": obj.aspect_ratio,
        "center": obj.Center,
        "color_histogram": compute_color_histogram(frame, obj),
        "color_histogram_2d": compute_color_histogram_2d(frame, obj),
        "color_histogram_3d": compute_color_histogram_3d(frame, obj),
        "hist_top": hist_top,
        "hist_bottom": hist_bottom,
        "hist_left": hist_left,
        "hist_right": hist_right,
        "mean_hsv": mean_hsv,
        "std_hsv": std_hsv,
        "hu_moments": compute_hu_moments(frame, obj),
        "contour_features": contour_feats,
        "edge_density": compute_edge_density(frame, obj),
        "gradient_mean": grad_mean,
        "gradient_std": grad_std,
        "width": obj.width,
        "height": obj.height,
    }


def compare_histograms(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0.0
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def compare_hu_moments(hu1, hu2):
    if hu1 is None or hu2 is None:
        return 0.0
    diff = np.abs(hu1 - hu2)
    max_diff = 5.0
    similarity = np.exp(-diff / max_diff)
    return float(np.mean(similarity))


def compare_contour_features(feat1, feat2):
    if feat1 is None or feat2 is None:
        return 0.0
    keys = ["solidity", "extent", "circularity"]
    scores = []
    for k in keys:
        delta = abs(feat1.get(k, 0) - feat2.get(k, 0))
        scores.append(max(0.0, 1.0 - delta * 3.0))
    return float(np.mean(scores))


def compute_iou(obj_a, obj_b):
    inter_left = max(obj_a.Left, obj_b.Left)
    inter_top = max(obj_a.Top, obj_b.Top)
    inter_right = min(obj_a.Right, obj_b.Right)
    inter_bottom = min(obj_a.Bottom, obj_b.Bottom)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    union_area = obj_a.area + obj_b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def score_detection_match(reference_obj, reference_features, candidate, frame):
    if reference_obj is None or reference_features is None:
        return 0.0
    if candidate.class_name != reference_features["class_name"]:
        return 0.0
    score = 40.0
    score += compute_iou(reference_obj, candidate) * 30.0
    area_ratio = min(reference_features["area"], candidate.area) / max(reference_features["area"], candidate.area, 1)
    score += area_ratio * 10.0
    aspect_ratio = reference_features["aspect_ratio"]
    aspect_delta = abs(aspect_ratio - candidate.aspect_ratio) / max(aspect_ratio, candidate.aspect_ratio, 1e-6)
    score += max(0.0, 1.0 - aspect_delta) * 5.0
    ref_center = reference_features["center"]
    dx = candidate.Center[0] - ref_center[0]
    dy = candidate.Center[1] - ref_center[1]
    distance = math.hypot(dx, dy)
    frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
    score += max(0.0, 1.0 - (distance / max(frame_diagonal * 0.35, 1.0))) * 10.0
    hist_score = compare_histograms(reference_features["color_histogram"], compute_color_histogram(frame, candidate))
    score += max(0.0, hist_score) * 5.0
    score += candidate.confidence * 5.0
    return score


def score_reacquisition_match(reference_features, candidate, frame, predicted_center=None):
    if not reference_features or candidate.class_name != reference_features["class_name"]:
        return 0.0

    area_ratio = min(reference_features["area"], candidate.area) / max(reference_features["area"], candidate.area, 1)
    ref_aspect = reference_features["aspect_ratio"]
    cand_aspect = candidate.aspect_ratio
    aspect_delta = abs(ref_aspect - cand_aspect) / max(ref_aspect, cand_aspect, 1e-6)
    aspect_similarity = max(0.0, 1.0 - aspect_delta)
    ref_w, ref_h = reference_features.get("width", 1), reference_features.get("height", 1)
    dim_sim_w = min(ref_w, candidate.width) / max(ref_w, candidate.width, 1)
    dim_sim_h = min(ref_h, candidate.height) / max(ref_h, candidate.height, 1)
    dim_consistency = (dim_sim_w + dim_sim_h) / 2.0
    distance = 0.0
    if predicted_center is not None:
        ref_cx, ref_cy = predicted_center
        dx = candidate.Center[0] - ref_cx
        dy = candidate.Center[1] - ref_cy
        distance = math.hypot(dx, dy)
        frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
        if distance > frame_diagonal * 0.6:
            return 0.0

    score = 0.0
    hu_score = compare_hu_moments(reference_features.get("hu_moments"), compute_hu_moments(frame, candidate))
    score += hu_score * 15.0
    ref_contour = reference_features.get("contour_features")
    cand_contour = compute_contour_features(frame, candidate)
    contour_score = compare_contour_features(ref_contour, cand_contour)
    score += contour_score * 15.0
    hist_3d_score = compare_histograms(reference_features.get("color_histogram_3d"), compute_color_histogram_3d(frame, candidate))
    score += max(0.0, hist_3d_score) * 25.0
    layout_scores = []
    for ref_key, cand_key in [("hist_top", compute_region_histograms(frame, candidate)[0]),
                              ("hist_bottom", compute_region_histograms(frame, candidate)[1]),
                              ("hist_left", compute_region_histograms(frame, candidate)[2]),
                              ("hist_right", compute_region_histograms(frame, candidate)[3])]:
        s = compare_histograms(reference_features.get(ref_key), cand_key)
        layout_scores.append(max(0.0, s))
    layout_score = sum(layout_scores) / max(len(layout_scores), 1)
    score += layout_score * 20.0
    shape_score = (aspect_similarity * 0.3 + dim_consistency * 0.4 + area_ratio * 0.3)
    score += shape_score * 15.0
    ref_edge = reference_features.get("edge_density", 0.0)
    cand_edge = compute_edge_density(frame, candidate)
    edge_sim = max(0.0, 1.0 - abs(ref_edge - cand_edge) / 0.3)
    ref_grad_mean = reference_features.get("gradient_mean", 0.0)
    ref_grad_std = reference_features.get("gradient_std", 0.0)
    cand_grad_mean, cand_grad_std = compute_gradient_magnitude_stats(frame, candidate)
    grad_sim = 1.0
    if cand_grad_mean is not None:
        grad_mean_sim = max(0.0, 1.0 - abs(ref_grad_mean - cand_grad_mean) / max(ref_grad_mean, 10))
        grad_std_sim = max(0.0, 1.0 - abs(ref_grad_std - cand_grad_std) / max(ref_grad_std, 10))
        grad_sim = (grad_mean_sim + grad_std_sim) / 2.0
    score += (edge_sim * 0.5 + grad_sim * 0.5) * 10.0
    if predicted_center is not None:
        frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
        score += max(0.0, 1.0 - (distance / max(frame_diagonal * 0.7, 1.0))) * 5.0
    else:
        score += 5.0
    score += candidate.confidence * 5.0
    return score

import math
import cv2
import numpy as np
import config


def _color_for_angle(deg, threshold=5):
    a = abs(deg)
    if a < threshold:
        return (0, 255, 0)
    if a < threshold * 3:
        return (0, 255, 255)
    return (0, 0, 255)


class TrackerVisualizer:
    def __init__(self, window_name="Tracker Scope"):
        self.window_name = window_name
        self.w = 540
        self.h = 620
        self.bg = (12, 12, 20)

        self.scope_cx = self.w // 2
        self.scope_cy = 295
        self.scope_r = 205

    def _draw_title(self, canvas):
        cv2.putText(canvas, "ANTENNA TRACKER", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 95, 130), 1)

        cv2.line(canvas, (10, 28), (self.w - 10, 28), (30, 32, 48), 1)

    def _draw_arrow(self, canvas, cx, cy, direction, color):
        s = 8
        if direction == "right":
            pts = np.array([[cx - s, cy - s], [cx - s, cy + s], [cx + s, cy]], dtype=np.int32)
        elif direction == "left":
            pts = np.array([[cx + s, cy - s], [cx + s, cy + s], [cx - s, cy]], dtype=np.int32)
        elif direction == "up":
            pts = np.array([[cx - s, cy + s], [cx + s, cy + s], [cx, cy - s]], dtype=np.int32)
        elif direction == "down":
            pts = np.array([[cx - s, cy - s], [cx + s, cy - s], [cx, cy + s]], dtype=np.int32)
        else:
            cv2.circle(canvas, (cx, cy), 5, color, -1)
            return
        cv2.fillPoly(canvas, [pts], color)

    def _draw_movement_panel(self, canvas, pan, tilt):
        cx = self.w // 2
        is_centered = abs(pan) < 1 and abs(tilt) < 1

        pan_dir = "right" if pan > 1 else "left" if pan < -1 else "center"
        tilt_dir = "down" if tilt > 1 else "up" if tilt < -1 else "center"
        pan_color = _color_for_angle(pan)
        tilt_color = _color_for_angle(tilt)

        pan_text = f"PAN {pan:+.1f}"
        tilt_text = f"TILT {tilt:+.1f}"

        (pw, _), _ = cv2.getTextSize(pan_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        (tw, _), _ = cv2.getTextSize(tilt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        sep = 30
        total_w = pw + tw + sep + 40
        start_x = cx - total_w // 2
        ay = 48

        arrow_offset = 14
        self._draw_arrow(canvas, start_x + arrow_offset, ay, pan_dir, pan_color)
        tx = start_x + arrow_offset + 18
        cv2.putText(canvas, pan_text, (tx, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pan_color, 2)

        sep_x = tx + pw + sep // 2
        cv2.putText(canvas, "|", (sep_x - 3, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 65, 90), 1)

        self._draw_arrow(canvas, sep_x + 15, ay, tilt_dir, tilt_color)
        tx2 = sep_x + 33
        cv2.putText(canvas, tilt_text, (tx2, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tilt_color, 2)

    def _draw_target_info(self, canvas, obj_name, is_tracking, confidence, fps):
        cx = self.w // 2

        if obj_name:
            if is_tracking:
                label = f"{obj_name}  ({confidence:.0f}%)"
                color = (0, 255, 0)
            else:
                label = f"LOST  {obj_name}"
                color = (0, 0, 255)
        else:
            label = "NO TARGET SELECTED"
            color = (80, 85, 110)

        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        sep = f"  \xa0  "
        fps_label = f"FPS: {fps:.0f}"
        (fw, _), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        total_w = lw + fw + 40
        start_x = cx - total_w // 2

        cv2.putText(canvas, label, (start_x, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(canvas, fps_label, (start_x + lw + 40, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 125, 150), 1)

    def _draw_scope(self, canvas, obj_norm, is_tracking):
        cx, cy, r = self.scope_cx, self.scope_cy, self.scope_r

        cv2.circle(canvas, (cx, cy), r, (45, 50, 70), 2)

        for frac in [0.25, 0.5, 0.75]:
            cv2.circle(canvas, (cx, cy), int(r * frac), (30, 33, 50), 1)

        tick_len = 10
        for deg in range(0, 360, 15):
            a = math.radians(deg)
            outer = r
            inner = r - tick_len if deg % 45 == 0 else r - 5
            x1 = int(cx + inner * math.cos(a))
            y1 = int(cy + inner * math.sin(a))
            x2 = int(cx + outer * math.cos(a))
            y2 = int(cy + outer * math.sin(a))
            cv2.line(canvas, (x1, y1), (x2, y2), (40, 45, 65), 1 if deg % 45 == 0 else 1)

        for deg, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
            a = math.radians(deg)
            lx = int(cx + (r - 18) * math.cos(a))
            ly = int(cy + (r - 18) * math.sin(a))
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, label, (lx - lw // 2, ly + lh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 110, 150), 1)

        cv2.line(canvas, (cx - r + 20, cy), (cx + r - 20, cy), (35, 38, 55), 1)
        cv2.line(canvas, (cx, cy - r + 20), (cx, cy + r - 20), (35, 38, 55), 1)

        cv2.circle(canvas, (cx, cy), 5, (70, 75, 100), 1)
        cv2.circle(canvas, (cx, cy), 2, (70, 75, 100), -1)

        if obj_norm is not None:
            nx, ny = obj_norm
            dist = math.hypot(nx, ny)
            if dist > 0.98:
                nx = nx / dist * 0.98
                ny = ny / dist * 0.98
            sx = int(cx + nx * r * 0.98)
            sy = int(cy + ny * r * 0.98)

            color = (0, 255, 0) if is_tracking else (0, 0, 255)

            cv2.line(canvas, (cx, cy), (sx, sy), (color[0] // 2, color[1] // 2, color[2] // 2), 1)
            cv2.circle(canvas, (sx, sy), 8, color, 2)
            cv2.circle(canvas, (sx, sy), 4, color, -1)

            ring_phase = (cv2.getTickCount() // 4) % 360
            for angle in range(0, 360, 45):
                a = math.radians(angle + ring_phase)
                px = int(sx + 12 * math.cos(a))
                py = int(sy + 12 * math.sin(a))
                cv2.circle(canvas, (px, py), 1, color, -1)

    def _draw_gauges(self, canvas, pan_deg, tilt_deg):
        gx = 60
        gw = self.w - 120
        mid = gx + gw // 2

        bar_h = 14
        y_pan = 510
        y_tilt = 548
        label_w = 42

        cv2.putText(canvas, "PAN", (12, y_pan + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 135, 160), 1)

        cv2.rectangle(canvas, (gx, y_pan), (gx + gw, y_pan + bar_h), (28, 30, 45), -1)
        cv2.rectangle(canvas, (gx, y_pan), (gx + gw, y_pan + bar_h), (45, 50, 70), 1)
        cv2.line(canvas, (mid, y_pan), (mid, y_pan + bar_h), (60, 65, 90), 1)

        pan_c = max(-config.MAX_PAN, min(config.MAX_PAN, pan_deg))
        pn = pan_c / max(config.MAX_PAN, 1)
        px = int(mid + pn * (gw // 2))
        px = max(gx + 3, min(gx + gw - 3, px))
        p_color = _color_for_angle(pan_deg, 3)
        cv2.rectangle(canvas, (px - 5, y_pan + 2), (px + 5, y_pan + bar_h - 2), p_color, -1)

        pv = f"{pan_deg:+.1f}"
        (pw, _), _ = cv2.getTextSize(pv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, pv, (self.w - pw - 12, y_pan + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 1)

        cv2.putText(canvas, f"-{config.MAX_PAN:.0f}", (gx + 4, y_pan + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 85, 110), 1)
        cv2.putText(canvas, f"+{config.MAX_PAN:.0f}", (gx + gw - 30, y_pan + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 85, 110), 1)

        cv2.putText(canvas, "TILT", (12, y_tilt + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 135, 160), 1)

        cv2.rectangle(canvas, (gx, y_tilt), (gx + gw, y_tilt + bar_h), (28, 30, 45), -1)
        cv2.rectangle(canvas, (gx, y_tilt), (gx + gw, y_tilt + bar_h), (45, 50, 70), 1)
        cv2.line(canvas, (mid, y_tilt), (mid, y_tilt + bar_h), (60, 65, 90), 1)

        tilt_c = max(-config.MAX_TILT, min(config.MAX_TILT, tilt_deg))
        tn = tilt_c / max(config.MAX_TILT, 1)
        tx = int(mid + tn * (gw // 2))
        tx = max(gx + 3, min(gx + gw - 3, tx))
        t_color = _color_for_angle(tilt_deg, 3)
        cv2.rectangle(canvas, (tx - 5, y_tilt + 2), (tx + 5, y_tilt + bar_h - 2), t_color, -1)

        tv = f"{tilt_deg:+.1f}"
        (tw, _), _ = cv2.getTextSize(tv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, tv, (self.w - tw - 12, y_tilt + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 1)

        cv2.putText(canvas, f"-{config.MAX_TILT:.0f}", (gx + 4, y_tilt + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 85, 110), 1)
        cv2.putText(canvas, f"+{config.MAX_TILT:.0f}", (gx + gw - 30, y_tilt + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 85, 110), 1)

    def _draw_servo_indicators(self, canvas, pan_deg, tilt_deg):
        cx, cy = 72, self.scope_cy
        r = 34

        cv2.circle(canvas, (cx, cy), r, (35, 38, 55), 1)
        cv2.circle(canvas, (cx, cy), r - 8, (35, 38, 55), 1)
        cv2.line(canvas, (cx - r + 4, cy), (cx + r - 4, cy), (40, 45, 65), 1)
        cv2.line(canvas, (cx, cy - r + 4), (cx, cy + r - 4), (40, 45, 65), 1)

        pan_r = math.radians(max(-90, min(90, pan_deg)))
        tilt_r = math.radians(max(-90, min(90, tilt_deg)))
        ex = int(cx + math.sin(pan_r) * (r - 6))
        ey = int(cy + math.sin(tilt_r) * (r - 6))
        ex = max(cx - r + 6, min(cx + r - 6, ex))
        ey = max(cy - r + 6, min(cy + r - 6, ey))

        p_color = _color_for_angle(pan_deg, 5)
        t_color = _color_for_angle(tilt_deg, 5)
        color = (max(p_color[0], t_color[0]), max(p_color[1], t_color[1]), max(p_color[2], t_color[2]))
        cv2.circle(canvas, (ex, ey), 5, color, -1)
        cv2.circle(canvas, (ex, ey), 7, (60, 65, 90), 1)

        cv2.putText(canvas, "SERVO", (cx - 18, cy + r + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 75, 100), 1)

    def _draw_status_bar(self, canvas, pan_deg, tilt_deg):
        is_centered = abs(pan_deg) < 1 and abs(tilt_deg) < 1
        status_color = (0, 255, 0) if is_centered else (0, 255, 255)
        status_text = " CENTERED" if is_centered else " TRACKING"

        cv2.circle(canvas, (22, self.h - 16), 4, status_color, -1)
        cv2.putText(canvas, status_text, (30, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        right = "ch1:PAN  ch2:TILT  |  [Q] quit"
        (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, right, (self.w - rw - 14, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 85, 110), 1)

    def update(self, pan_deg, tilt_deg, obj_norm, is_tracking, confidence, fps, obj_name):
        canvas = np.full((self.h, self.w, 3), self.bg, dtype=np.uint8)

        self._draw_title(canvas)
        self._draw_movement_panel(canvas, pan_deg, tilt_deg)
        self._draw_target_info(canvas, obj_name, is_tracking, confidence, fps)
        self._draw_scope(canvas, obj_norm, is_tracking)
        self._draw_gauges(canvas, pan_deg, tilt_deg)
        self._draw_servo_indicators(canvas, pan_deg, tilt_deg)
        self._draw_status_bar(canvas, pan_deg, tilt_deg)

        cv2.imshow(self.window_name, canvas)

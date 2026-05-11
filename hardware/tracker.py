try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False

import config


class AntennaTracker:
    def __init__(self, port="COM3", baud=57600, pan_channel=1, tilt_channel=2):
        self.conn = None
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel

        if not HAS_MAVLINK:
            print("pymavlink not installed — mock mode (no servo commands)")
            return

        try:
            self.conn = mavutil.mavlink_connection(port, baud=baud, autoreconnect=True)
            self.conn.wait_heartbeat(timeout=10)
            print(f"Tracker connected on {port} (baud={baud})")
            print(f"  Pan: ch{pan_channel}, Tilt: ch{tilt_channel}")
        except Exception as e:
            print(f"Tracker connection failed ({e}) — mock mode")
            self.conn = None

    def _angle_to_pwm(self, angle_deg):
        return int(config.SERVO_CENTER + (angle_deg / 90.0) * config.SERVO_RANGE)

    def _send_servo(self, channel, pwm):
        if self.conn is None:
            return
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
            channel, pwm, 0, 0, 0, 0, 0
        )

    def send_pan(self, angle_deg):
        clamped = max(-90.0, min(90.0, angle_deg))
        pwm = self._angle_to_pwm(clamped)
        self._send_servo(self.pan_channel, pwm)
        mode = "MAVLink" if self.conn else "MOCK"
        direction = "RIGHT" if clamped > 1 else "LEFT" if clamped < -1 else "CENTER"
        print(f"[{mode}] PAN ch{self.pan_channel}: {clamped:+.1f} deg ({direction}) PWM={pwm}")

    def send_tilt(self, angle_deg):
        clamped = max(-90.0, min(90.0, angle_deg))
        pwm = self._angle_to_pwm(clamped)
        self._send_servo(self.tilt_channel, pwm)
        mode = "MAVLink" if self.conn else "MOCK"
        direction = "DOWN" if clamped > 1 else "UP" if clamped < -1 else "CENTER"
        print(f"[{mode}] TILT ch{self.tilt_channel}: {clamped:+.1f} deg ({direction}) PWM={pwm}")

    def close(self):
        if self.conn:
            self.send_pan(0)
            self.send_tilt(0)
            self.conn.close()
            print("Tracker disconnected")

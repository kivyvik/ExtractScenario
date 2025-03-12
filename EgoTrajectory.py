from typing import Tuple, List


class EgoTrajectory:
    def __init__(self):
        """
        Initialize an empty ego trajectory.
        Stores timestamps, positions (x, y), speed, orientation, and optionally acceleration.
        """
        self.start_frames = [] # First FrameID at which gps info updated
        self.timestamps = []  # List of timestamps
        # List of (x, y) tuples
        self.positions = []  # type: List[Tuple[int, int]]
        self.speeds = []  # List of speeds (m/s)
        self.orientations = []  # List of orientations (e.g., heading in radians)
        self.accelerations = []  # List of accelerations (optional)

    def add_point(self, timestamp, x, y, speed, frame_id=-1, orientation=None, acceleration=None):
        """
        Add a trajectory point.

        :param frame_id: FrameID
        :param timestamp: Time in seconds.
        :param x: X-coordinate of the ego vehicle.
        :param y: Y-coordinate of the ego vehicle.
        :param speed: Speed in m/s.
        :param orientation: Orientation (heading) in radians.
        :param acceleration: Optional acceleration in m/s².
        """
        self.start_frames.append(frame_id)
        self.timestamps.append(timestamp)
        self.positions.append((x, y))
        self.speeds.append(speed)
        # self.orientations.append(orientation)
        # self.accelerations.append(acceleration if acceleration is not None else 0.0)

    def get_latest(self):
        """Retrieve the most recent trajectory point."""
        if not self.timestamps:
            return None
        return {
            "timestamp": self.timestamps[-1],
            "position": self.positions[-1],
            "speed": self.speeds[-1],
            "orientation": self.orientations[-1],
            "acceleration": self.accelerations[-1]
        }

    def interpolate(self, target_time):
        """
        Interpolate the trajectory to estimate the state at a given timestamp.

        :param target_time: The target time to interpolate.
        :return: Interpolated (x, y, speed, orientation, acceleration) or None if out of bounds.
        """
        if not self.timestamps or target_time < self.timestamps[0] or target_time > self.timestamps[-1]:
            return None

        for i in range(len(self.timestamps) - 1):
            t1, t2 = self.timestamps[i], self.timestamps[i + 1]
            if t1 <= target_time <= t2:
                ratio = (target_time - t1) / (t2 - t1)
                x = self.positions[i][0] + ratio * (self.positions[i + 1][0] - self.positions[i][0])
                y = self.positions[i][1] + ratio * (self.positions[i + 1][1] - self.positions[i][1])
                speed = self.speeds[i] + ratio * (self.speeds[i + 1] - self.speeds[i])
                orientation = self.orientations[i] + ratio * (self.orientations[i + 1] - self.orientations[i])
                acceleration = self.accelerations[i] + ratio * (self.accelerations[i + 1] - self.accelerations[i])
                return x, y, speed, orientation, acceleration

        return None

    def to_dict(self):
        """Convert trajectory to a dictionary format."""
        return {
            "timestamps": self.timestamps,
            "positions": self.positions,
            "speeds": self.speeds,
            "orientations": self.orientations,
            "accelerations": self.accelerations
        }

    def __repr__(self):
        return f"EgoTrajectory(points={len(self.timestamps)})"

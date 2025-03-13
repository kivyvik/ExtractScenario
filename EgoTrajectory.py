from typing import Tuple, List

import numpy as np


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
        self.speeds = []  # List of speeds (km/h)
        self.headings = []  # List of orientations (e.g., heading in radians)
        self.accelerations = []  # List of accelerations (optional)

    def shift_to_origin(self):
        avg_x = sum(x for x, y in self.positions) / len(self.positions)
        avg_y = sum(y for x, y in self.positions) / len(self.positions)

        self.positions = [(x - avg_x, y - avg_y) for x, y in self.positions]

    def compute_heading(self):
        """Compute heading (orientation) from a list of (x, y) positions."""
        for i in range(len(self.positions) - 1):
            x1, y1 = self.positions[i]
            x2, y2 = self.positions[i + 1]

            theta = np.arctan2(y2 - y1, x2 - x1)  # Radians
            self.headings.append(theta)

        # Assign last heading same as previous
        if len(self.headings) > 0:
            self.headings.append(self.headings[-1])

    def rotate_trajectory(self):
        """
        Rotates the trajectory to align with the x-axis.
        """
        if not self.positions:
            return

        # Compute the average heading
        avg_heading = np.arctan2(
            self.positions[-1][1] - self.positions[0][1],
            self.positions[-1][0] - self.positions[0][0]
        )

        # Compute rotation matrix
        cos_theta = np.cos(-avg_heading)
        sin_theta = np.sin(-avg_heading)

        def rotate_point(x, y):
            """ Rotate a point around the origin """
            return (
                x * cos_theta - y * sin_theta,
                x * sin_theta + y * cos_theta
            )

        # Rotate all positions
        self.positions = [rotate_point(x, y) for x, y in self.positions]

        # Adjust headings
        if self.headings:
            self.headings = [(h - avg_heading) % (2 * np.pi) for h in self.headings]

    def add_point(self, timestamp, x, y, speed, frame_id=-1, acceleration=None):
        """
        Add a trajectory point.

        :param frame_id: FrameID
        :param timestamp: Time in seconds.
        :param x: X-coordinate of the ego vehicle.
        :param y: Y-coordinate of the ego vehicle.
        :param speed: Speed in km/h.
        :param acceleration: Optional acceleration in m/s².
        """
        self.start_frames.append(frame_id)
        self.timestamps.append(timestamp)
        self.positions.append((x, y))
        self.speeds.append(speed)
        # self.accelerations.append(acceleration if acceleration is not None else 0.0)

    def get_latest(self):
        """Retrieve the most recent trajectory point."""
        if not self.timestamps:
            return None
        return {
            "timestamp": self.timestamps[-1],
            "position": self.positions[-1],
            "speed": self.speeds[-1],
            "orientation": self.headings[-1],
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
                orientation = self.headings[i] + ratio * (self.headings[i + 1] - self.headings[i])
                acceleration = self.accelerations[i] + ratio * (self.accelerations[i + 1] - self.accelerations[i])
                return x, y, speed, orientation, acceleration

        return None

    def to_dict(self):
        """Convert trajectory to a dictionary format."""
        return {
            "timestamps": self.timestamps,
            "positions": self.positions,
            "speeds": self.speeds,
            "orientations": self.headings,
            "accelerations": self.accelerations
        }

    def __repr__(self):
        return f"EgoTrajectory(points={len(self.timestamps)})"

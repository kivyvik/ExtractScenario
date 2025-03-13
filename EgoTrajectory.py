import bisect
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

    def interpolate(self, target_time):
        """
        Interpolate the trajectory to estimate the state at a given timestamp.

        :param target_time: The target time to interpolate.
        :return: Interpolated (x, y, speed, orientation) or None if out of bounds.
        """
        if target_time < self.timestamps[0] or target_time > self.timestamps[-1]:
            return None

        # Find the indices such that timestamps[i] <= target_time <= timestamps[i+1]
        idx = np.searchsorted(self.timestamps, target_time) - 1
        if idx < 0:
            idx = 0

        t0, t1 = self.timestamps[idx], self.timestamps[idx + 1]
        factor = (target_time - t0) / (t1 - t0)

        x = self.positions[idx][0] + factor * (self.positions[idx + 1][ 0] - self.positions[idx][ 0])
        y = self.positions[idx][ 1] + factor * (self.positions[idx + 1][ 1] - self.positions[idx][ 1])
        speed = self.speeds[idx] + factor * (self.speeds[idx + 1] - self.speeds[idx])
        heading = self.headings[idx] + factor * (self.headings[idx + 1] - self.headings[idx])
        return x, y, speed, heading

    def __repr__(self):
        return f"EgoTrajectory(points={len(self.timestamps)})"

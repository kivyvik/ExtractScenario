from typing import Tuple, List

import numpy as np

class VehicleTrajectory:
    def __init__(self, _id):
        self.id = _id
        self.timestamps = []  # List of timestamps
        # List of (x, y) tuples
        self.positions = []  # type: List[Tuple[int, int]]
        self.speeds = []  # List of speeds (km/h)
        self.headings = []  # List of orientations (e.g., heading in radians)
        self.accelerations = []  # List of accelerations (optional)
        self.length = 0
        self.width = 0

    def set_bounding_box(self, length, width):
        self.length = length
        self.width = width

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

    def add_point(self, timestamp, x, y):
        """
        Add a trajectory point.

        :param timestamp: Time in seconds.
        :param x: X-coordinate of the ego vehicle.
        :param y: Y-coordinate of the ego vehicle.
        """
        self.timestamps.append(timestamp)
        self.positions.append((x, y))
        # self.speeds.append(speed)
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
        heading = self.headings[idx] + factor * (self.headings[idx + 1] - self.headings[idx])
        return x, y, heading

    def __repr__(self):
        return f"VehicleTrajectory(points={len(self.timestamps)})"
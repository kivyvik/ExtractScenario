import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from EgoTrajectory import EgoTrajectory
from main import load_data, extract_ego_trajectory, extract_lane_boundaries


# Helper function to compute the ego car rectangle corners.
# The rectangle is computed given that ego_pos is at the front center.
# Typical car dimensions (length=4.5, width=2.0) are used.
# -----------------------------------------------------------------------------
def get_ego_rectangle(ego_pos, heading, car_length=4.5, car_width=2.0):
    ego_pos = np.array(ego_pos)
    # Unit vector along the car's heading
    v = np.array([np.cos(heading), np.sin(heading)])
    # Perpendicular (to the left) for the width direction
    v_perp = np.array([-np.sin(heading), np.cos(heading)])

    # The ego_pos is the front-center. So:
    front_left = ego_pos + (car_width / 2) * v_perp
    front_right = ego_pos - (car_width / 2) * v_perp
    # Rear center is located car_length behind the front
    back_center = ego_pos - car_length * v
    back_left = back_center + (car_width / 2) * v_perp
    back_right = back_center - (car_width / 2) * v_perp

    # Return polygon corners in order (to draw a closed rectangle)
    return np.array([front_left, front_right, back_right, back_left])


# -----------------------------------------------------------------------------
# Helper function to transform global coordinates into the local (ego) view.
# A fixed offset (default (10, 0)) is added so that the ego's front appears at that
# location in the plot window. This fixed offset lets the window cover some area behind
# and ahead of the ego.
# -----------------------------------------------------------------------------
def transform_to_local(points, ego_position, offset=np.array([10, 0])):
    """
    points: array of (x,y) points in global coordinates
    ego_position: the current ego position (global)
    offset: the translation to place the ego vehicle in the plot window.
    """
    points = np.array(points)
    return points - np.array(ego_position) + offset


# -----------------------------------------------------------------------------
# Main function to create a video animation.
# -----------------------------------------------------------------------------
def create_video(ego_traj, lane_boundaries, view_offset=np.array([10, 0]), video_filename='ego_video.mp4'):
    """
    ego_traj: an instance of EgoTrajectory containing the vehicle data.
    lane_boundaries: list of polylines (each polyline is a list of (x,y) points in global coords).
    view_offset: a fixed offset so that ego's front position is drawn at this location in the plot.
    video_filename: output video filename.

    This function adjusts the video speed so that the video duration matches the real
    simulation duration based on the trajectory timestamps.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set fixed view window: length 80, width 30 (x: 0 to 80, y: -15 to 15)
    ax.set_xlim(0, 80)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_title('Ego Trajectory and Lane Boundaries')

    # Create a Line2D object for each lane boundary polyline
    lane_lines = []
    for _ in lane_boundaries:
        line, = ax.plot([], [], 'k-', linewidth=2)  # lane boundaries in black
        lane_lines.append(line)

    # Create a Polygon patch for the ego vehicle (will be updated each frame)
    ego_patch = Polygon([[0, 0]], closed=True, fc='blue', ec='black', alpha=0.7)
    ax.add_patch(ego_patch)

    # Display the timestamp in the upper left
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Compute average time difference between frames (in seconds)
    if len(ego_traj.timestamps) > 1:
        total_duration = ego_traj.timestamps[-1] - ego_traj.timestamps[0]
        avg_dt = total_duration / (len(ego_traj.timestamps) - 1)
        print(ego_traj.timestamps)
    else:
        avg_dt = 0.1  # default fallback

    # Set the interval in milliseconds for the animation update to match real time
    interval = avg_dt * 1000  # milliseconds

    # Calculate FPS for the video writer so that video duration matches simulation duration
    fps = len(ego_traj.timestamps) / (ego_traj.timestamps[-1] - ego_traj.timestamps[0]) if total_duration > 0 else 10

    def init():
        for line in lane_lines:
            line.set_data([], [])
        ego_patch.set_xy([[0, 0]])
        time_text.set_text('')
        return lane_lines + [ego_patch, time_text]

    def animate(i):
        # Retrieve ego state at frame i
        ego_pos = ego_traj.positions[i]  # global coordinates
        ego_heading = ego_traj.headings[i]

        # Update lane boundaries: transform each polyline to ego-local coordinates
        for j, polyline in enumerate(lane_boundaries):
            polyline = np.array(polyline)
            local_polyline = transform_to_local(polyline, ego_pos, offset=view_offset)
            lane_lines[j].set_data(local_polyline[:, 0], local_polyline[:, 1])

        # Compute and update the ego vehicle rectangle
        rect_global = get_ego_rectangle(ego_pos, ego_heading)
        rect_local = transform_to_local(rect_global, ego_pos, offset=view_offset)
        ego_patch.set_xy(rect_local)

        # Update timestamp text
        time_text.set_text(f'Time: {ego_traj.timestamps[i]:.2f}s')
        return lane_lines + [ego_patch, time_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(ego_traj.timestamps),
                                  init_func=init, blit=True, interval=interval)

    # Save the animation using the computed fps so that the video reflects real timestamps.
    Writer = animation.writers['ffmpeg'] if 'ffmpeg' in animation.writers.list() else animation.PillowWriter
    writer = Writer(fps=fps)
    ani.save(video_filename, writer=writer)
    plt.close(fig)
    print(f"Video saved as {video_filename} (FPS: {fps:.2f}, Interval: {interval:.1f} ms per frame)")

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    data_bundle = load_data()

    ego_traj = extract_ego_trajectory(data_bundle.df_car)
    df_lka = data_bundle.dfs_mobileye['LKA']
    left, right = extract_lane_boundaries(ego_traj, df_lka)
    lane_boundaries = [left, right]
    # Generate and save the video
    create_video(ego_traj, lane_boundaries, view_offset=np.array([10, 0]), video_filename='ego_video.mp4')

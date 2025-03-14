import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

from EgoTrajectory import EgoTrajectory
from VehicleTrajectory import VehicleTrajectory
from visualizer import Visualizer


@dataclass
class DataBundle:
    df_car: pd.DataFrame
    df_radar: pd.DataFrame
    dfs_mobileye: Dict[str, pd.DataFrame]


def load_data() -> DataBundle:
    """
    读取 '.xlsx' 中的 ‘Car' sheet, which columns are:
    ['FrameID', 'Vehicle Speed(km/h)', 'Engine Speed(rpm)', 'Accelerograph',
       'Brake Info', 'Brake(kpa)', 'Gear', 'Steering Wheel(°)', 'Yaw_rate',
       'Longitudinal_Acceleration', 'Lateral_Acceletation', 'Left_Light',
       'Right_Light', 'Passing_Light', 'High_Light', 'Wheel_FR', 'Wheel_FL',
       'Wheel_RR', 'Wheel_RL', 'Whistle_Single', 'Wiper', 'DateTime',
       'Longitude', 'Latitude', 'Course', 'Height', 'GPS_Time', 'Angle',
       'CIPV(object ID)', 'THW(0.1s)', 'Relative Times(ms)']
    读取 '_radar.xlsx' 中的 'Radar' sheet, which columns are:
    ['FrameID', 'ObjectID', 'Class', 'MeasState', 'ProbOfExist',
       'DistLong(m)', 'DistLat(m)', 'VrelLong(m/s)', 'VrelLat(m/s)',
       'ArelLong(m/s2)', 'ArelLat(m/s2)', 'DynProp', 'Length(m)', 'Width(m)',
       'OrientationAngle(deg)']
    读取 '_mobileye.xlsx’ 中的所有 sheet, which sheets are:
    ['Standard', 'Obstacle', 'TSR', 'Lane', 'LKA', 'Next Lane']
    其中 'Next Lane' 有可能为 'NL'
    """
    data_path = r'.\data\2019-04-15-10-36-22_camera#01' # 跟车
    car = data_path + r'.xlsx'
    radar = data_path + r'_radar.xlsx'
    mobileye = data_path+  r'_mobileye.xlsx'

    df_car = pd.read_excel(car, sheet_name='Car')
    df_radar = pd.read_excel(radar, sheet_name='Radar')
    dfs_mobileye = pd.read_excel(mobileye, sheet_name=None)
    return DataBundle(df_car=df_car, df_radar=df_radar, dfs_mobileye=dfs_mobileye)


def euclidean_distance(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)


def extract_lane_boundaries(ego_traj: EgoTrajectory, df: pd.DataFrame):
    """
    extract left and right lane boundaries of the lane on which Ego is running
    :param ego_traj: EgoTrajectory
    :param df: df_lka (or df_nl), 'LKA' (Lane Keeping Assistant) sheet in '_mobileye.xlsx'
    :return:
    """
    left = []
    right = []
    frames = []
    # note: len(start_frames) = len(positions) + 1
    assert len(ego_traj.start_frames) == len(ego_traj.positions) + 1
    for i in range(len(ego_traj.positions)):
        frame_id_start = ego_traj.start_frames[i]
        frame_id_end = ego_traj.start_frames[i+1]
        theta = ego_traj.headings[i]
        rows = pd.DataFrame() # type: pd.DataFrame
        frame_id = frame_id_start
        while frame_id < frame_id_end:
            rows = df.loc[df['FrameID'] == frame_id]
            if len(rows) == 2:
                break
            frame_id += 1
        if frame_id == frame_id_end:
            raise ValueError(f'Cannot find given FrameID: {frame_id_start}->{frame_id_end}')
        position = ego_traj.positions[i]
        for _, row in rows.iterrows():
            distance = abs(row['C0'])
            # Compute perpendicular offsets
            dx = distance * np.cos(theta + np.pi / 2)  # +90° for left
            dy = distance * np.sin(theta + np.pi / 2)
            # print(distance, dx, dy)
            if row['Direction'].lower() == 'left':
                left.append((position[0] + dx, position[1] + dy))
            elif row['Direction'].lower() == 'right':
                right.append((position[0] - dx, position[1] - dy))
            else:
                raise ValueError(f"FrameID={frame_id}, field 'Direction' is invalid")
        frames.append(frame_id)
    assert len(left) == len(right)  == len(ego_traj.positions)
    left_interp = []
    right_interp = []
    for i in range(len(left) - 1):
        frame_id = frames[i]
        rows = df.loc[df['FrameID'] == frame_id]
        rows = rows.fillna(0)
        start = ego_traj.positions[i]
        end = ego_traj.positions[i+1]
        theta = ego_traj.headings[i]
        for _, row in rows.iterrows():
            C0 = row['C0']
            C1 = row['C1']
            C2 = row['C2']
            C3 = row['C3']
            distance = euclidean_distance(start, end)
            Zs = np.linspace(0.0, distance, 5, endpoint=False) # physical longitudinal distance from camera
            for z in Zs:
                new_position = start[0] + z * np.cos(theta), start[1] + z * np.sin(theta)
                X = C3 * z ** 3 + C2 * z ** 2 + C1 * z + C0 # physical lateral distance from camera
                X = abs(X)
                # Compute perpendicular offsets
                dx = X * np.cos(theta + np.pi / 2)  # +90° for left
                dy = X * np.sin(theta + np.pi / 2)
                if row['Direction'].lower() == 'left':
                    left_interp.append((new_position[0] + dx, new_position[1] + dy))
                elif row['Direction'].lower() == 'right':
                    right_interp.append((new_position[0] - dx, new_position[1] - dy))
    # handle last point
    if len(left) > 0:
        left_interp.append(left[-1])
        right_interp.append(right[-1])

    return left_interp, right_interp


def extract_all_lane_boundaries(data_bundle: DataBundle, ego_traj: EgoTrajectory):
    lane_boundaries = []
    df_lka = data_bundle.dfs_mobileye['LKA']
    left, right = extract_lane_boundaries(ego_traj, df_lka)
    lane_boundaries.append(left)
    lane_boundaries.append(right)
    try:
        df_nl = data_bundle.dfs_mobileye['NL']
    except KeyError:
        df_nl = data_bundle.dfs_mobileye['Next Lane']
    if df_nl is None:
        raise ValueError('cannot find sheet "NL" or "Next Lane" in mobileye.xlsx')
    left, right = extract_lane_boundaries(ego_traj, df_nl)
    lane_boundaries.append(left)
    lane_boundaries.append(right)
    return lane_boundaries


def extract_other_vehicle_trajectory(data_bundle: DataBundle, ego_traj: EgoTrajectory):
    """
    extract_other_vehicle_trajectory
    :param data_bundle: extract df_obstacle from '.mobileye.xlsx'
    :param ego_traj: EgoTrajectory for reference
    :return:
    """
    def generate_new_id(existing_dict):
        new_id = max(existing_dict.keys(), default=0) + 1
        return new_id

    df = data_bundle.dfs_mobileye['Obstacle']
    id_to_traj = {}
    for i in range(len(ego_traj.positions)):
        frame_id_start = ego_traj.start_frames[i]
        frame_id_end = ego_traj.start_frames[i+1]
        frame_id = frame_id_start
        theta = ego_traj.headings[i]
        while frame_id < frame_id_end:
            rows = df.loc[df['FrameID'] == frame_id]
            cols = ['Obstacle_ID', 'Obstacle_Pos_X', 'Obstacle_Pos_Y', 'Obstacle_Width', 'Obstacle_Length']
            if rows[cols].notna().all().all():
                break
            frame_id += 1
        if frame_id == frame_id_end:
            continue
        rows = df.loc[df['FrameID'] == frame_id]
        for _, row in rows.iterrows():
            obs_id = row['Obstacle_ID']
            if obs_id not in id_to_traj:
                traj = VehicleTrajectory(obs_id)
                print(f'Creating new object, id={obs_id}')
                traj.set_bounding_box(row['Obstacle_Length'], row['Obstacle_Width'])
                id_to_traj[obs_id] = traj
            obs_traj = id_to_traj[obs_id]
            pos = np.array([ego_traj.positions[i][0], ego_traj.positions[i][1]])
            rel_x = row['Obstacle_Pos_X'] # always > 0
            rel_y = row['Obstacle_Pos_Y']
            v = np.array([np.cos(theta), np.sin(theta)])
            pos += rel_x * v
            v_perp = np.array([-np.sin(theta), np.cos(theta)])
            pos += rel_y * v_perp
            if obs_traj.timestamps and ego_traj.timestamps[i] - obs_traj.timestamps[-1] > 4:
                # it is a new obstacle
                new_id = generate_new_id(id_to_traj)
                id_to_traj[new_id] = obs_traj
                print(f'moving old id={obs_id} to id={new_id}')
                traj = VehicleTrajectory(obs_id)
                print(f'Creating new object, id={obs_id}')
                traj.set_bounding_box(row['Obstacle_Length'], row['Obstacle_Width'])
                id_to_traj[obs_id] = traj
                obs_traj = traj
            obs_traj.add_point(ego_traj.timestamps[i], pos[0], pos[1])
    trajectories = []
    for traj in id_to_traj.values(): # filter obstacles
        if len(traj.positions) >= 2:
            traj.compute_heading()
            trajectories.append(traj)
    return trajectories


def extract_ego_trajectory(df: pd.DataFrame) -> EgoTrajectory:
    """
    Extract Information from sheet 'Car' using gps_time

    :param df: df_car, which contains column 'GPS_Time'
    :return: EgoTrajectory object
    """
    ego_traj = EgoTrajectory()
    assert 'GPS_Time' in df.columns, "missing 'GPS_Time' column"
    df = process_gps_time(df)
    assert 'relative_time' in df.columns, "missing 'relative_time' column"

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    filtered = df.drop_duplicates(subset=['Longitude', 'Latitude'])
    for index, row in filtered.iterrows():
        x, y = transformer.transform(row['Longitude'], row['Latitude'])
        ego_traj.add_point(row['relative_time'], x, y, row['Vehicle Speed(km/h)'], frame_id=row['FrameID'])

    last_frame_id = df.iloc[-1]['FrameID']
    ego_traj.start_frames.append(last_frame_id + 1)

    ego_traj.shift_to_origin()
    ego_traj.rotate_trajectory()
    ego_traj.compute_heading()

    return ego_traj


def process_gps_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'GPS_Time' column in the DataFrame to datetime format and calculates the relative time
    (in seconds) with respect to the earliest timestamp in the 'GPS_Time' column. The earliest timestamp
    is set as time 0.

    :param df: A pandas DataFrame containing a 'GPS_Time' column with timestamps.
    :return: The DataFrame with the 'GPS_Time' column converted to datetime format and an additional
              'relative_time' column representing the time difference (in seconds) from the earliest timestamp.

    :example:
    Given a 'GPS_Time' column with values like:
    2019-4-15_2:41:0.574
    The function will:
    - Convert these strings into datetime objects.
    - Calculate the relative time (in seconds) by subtracting the earliest timestamp.

    :note:
    - Rows with invalid or missing 'GPS_Time' values will be dropped.
    - The result is a modified DataFrame with an added 'relative_time' column.
    """
    df = df.dropna(subset=['GPS_Time'])
    df['GPS_Time'] = pd.to_datetime(df['GPS_Time'], format="%Y-%m-%d_%H:%M:%S.%f", errors='coerce')
    df['relative_time'] = (df['GPS_Time'] - df['GPS_Time'].min()).dt.total_seconds()
    # print(df['relative_time'][:5])
    return df


def main():
    data_bundle = load_data()

    ego_traj = extract_ego_trajectory(data_bundle.df_car)
    lane_boundaries = extract_all_lane_boundaries(data_bundle, ego_traj)
    other_vehicle_trajectory = extract_other_vehicle_trajectory(data_bundle, ego_traj)
    print(other_vehicle_trajectory)
    for traj in other_vehicle_trajectory:
        print(traj.timestamps)

    visualizer = Visualizer(ego_traj, other_vehicle_trajectory, lane_boundaries)
    visualizer.visualize()


if __name__ == '__main__':
    main()

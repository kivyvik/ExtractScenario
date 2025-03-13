import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
import plotly.graph_objects as go

from Lane import Lane
from EgoTrajectory import EgoTrajectory
from LaneBoundary import LaneBoundary


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
    data_path = r'.\data'
    car = os.path.join(data_path, r'2019-04-15-10-36-22_camera#01.xlsx')
    radar = os.path.join(data_path, r'2019-04-15-10-36-22_camera#01_radar.xlsx')
    mobileye = os.path.join(data_path, r'2019-04-15-10-36-22_camera#01_mobileye.xlsx')

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
    :param df: df_lka, 'LKA' (Lane Keeping Assistant) sheet in '_mobileye.xlsx'
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
            Zs = np.linspace(0.0, distance, 5) # physical longitudinal distance from camera
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

    ego_traj.compute_heading()
    ego_traj.shift_to_origin()

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


def add_trace(fig: go.Figure, trajectory: List[Tuple[int, int]], name):
    """
    使用 Plotly 可视化车辆轨迹
    :param name: name for the trace
    :param fig: go.Figure
    :param trajectory: List of (x,y) co-ordinates
    """


    # 1) 绘制车辆轨迹
    x_vals = [pt[0] for pt in trajectory]
    y_vals = [pt[1] for pt in trajectory]
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name=name,
        line=dict(color='black'),
        marker=dict(size=4, color='red')
    ))


def main():
    data_bundle = load_data()
    # print(data_bundle.df_car.columns)
    # print(data_bundle.df_radar.columns)
    # print(data_bundle.dfs_mobileye.keys())

    ego_traj = extract_ego_trajectory(data_bundle.df_car)
    df_lka = data_bundle.dfs_mobileye['LKA']
    left, right = extract_lane_boundaries(ego_traj, df_lka)

    fig = go.Figure()
    add_trace(fig, ego_traj.positions, name='ego')
    add_trace(fig, left, name='left')
    add_trace(fig, right, name='right')
    fig.show()


if __name__ == '__main__':
    main()

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from pyproj import Transformer
import plotly.graph_objects as go

from Lane import Lane
from EgoTrajectory import EgoTrajectory


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
    for index, row in df.iterrows():
        x, y = transformer.transform(df['Longitude'], df['Latitude'])
        ego_traj.add_point(df['relative_time'], x, y, df['Vehicle Speed(km/h)'], frame_id=df['FrameID'])

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


def visualize(trajectory: List[Tuple[int, int]]):
    """
    使用 Plotly 可视化车辆轨迹
    :param trajectory: List of (x,y) co-ordinates
    """
    fig = go.Figure()

    # 1) 绘制车辆轨迹
    x_vals = [pt[0] for pt in trajectory]
    y_vals = [pt[1] for pt in trajectory]
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name='Car Trajectory',
        line=dict(color='black'),
        marker=dict(size=4, color='red')
    ))

    fig.show()


def main():
    data_bundle = load_data()
    print(data_bundle.df_car.columns)
    print(data_bundle.df_radar.columns)
    print(data_bundle.dfs_mobileye.keys())

    ego_traj = extract_ego_trajectory(data_bundle.df_car)
    visualize(ego_traj.positions)


if __name__ == '__main__':
    main()

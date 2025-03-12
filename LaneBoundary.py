from enum import Enum
from typing import List

class BoundaryType(Enum):
    DASHED = 0        # 0: Dashed boundary
    SOLID = 1         # 1: Solid boundary
    UNDECIDED = 2     # 2: Undecided boundary type
    ROAD_EDGE = 3     # 3: Road edge boundary
    DOUBLE_LANE_MARK = 4  # 4: Double lane mark boundary
    BOTTS_DOTS = 5    # 5: Botts' dots boundary
    INVALID = 6       # 6: Invalid boundary type

    def __str__(self):
        return self.name  # To print the name of the enum (e.g., 'DASHED')

class LaneBoundary:
    def __init__(self, boundary_id: int, geometry:List[int], boundary_type: BoundaryType, reference_lanes=None):
        self.boundary_id = boundary_id
        self.geometry = geometry  # List of points or parametric representation
        self.boundary_type = boundary_type
        # self.reference_lanes = reference_lanes  # List of lane IDs sharing this boundary
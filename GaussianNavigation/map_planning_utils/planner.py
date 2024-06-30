import skimage
import numpy as np
from skimage.draw import line_aa, line
import utils.pose as pu
from map_planning_utils.mapper import Semantic_Mapping
from map_planning_utils.fmm_planner import FMMPlanner
import math


class Planner():

    def __init__(self) -> None:
        
        self.selem = skimage.morphology.disk(3)



    def reset(self, mapper:Semantic_Mapping):
        '''
        reset collision map
            visited map
            last location
            last action
        '''

        start_x, start_y, start_o, planning_window = \
            mapper.get_planner_pose_inputs()
        self.last_loc = [start_x, start_y, start_o]

        map_shape = mapper.full_map.shape
        map_shape = map_shape[2:]
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)

        self.last_action = 0
        self.col_width = 1
        self.been_stuck = False

        # goal map store
        self.global_goal_map = None

    def plan(self, goal_grid, mapper:Semantic_Mapping, selem: int=10, set_global=False):
        

        map_pred = np.rint(mapper.local_map.cpu().numpy()[0, 0, ...])
        start_x, start_y, start_o, planning_window = \
            mapper.get_planner_pose_inputs()
        r, c = start_y, start_x
        start = [int(r * 100.0 / mapper.resolution - planning_window[0]),
                 int(c * 100.0 / mapper.resolution - planning_window[2])]
        start = pu.threshold_poses(start, map_pred.shape)


        if self.global_goal_map is None:
            if isinstance(goal_grid, list):
                goal = np.zeros_like(map_pred)
                goal_grid = pu.threshold_poses(goal_grid, map_pred.shape)
                goal[goal_grid[0], goal_grid[1]] = 1
            elif isinstance(goal_grid, np.ndarray):
                goal = goal_grid * 1.

            if set_global:
                map_shape = mapper.full_map.shape
                map_shape = map_shape[2:]
                self.global_goal_map = np.zeros(map_shape)
                self.global_goal_map[planning_window[0]:planning_window[1],\
                      planning_window[2]:planning_window[3]] = goal * 1.
        else:
            goal = self.global_goal_map[planning_window[0]:planning_window[1],\
                      planning_window[2]:planning_window[3]] * 1.  
            
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / mapper.resolution - planning_window[0]),
                        int(c * 100.0 / mapper.resolution - planning_window[2])]
        last_start = pu.threshold_poses(last_start, map_pred.shape)

        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[planning_window[0]:planning_window[1],\
                      planning_window[2]:planning_window[3]][rr, cc] += 1
        
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = start_x, start_y, start_o
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1                

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < 0.20:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / mapper.resolution), \
                            int(c * 100 / mapper.resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        # init last loc
        self.last_loc = [start_x, start_y, start_o]

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window, selem)
        
        if stop:
            # action = 0  # Stop
            action = {
                "action": ("velocity_control", "velocity_stop"),
                "action_args": {
                    "angular_velocity": np.array([0]),
                    "linear_velocity": np.array([0]),
                    "camera_pitch_velocity": np.array([0]),
                    "velocity_stop": np.array([1]),
                },
            }
            self.last_action = 0
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            # if relative_angle > self.args.turn_angle / 2.:
            if relative_angle > 15.:
                # action = 3  # Right
                self.last_action = 3
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": -ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            # elif relative_angle < -self.args.turn_angle / 2.:
            elif relative_angle < -15.:
                # action = 2  # Left
                self.last_action = 2
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            else:
                # action = 1  # Forward
                self.last_action = 1
                lin_vel = np.array([pu.get_l2_distance(stg_x, start[0], stg_y, start[1]) * 5 / 35.])
                lin_vel = np.clip(lin_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": np.array([0]),
                        "linear_velocity": lin_vel,
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }

        action = {'action': self.last_action}

        return action

    def _get_stg(self, grid, start, goal, planning_window, selem):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        ########################
        # selem = skimage.morphology.disk(3)
        # grid = skimage.morphology.dilation(grid, selem)
        # selem = skimage.morphology.disk(1)
        # grid = skimage.morphology.erosion(grid, selem)
        ########################

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)
        visited = add_boundary(self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2], value=0)

        planner = FMMPlanner(traversible)

        selem = skimage.morphology.disk(selem)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True

        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]

        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        if replan:
            stg_x, stg_y, _, stop = planner.get_short_term_goal(state, 2)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
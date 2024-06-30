import numpy as np
from magnum import Quaternion
import quaternion  # noqa: F401 # pylint: disable=unused-import
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
from skimage.draw import line
import numba

numba.jit(nopython=True)
def get_evenly_distributed_subset(xyo: np.ndarray):
    '''
    xyo: n * 3 defines the spatial location
    '''
    num_points = xyo.shape[0]
    grid_resolution = 2.0 # 1.5m for grid resolution
    angle_resolution = np.deg2rad(15) # 30 for angle resolution

    min_x = np.min(xyo[:, 0])
    max_x = np.max(xyo[:, 0])
    min_y = np.min(xyo[:, 1])
    max_y = np.max(xyo[:, 1])

    grid_x_range = int(np.ceil((max_x - min_x) / grid_resolution))
    grid_y_range = int(np.ceil((max_y - min_y) / grid_resolution))
    grid_o_range = int(np.ceil(2 * np.pi / angle_resolution))

    hash_table = np.zeros((grid_x_range, grid_y_range, grid_o_range, 4))
    count_subset = 0
    index_subset = np.zeros((num_points, ))
    
    for i in range(num_points):
        curr_xyo = xyo[i]
        grid_x = int(np.floor((curr_xyo[0] - min_x) / grid_resolution))
        grid_y = int(np.floor((curr_xyo[1] - min_y) / grid_resolution))
        grid_o = int(np.floor((curr_xyo[2] - (-np.pi)) / angle_resolution))
        if hash_table[grid_x, grid_y, grid_o, 3] == 0:
            hash_table[grid_x, grid_y, grid_o, :3] = curr_xyo
            hash_table[grid_x, grid_y, grid_o, 3] = 1
            index_subset[count_subset] = i
            count_subset += 1
    subset = np.zeros((count_subset, ))
    for i in range(count_subset):
        subset[i] = index_subset[i]        

    # num_subset_points = np.sum(hash_table[:, :, :, 3].flatten())
    # subset = np.zeros((num_subset_points, 3))
    # count = 0
    # for i in range(grid_x_range):
    #     for j in range(grid_y_range):
    #         for z in range(grid_o_range):
    #             if hash_table[i, j, z, 3] == 1:
    #                 subset[count] = hash_table[i, j, z, :3]

    return subset


def quaternion_from_coeff(coeffs: List[float]) -> quaternion.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = quaternion.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_rotate_vector(
    quat: quaternion.quaternion, v: np.ndarray
) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag



def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j**2 + q_k**2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i**2 + q_k**2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i**2 + q_j**2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat

def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def list2qua(li):
    return Quaternion(((li[0], li[1], li[2]), li[3]))

def qua2list(qua):
    if isinstance(qua, Quaternion):
        return [qua.vector[0], qua.vector[1], qua.vector[2], qua.scalar]
    elif isinstance(qua, quaternion.quaternion):
        return [qua.x, qua.y, qua.z, qua.w]
    else:
        return None

def matrix2tra_qua(matrix):
    '''
    qua: x,y,z,w
    '''
    quat = Quaternion.from_matrix(matrix[:3, :3])
    quat = np.array(qua2list(quat))
    trans = matrix[:3, 3]
    return trans, quat


import numpy as np
from magnum import Quaternion
import quaternion  # noqa: F401 # pylint: disable=unused-import
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
from skimage.draw import line

def check_the_same_floor(posA, posB): 
    target_vector = np.array(posB) - np.array(posA) 
    y = target_vector[1] 
    target_vector[1] = 0 
    target_length = np.linalg.norm(np.array([target_vector[0], -target_vector[2]])) 
    angle = np.arctan2(y, target_length)
    return ((angle > -np.pi/6) or (angle < np.pi/6))

def quaternion_from_coeff(coeffs: List[float]) -> quaternion.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = quaternion.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_rotate_vector(
    quat: quaternion.quaternion, v: np.ndarray
) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag



def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j**2 + q_k**2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i**2 + q_k**2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i**2 + q_j**2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def list2qua(li):
    return Quaternion(((li[0], li[1], li[2]), li[3]))

def qua2list(qua):
    if isinstance(qua, Quaternion):
        return [qua.vector[0], qua.vector[1], qua.vector[2], qua.scalar]
    elif isinstance(qua, quaternion.quaternion):
        return [qua.x, qua.y, qua.z, qua.w]
    else:
        return None

def draw_rectangle(obstacle_map, start, goal, width):
    '''
    obstacle_map: indicates shape
    start: line start
    goal: line end
    width: half width of the rectangle
    '''
    if goal[0] == start[0]:
        theta = np.pi/2
    else:
        theta = np.arctan((goal[1] - start[1])/(goal[0] - start[0]))
    ws = width * np.sin(theta)
    wc = width * np.cos(theta)
    h, w = obstacle_map.shape
    sig = np.array([[1, -1],
                    [-1, -1],
                    [-1, 1],
                    [1, 1]])
    wid = np.array([[-ws, wc],
                    [wc, ws]])
    vertices = np.matmul(sig, wid) + np.array([start, start, goal, goal])
    return vertices

def fill_rectangle(obstacle_map, vertices):
    # Create a (h, w) array filled with 0
    img = np.zeros_like(obstacle_map, dtype=np.uint8)
    img_test = np.zeros_like(obstacle_map, dtype=np.uint8)

    # Convert vertices to numpy array
    pts = np.array(vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Fill the rectangle with value 1
    cv2.fillPoly(img, [pts], color=(1))
    return img

def ca_short_term_goal(obstacle_map, start, top_k_goal, width):
    '''
    obstacle map (H, W): 1 means occupied, 0 means free
    start (2, ): planning start
    top_k_goal (k, 2): fmm generated distance field top k shortest goal within range of step size
    constant (property: h, w): car model constant

    important: the start and top_k_goal there indicate pos in numpy array, different from opencv

    functionality: find the short term goal avoiding collision
    '''
    for i in range(len(top_k_goal)):
        if start[0] ==top_k_goal[i][0] and start[1] == top_k_goal[i][1]:
            continue
        vertices = draw_rectangle(obstacle_map, start, top_k_goal[i], width)
        vertices = vertices[: ,[1, 0]]
        rectangle = fill_rectangle(obstacle_map, vertices)
        if np.any((rectangle + obstacle_map) == 2):
            continue
        else:
            return top_k_goal[i]
    return None




def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.:
        o -= 360.

    return x, y, o


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords

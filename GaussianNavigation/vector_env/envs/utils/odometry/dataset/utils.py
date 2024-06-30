import numpy as np
import quaternion


def get_relative_egomotion(data, EPS=1e-8):
    '''
        Get agent states (source and target) from
        the simulator
    '''
    pos_s, rot_s = (
        np.asarray(
            data["source_agent_state"]["position"],
            dtype=np.float32
        ),
        np.asarray(
            data["source_agent_state"]["rotation"],
            dtype=np.float32
        )
    )

    pos_t, rot_t = (
        np.asarray(
            data["target_agent_state"]["position"],
            dtype=np.float32
        ),
        np.asarray(
            data["target_agent_state"]["rotation"],
            dtype=np.float32
        )
    )

    rot_s, rot_t = (
        quaternion.as_quat_array(rot_s),
        quaternion.as_quat_array(rot_t)
    )

    '''
        Convert source/target rotation arrays to 3x3 rotation matrices
    '''
    rot_s2w, rot_t2w = (
        quaternion.as_rotation_matrix(rot_s),
        quaternion.as_rotation_matrix(rot_t)
    )

    '''
        Construct the 4x4 transformation [agent-->world] matrices
        corresponding to the source and target agent state.
    '''
    trans_s2w, trans_t2w = (
        np.zeros(shape=(4, 4), dtype=np.float32),
        np.zeros(shape=(4, 4), dtype=np.float32)
    )
    trans_s2w[3, 3], trans_t2w[3, 3] = 1., 1.
    trans_s2w[0:3, 0:3], trans_t2w[0:3, 0:3] = rot_s2w, rot_t2w
    trans_s2w[0:3, 3], trans_t2w[0:3, 3] = pos_s, pos_t

    '''
        Construct the 4x4 transformation [world-->agent] matrices
        corresponding to the source and target agent state
        by inverting the earlier transformation matrices
    '''
    trans_w2s = np.linalg.inv(trans_s2w)
    trans_w2t = np.linalg.inv(trans_t2w)

    '''
        Construct the 4x4 transformation [target-->source] matrix
        (viewing things from the ref frame of source)
        -- take a point in the agent's coordinate at target state,
        -- transform that to world coordinates (trans_t2w)
        -- transform that to the agent's coordinates at source state (trans_w2s)
    '''
    trans_t2s = np.matmul(trans_w2s, trans_t2w)

    rotation = quaternion.as_rotation_vector(
        quaternion.from_rotation_matrix(trans_t2s[0:3, 0:3])
    )
    assert np.abs(rotation[0]) < EPS
    assert np.abs(rotation[2]) < EPS

    return {
        "translation": trans_t2s[0:3, 3],
        "rotation": rotation[1]
    }

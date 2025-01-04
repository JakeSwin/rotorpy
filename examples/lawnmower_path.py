import zmq
import numpy as np

def get_path(min_coord=-30, max_coord=30, step=5, altitude=8):
    steps = np.arange(min_coord, max_coord, step)
    Ys, Xs = np.meshgrid(steps, steps)
    Ys[1::2, :] = Ys[1::2, ::-1]
    target_poses = np.column_stack((Xs.flatten(), Ys.flatten(), np.repeat(8, len(Ys.flatten()))))
    target_yaws = np.repeat(0, len(Ys.flatten()))
    # Rotation mask for yaws
    # yaw_mask = (np.arange(len(target_yaws)) // len(steps)) % 2 == 1
    # target_yaws[yaw_mask] = np.pi 
    target_poses = np.insert(target_poses, 0, [0, 0, altitude], axis=0)
    target_yaws = np.insert(target_yaws, 0, 0)
    return zip(target_poses.tolist(), target_yaws.tolist())

if __name__ == "__main__":
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5565")

    for pose, yaw in get_path():
        data = np.array(pose + [yaw], dtype=np.float32)
        socket.send(data)
        print("sent data")

        msg = socket.recv()
        print(msg)

    
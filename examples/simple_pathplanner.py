import zmq
import numpy as np

def find_indexes_within_radius(array_size, x, y, radius):
    x_min = max(int(-array_size/2), int(np.floor(x - radius)))
    x_max = min(int(array_size/2) - 1, int(np.ceil(x + radius)))
    y_min = max(int(-array_size/2) + 1, int(np.floor(y - radius)))
    y_max = min(int(array_size/2), int(np.ceil(y + radius)))
    
    indices = []
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if np.sqrt((i - x)**2 + (j - y)**2) <= radius:
                indices.append((i, j))
    
    return indices

def random_next_pos(x, y, radius):
    x_min, x_max = x - radius, x + radius
    y_min, y_max = y - radius, y + radius

    next_x = np.random.uniform(x_min, x_max)
    next_y = np.random.uniform(y_min, y_max)
    print(f"Random: x:{next_x}, y:{next_y}")
    return next_x, next_y

if __name__ == "__main__":
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5565")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5560")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # Send liftoff command
    last_pos = [0, 0, 8]
    payload = np.array(last_pos + [0], dtype=np.float32)
    socket.send(payload)
    print("sent data")
    msg = socket.recv()
    print(msg)

    has_gp_data = False
    gp_data = None
    radius = 13
    gp_size = radius

    while True:
        try:
            metadata = subscriber.recv_json(flags=zmq.NOBLOCK)
            msg = subscriber.recv(flags=zmq.NOBLOCK)
            gp_data = np.frombuffer(msg, dtype=metadata["dtype"]).reshape(metadata["shape"])
            gp_size = len(gp_data[1][0])
            has_gp_data = True
        except Exception:
            x = last_pos[0]
            y = last_pos[1]
            indexes = find_indexes_within_radius(gp_size, x, y, radius)
            print(indexes)
            if has_gp_data:
                covs = []
                sample_idxs = np.array(indexes)
                sample_idxs[:, 0] = sample_idxs[:, 0] + 30
                sample_idxs[:, 1] = -sample_idxs[:, 1] + 30
                for (x_idx, y_idx) in sample_idxs:
                    covs.append(gp_data[1][y_idx, x_idx])
                covs_sorted = np.argsort(covs)
                highest_cov_idx = np.where(covs_sorted == len(indexes)-1)[0][0]
                next_x, next_y = indexes[highest_cov_idx]
                print(f"from GP: x:{next_x}, y:{next_y}")
                if next_x == x and next_y == y:
                    random_idx = np.random.randint(len(indexes))
                    next_x, next_y = indexes[random_idx]
                    print(f"Random: x:{next_x}, y:{next_y}")
            else:
                random_idx = np.random.randint(len(indexes))
                next_x, next_y = indexes[random_idx]
                print(f"Random: x:{next_x}, y:{next_y}")
                # x_min, x_max = x - radius, x + radius
                # y_min, y_max = y - radius, y + radius

                # next_x = np.random.uniform(x_min, x_max)
                # next_y = np.random.uniform(y_min, y_max)
                # print(f"Random: x:{next_x}, y:{next_y}")

            last_pos[0] = next_x
            last_pos[1] = next_y

            payload = np.array(last_pos + [0], dtype=np.float32)
            socket.send(payload)
            print("sent data")
            msg = socket.recv()
            print(msg)
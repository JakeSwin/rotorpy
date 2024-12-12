import zmq
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc
import multiprocessing
from dash.dependencies import Input, Output
from matplotlib.animation import FuncAnimation
import time

from pykrige.ok import OrdinaryKriging

# Data storage for GP
Xs = []
Ys = []
weed_chance = []

def update_plot(socket, flags=0):
    if len(Xs) < 10:
        return

    OK = OrdinaryKriging(
        Xs,
        Ys,
        weed_chance,
        variogram_model='exponential',
        verbose=False,
        enable_plotting=False,
    )

    gridx = np.arange(-30, 30, 1, dtype='float64')
    gridy = np.arange(-30, 30, 1, dtype='float64')
    zstar, ss = OK.execute("grid", gridx, gridy)

    # Package and send GP data
    data = np.stack([zstar, ss])
    metadata = dict(
        dtype=str(data.dtype),
        shape=data.shape,
    )
    socket.send_json(metadata, flags | zmq.SNDMORE)
    socket.send(data, flags)

def get_value_of_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 0])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 0])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    red_mask = cv2.add(mask1, mask2)

    red_pixel_count = cv2.countNonZero(red_mask)

    total_pixels = image.shape[0] * image.shape[1]

    avg_pool = red_pixel_count / total_pixels

    return avg_pool, red_mask

def get_image(message):
    image_bytes = message[1]
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    avg_pool, red_mask = get_value_of_image(image)

    image = cv2.bitwise_and(image, image, mask=red_mask) 

    return image, avg_pool

def main():
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.connect("tcp://localhost:5557")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5559")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    pub = context.socket(zmq.PUB)
    pub.bind("tcp://*:5560")

    image = None

    cv2.namedWindow("Received Image", cv2.WINDOW_NORMAL)

    figure_created = False

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not image is None:
                cv2.imshow("Received Image", image)

            try:
                msg = subscriber.recv_string(flags=zmq.NOBLOCK)
            except Exception:
                continue

            socket.send(b'')
            message = socket.recv_multipart()

            image, avg_pool = get_image(message)

            position_bytes = message[3]
            pos = np.frombuffer(position_bytes, np.float32)

            print(f"Position: {pos}, Avg Pool: {avg_pool}" + " "*20, end="\r")

            if pos[0] not in Xs and pos[2] not in Ys:
                Xs.append(pos[0])
                Ys.append(pos[2])
                weed_chance.append(avg_pool)

            update_plot(pub)

    except KeyboardInterrupt:
        print("Stopping")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        subscriber.close()
        pub.close()
        context.term()

if __name__ == "__main__":
    main()
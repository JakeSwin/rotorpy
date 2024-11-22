import zmq
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from pykrige.ok import OrdinaryKriging

# Data storage for GP
Xs = []
Ys = []
weed_chance = []

# Setup matplot
plt.ion()


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
    poller = zmq.Poller()

    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.connect("tcp://localhost:5557")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5559")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # poller.register(socket, zmq.POLLIN)
    # poller.register(subscriber, zmq.POLLIN)

    i = 0
    take_image = False
    waiting_for_image = False
    image = None

    cv2.namedWindow("Received Image", cv2.WINDOW_NORMAL)
    fig, ax = plt.subplots()
    plt.show(block=False)

    try:
        while True:
            # socks = dict(poller.poll())

            # if subscriber in socks and socks[subscriber] == zmq.POLLIN:
            #     if subscriber.recv_string(flags=zmq.NOBLOCK) == "take_image":
            #         take_image = True
            #         socket.send(b'')
            #         waiting_for_image = True
            # elif not take_image:
            #     continue

            # if socket in socks and socks[socket] == zmq.POLLIN:
            #     message = socket.recv_multipart()
            #     take_image = False
            #     waiting_for_image = False
            # else:
            #     continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not image is None:
                cv2.imshow("Received Image", image)

            try:
                msg = subscriber.recv_string(flags=zmq.NOBLOCK)
            except Exception:
                print("no message")
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

            i += 1

            if i > 10 and len(Xs) > 10:
                i = 0
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

                ax.clear()
                cax = ax.imshow([[]], extent=(-3, 3, -3, 3), origin="lower")
                ax.set_ylim(ax.get_ylim()[::-1])
                cbar = plt.colorbar(cax)

                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.1)  # Add small delay for interaction
                # fig.canvas.draw()
                # fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("Stopping")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        subscriber.close()
        context.term()

if __name__ == "__main__":
    main()
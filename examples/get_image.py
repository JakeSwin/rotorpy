import zmq
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pykrige.ok import OrdinaryKriging

# Data storage for GP
Xs = []
Ys = []
weed_chance = []

# Setup matplot
plt.ion()

fig, ax = plt.subplots()
cax = plt.imshow([[]], extent=(-3, 3, -3, 3), origin="lower")
ax.set_ylim(ax.get_ylim()[::-1])
cbar = plt.colorbar(cax)

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

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5557")

    socket.subscribe("")

    i = 0

    try:
        while True:
            message = socket.recv_multipart()

            image_bytes = message[1]
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("Received Image", image)

            avg_pool, _ = get_value_of_image(image)

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

                gridx = np.arange(-3, 3, 0.1, dtype='float64')
                gridy = np.arange(-3, 3, 0.1, dtype='float64')
                zstar, ss = OK.execute("grid", gridx, gridy)

                cax.set_data(zstar)
                fig.canvas.draw()
                fig.canvas.flush_events()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
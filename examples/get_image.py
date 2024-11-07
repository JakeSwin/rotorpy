import zmq
import cv2
import numpy as np

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5557")

    socket.subscribe("")

    try:
        while True:
            message = socket.recv_multipart()

            image_bytes = message[1]

            nparr = np.frombuffer(image_bytes, np.uint8)

            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("Received Image", image)

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
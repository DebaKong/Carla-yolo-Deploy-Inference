import time
import cv2

from carla_tcp import TcpClient
from carla_world import CarlaWorld
from carla_camera import RGBCamera
from carla_visualization import Visualizer

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

WSL_IP = "172.18.192.102"
WSL_PORT = 5000

class CarlaApplication:
    def __init__(self):
        self.world = CarlaWorld(CARLA_HOST, CARLA_PORT)
        self.camera = None
        self.tcp = TcpClient(WSL_IP, WSL_PORT)
        self.visualizer = Visualizer()

    def start(self):
        self.world.connect()
        vehicle = self.world.spawn_vehicle()

        self.camera = RGBCamera(
            self.world.world,
            vehicle,
            width=1280,
            height=720,
            fov=90,
            jpeg_quality=85
        )
        self.camera.spawn()

        self.tcp.connect()

        print("[APP] Application started.")

    def run(self):
        last_frame_id = None

        try:
            while True:
                # Spectator跟随RGB Camera
                self.camera.update_spectator()

                frame = self.camera.get_latest_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                if frame["frame_id"] == last_frame_id:
                    time.sleep(0.01)
                    continue

                last_frame_id = frame["frame_id"]

                # Send the JPEG frame data over TCP
                self.tcp.send_message(frame["data"])

                print(
                    f"[APP] Sent frame {frame['frame_id']} "
                    f"JPEG size="
                    f"{len(frame['data']) / 1024:.2f} KB"
                )

                # Receive tracks from the server
                tracks = self.tcp.receive_tracks()
                print(
                    f"[Track]"
                    f"frame={frame['frame_id']} "
                    f"objects={len(tracks)}"
                )

                for track in tracks:
                    print(
                        f"class={track['class_id']} "
                        f"ID={track['track_id']} "
                        f"conf={track['confidence']:.3f} "
                        f"box=("
                        f"{track['x']}, "
                        f"{track['y']}, "
                        f"{track['width']}, "
                        f"{track['height']}"
                        f")"
                    )

                # Visualize the tracks on the image
                img = frame["bgr_image"].copy()
                img = self.visualizer.draw(img, tracks)

                cv2.imshow("CARLA RGB Camera", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("[APP] Stopping by user.")

    def stop(self):
        self.tcp.close()

        if self.camera is not None:
            self.camera.destroy()

        self.world.cleanup()

        print("[APP] Cleanup completed.")

def main():
    app = CarlaApplication()

    try:
        app.start()
        app.run()
    finally:
        app.stop()

if __name__ == "__main__":
    main()
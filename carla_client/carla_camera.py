import cv2
import numpy as np
import carla

class RGBCamera:
    def __init__(self, world, vehicle, width=1280, height=720, fov: int = 90, jpeg_quality: int = 85):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.fov = fov
        self.jpeg_quality = jpeg_quality

        self.camera = None
        self.latest_frame = None

    def spawn(self):
        if self.vehicle is None:
            raise RuntimeError("Vehicle is not spawned.")

        blueprint_libraby = self.world.get_blueprint_library()

        camera_bp = blueprint_libraby.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", str(self.fov))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.0)
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        self.camera.listen(self._image_callback)

        print(
            f"[CARLA] RGB Camera spawned:"
            f"{self.width}x{self.height}, FOV: {self.fov}"
        )

    def update_spectator(self):
        """
        让CARLA Spectator跟随RGB Camera。
        Spectator只用于Windows端调试显示
        不参与实际感知数据传输。
        """
        if self.camera is None:
            return

        spectator = self.world.get_spectator()

        camera_transform = self.camera.get_transform()

        spectator.set_transform(camera_transform)

    def _image_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # CARLA: BGRA to OpenCV: BGR
        bgr_image = array[:, :, :3]

        success, encoded_image = cv2.imencode(
            ".jpg",
            bgr_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )

        if not success:
            print("[CARLA] Failed to encode image.")
            return

        self.latest_frame = {
            "frame_id": image.frame,
            "timestamp": image.timestamp,
            "data": encoded_image.tobytes(),
            "bgr_image": bgr_image
        }

    def get_latest_frame(self):
        return self.latest_frame

    def destroy(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
            print("[CARLA] RGB Camera destroyed.")
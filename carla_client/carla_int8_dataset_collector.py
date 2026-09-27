import carla
import cv2
import os
import csv
import random
import time
import queue
import math
import numpy as np


class CarlaYOLODatasetCollector:

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        output_dir="Calibration",
        target_images=500
    ):

        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.target_images = target_images

        # ============================================================
        # 数据集目录
        # ============================================================

        self.image_dir = os.path.join(
            self.output_dir,
            "images"
        )

        self.label_dir = os.path.join(
            self.output_dir,
            "labels"
        )

        os.makedirs(
            self.image_dir,
            exist_ok=True
        )

        os.makedirs(
            self.label_dir,
            exist_ok=True
        )

        self.metadata_path = os.path.join(
            self.output_dir,
            "metadata.csv"
        )

        self._init_metadata()

        # ============================================================
        # CARLA对象
        # ============================================================

        self.client = None
        self.world = None
        self.traffic_manager = None

        self.ego_vehicle = None

        self.camera = None
        self.instance_camera = None

        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []

        # ============================================================
        # Sensor Queue
        # ============================================================

        self.image_queue = queue.Queue(maxsize=5)
        self.instance_queue = queue.Queue(maxsize=5)

        # ============================================================
        # 数据集统计
        # ============================================================

        self.image_count = 0

        # ============================================================
        # Camera参数
        # ============================================================

        self.image_width = 1280
        self.image_height = 720
        self.camera_fov = 90.0

        # ============================================================
        # 场景采样控制
        #
        # 每个场景最多8张
        #
        # 同一个场景中：
        # 图片1 -> 图片2 >= 300 frame
        # 图片2 -> 图片3 >= 300 frame
        #
        # 实际间隔随机300~700 frame
        # ============================================================

        self.min_frame_gap = 300
        self.max_frame_gap = 700

        self.next_save_frame = 0

        # 上一次“成功保存”的图片对应的CARLA frame
        self.last_saved_frame = None

        # 每个场景最多8张
        self.max_images_per_scene = 8

        # 一个场景最多运行4000 frame
        #
        # 8张图片理论上至少需要3200 frame，
        # 留出一定余量给目标过滤。
        self.max_scene_frames = 4000

        # ============================================================
        # 目标过滤
        # ============================================================

        self.min_objects = 1

        # 要求每张图片至少有一个车辆
        self.require_vehicle = True

        # 最少可见像素
        self.min_visible_pixels = 140

        # 可见比例
        self.min_visible_ratio = 0.25

        # mask填充比例
        self.min_mask_fill_ratio = 0.15

        # 最大截断比例
        self.max_truncation_ratio = 0.35

        # 最小目标框
        self.min_bbox_width = 14
        self.min_bbox_height = 14

        # 距离范围
        self.min_object_distance = 5.0
        self.max_object_distance = 40.0

        # ============================================================
        # 地图
        # ============================================================

        self.preferred_maps = [
            "Town01",
            "Town02",
            "Town03",
            "Town04",
            "Town05",
            "Town10HD"
        ]

        self.available_maps = []
        self.selected_maps = []

        # ============================================================
        # 天气
        # ============================================================

        self.weather_presets = [

            (
                "ClearNoon",
                carla.WeatherParameters.ClearNoon
            ),

            (
                "CloudyNoon",
                carla.WeatherParameters.CloudyNoon
            ),

            (
                "WetNoon",
                carla.WeatherParameters.WetNoon
            ),

            (
                "WetCloudyNoon",
                carla.WeatherParameters.WetCloudyNoon
            ),

            (
                "SoftRainNoon",
                carla.WeatherParameters.SoftRainNoon
            ),

            (
                "ClearSunset",
                carla.WeatherParameters.ClearSunset
            ),

            (
                "CloudySunset",
                carla.WeatherParameters.CloudySunset
            )
        ]

        # ============================================================
        # 交通流量
        # ============================================================

        self.traffic_levels = {

            "Medium": {
                "vehicles": 50,
                "walkers": 20
            },

            "High": {
                "vehicles": 65,
                "walkers": 25
            },

            "VeryHigh": {
                "vehicles": 80,
                "walkers": 30
            }
        }

        # ============================================================
        # YOLO类别
        #
        # person      0
        # bicycle     1
        # car         2
        # motorcycle  3
        # bus         5
        # truck       7
        # ============================================================

        self.class_ids = {

            "person": 0,
            "bicycle": 1,
            "car": 2,
            "motorcycle": 3,
            "bus": 5,
            "truck": 7
        }

        # ============================================================
        # 当前场景统计
        # ============================================================

        self.scene_frames = 0
        self.scene_image_count = 0

    # ==================================================================
    # Metadata
    # ==================================================================

    def _init_metadata(self):

        if not os.path.exists(self.metadata_path):

            with open(
                self.metadata_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    "image",
                    "label",
                    "map",
                    "weather",
                    "traffic",
                    "frame",
                    "objects"
                ])

    # ==================================================================
    # 连接CARLA
    # ==================================================================

    def connect(self):

        print("=" * 70)
        print("Connecting to CARLA...")
        print("=" * 70)

        self.client = carla.Client(
            self.host,
            self.port
        )

        self.client.set_timeout(20.0)

        version = self.client.get_server_version()

        print(
            f"CARLA version: {version}"
        )

        if version != "0.9.16":

            print(
                "WARNING: "
                "This script is designed for CARLA 0.9.16."
            )

        self.world = self.client.get_world()

        print(
            f"Current map: "
            f"{self.world.get_map().name}"
        )

    # ==================================================================
    # 获取地图
    # ==================================================================

    def detect_maps(self):

        self.available_maps = (
            self.client.get_available_maps()
        )

        self.selected_maps = []

        for preferred in self.preferred_maps:

            for available in self.available_maps:

                if available.endswith(preferred):

                    self.selected_maps.append(
                        available
                    )

                    break

        print("Selected maps:")

        for map_name in self.selected_maps:

            print(
                f"  {map_name}"
            )

        return len(self.selected_maps) > 0

    # ==================================================================
    # 设置同步模式
    # ==================================================================

    def configure_sync(self):

        settings = self.world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        self.world.apply_settings(settings)

        self.traffic_manager = (
            self.client.get_trafficmanager(8000)
        )

        self.traffic_manager.set_synchronous_mode(
            True
        )

        self.traffic_manager.set_random_device_seed(
            random.randint(0, 100000)
        )

    # ==================================================================
    # 加载地图
    # ==================================================================

    def load_map(self, map_name):

        print("=" * 70)

        print(
            f"Loading map: {map_name}"
        )

        print("=" * 70)

        try:

            self.world = self.client.load_world(
                map_name,
                reset_settings=False
            )

            # 旧场景Actor已经随着地图切换被销毁
            self.ego_vehicle = None
            self.camera = None
            self.instance_camera = None

            self.vehicles.clear()
            self.walkers.clear()
            self.walker_controllers.clear()

            # 重新配置同步模式
            settings = self.world.get_settings()

            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05

            self.world.apply_settings(
                settings
            )

            self.traffic_manager = (
                self.client.get_trafficmanager(8000)
            )

            self.traffic_manager.set_synchronous_mode(
                True
            )

            self.traffic_manager.set_random_device_seed(
                random.randint(0, 100000)
            )

            # 等待地图稳定
            for _ in range(20):

                self.world.tick()

            print(
                "Map loaded:",
                self.world.get_map().name
            )

            return True

        except Exception as e:

            print(
                f"Map loading error: {e}"
            )

            return False

    # ==================================================================
    # 设置天气
    # ==================================================================

    def set_weather(self, weather):

        self.world.set_weather(
            weather
        )

    # ==================================================================
    # 清空RGB Queue
    # ==================================================================

    def clear_image_queue(self):

        while True:

            try:

                self.image_queue.get_nowait()

            except queue.Empty:

                break

    # ==================================================================
    # 清空Instance Queue
    # ==================================================================

    def clear_instance_queue(self):

        while True:

            try:

                self.instance_queue.get_nowait()

            except queue.Empty:

                break

    # ==================================================================
    # 清空全部Queue
    # ==================================================================

    def clear_queues(self):

        self.clear_image_queue()
        self.clear_instance_queue()

    # ==================================================================
    # RGB Camera Callback
    # ==================================================================

    def camera_callback(self, image):

        try:

            if self.image_queue.full():

                try:

                    self.image_queue.get_nowait()

                except queue.Empty:

                    pass

            self.image_queue.put_nowait(
                image
            )

        except queue.Full:

            pass

    # ==================================================================
    # Instance Camera Callback
    # ==================================================================

    def instance_callback(self, image):

        try:

            if self.instance_queue.full():

                try:

                    self.instance_queue.get_nowait()

                except queue.Empty:

                    pass

            self.instance_queue.put_nowait(
                image
            )

        except queue.Full:

            pass

    # ==================================================================
    # Spawn Ego Vehicle
    # ==================================================================

    def spawn_ego(self):

        library = (
            self.world.get_blueprint_library()
        )

        preferred = [

            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.lincoln.mkz_2020",
            "vehicle.toyota.prius"
        ]

        ego_bp = None

        for name in preferred:

            try:

                ego_bp = library.find(name)

                break

            except Exception:

                pass

        if ego_bp is None:

            ego_bp = random.choice(
                library.filter("vehicle.*")
            )

        spawn_points = (
            self.world.get_map().get_spawn_points()
        )

        random.shuffle(
            spawn_points
        )

        for transform in spawn_points:

            vehicle = (
                self.world.try_spawn_actor(
                    ego_bp,
                    transform
                )
            )

            if vehicle is not None:

                self.ego_vehicle = vehicle

                self.ego_vehicle.set_autopilot(
                    True,
                    self.traffic_manager.get_port()
                )

                print(
                    f"Ego vehicle: "
                    f"{vehicle.type_id}"
                )

                return True

        return False

    # ==================================================================
    # RGB Camera
    # ==================================================================

    def spawn_camera(self):

        library = (
            self.world.get_blueprint_library()
        )

        camera_bp = library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(self.image_width)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(self.image_height)
        )

        camera_bp.set_attribute(
            "fov",
            str(self.camera_fov)
        )

        camera_bp.set_attribute(
            "sensor_tick",
            "0.05"
        )

        transform = carla.Transform(

            carla.Location(
                x=1.55,
                y=0.0,
                z=1.55
            ),

            carla.Rotation(
                pitch=-2.0,
                yaw=0.0,
                roll=0.0
            )
        )

        self.camera = self.world.spawn_actor(

            camera_bp,
            transform,

            attach_to=self.ego_vehicle,

            attachment_type=(
                carla.AttachmentType.Rigid
            )
        )

        self.camera.listen(
            self.camera_callback
        )

        print(
            "RGB camera spawned."
        )

        return transform

    # ==================================================================
    # Instance Segmentation Camera
    # ==================================================================

    def spawn_instance_camera(
        self,
        transform
    ):

        library = (
            self.world.get_blueprint_library()
        )

        instance_bp = library.find(
            "sensor.camera.instance_segmentation"
        )

        instance_bp.set_attribute(
            "image_size_x",
            str(self.image_width)
        )

        instance_bp.set_attribute(
            "image_size_y",
            str(self.image_height)
        )

        instance_bp.set_attribute(
            "fov",
            str(self.camera_fov)
        )

        instance_bp.set_attribute(
            "sensor_tick",
            "0.05"
        )

        self.instance_camera = (
            self.world.spawn_actor(

                instance_bp,
                transform,

                attach_to=self.ego_vehicle,

                attachment_type=(
                    carla.AttachmentType.Rigid
                )
            )
        )

        self.instance_camera.listen(
            self.instance_callback
        )

        print(
            "Instance segmentation camera spawned."
        )

    # ==================================================================
    # Spawn Vehicles
    # ==================================================================

    def spawn_vehicles(self, number):

        print(
            f"Spawning {number} vehicles..."
        )

        library = (
            self.world.get_blueprint_library()
        )

        blueprints = (
            library.filter("vehicle.*")
        )

        spawn_points = (
            self.world.get_map().get_spawn_points()
        )

        random.shuffle(
            spawn_points
        )

        count = 0

        for transform in spawn_points:

            if count >= number:
                break

            if self.ego_vehicle is not None:

                distance = (
                    transform.location.distance(
                        self.ego_vehicle.get_location()
                    )
                )

                if distance < 15:

                    continue

            bp = random.choice(
                blueprints
            )

            if bp.has_attribute(
                "number_of_wheels"
            ):

                wheels = (
                    bp.get_attribute(
                        "number_of_wheels"
                    ).as_int()
                )

                if wheels != 4:

                    continue

            vehicle = (
                self.world.try_spawn_actor(
                    bp,
                    transform
                )
            )

            if vehicle is not None:

                self.vehicles.append(
                    vehicle
                )

                vehicle.set_autopilot(
                    True,
                    self.traffic_manager.get_port()
                )

                self.traffic_manager.\
                    vehicle_percentage_speed_difference(
                        vehicle,
                        random.uniform(-25, 10)
                    )

                count += 1

        print(
            f"Actual vehicles: {count}"
        )

    # ==================================================================
    # Spawn Pedestrians
    # ==================================================================

    def spawn_walkers(self, number):

        print(
            f"Spawning {number} pedestrians..."
        )

        library = (
            self.world.get_blueprint_library()
        )

        walker_bps = (
            library.filter(
                "walker.pedestrian.*"
            )
        )

        controller_bp = library.find(
            "controller.ai.walker"
        )

        for _ in range(number * 3):

            if len(self.walkers) >= number:

                break

            location = (
                self.world.get_random_location_from_navigation()
            )

            if location is None:

                continue

            transform = carla.Transform(
                location
            )

            bp = random.choice(
                walker_bps
            )

            if bp.has_attribute(
                "is_invincible"
            ):

                bp.set_attribute(
                    "is_invincible",
                    "false"
                )

            walker = (
                self.world.try_spawn_actor(
                    bp,
                    transform
                )
            )

            if walker is not None:

                self.walkers.append(
                    walker
                )

        self.world.tick()

        for walker in self.walkers:

            controller = (
                self.world.spawn_actor(
                    controller_bp,
                    carla.Transform(),
                    attach_to=walker
                )
            )

            if controller is not None:

                self.walker_controllers.append(
                    controller
                )

        for controller in self.walker_controllers:

            controller.start()

            destination = (
                self.world.get_random_location_from_navigation()
            )

            if destination is not None:

                controller.go_to_location(
                    destination
                )

            controller.set_max_speed(
                random.uniform(1.0, 2.5)
            )

        print(
            f"Actual pedestrians: "
            f"{len(self.walkers)}"
        )

    # ==================================================================
    # Warmup
    # ==================================================================

    def warmup(self, frames=80):

        print(
            "Warming up..."
        )

        self.clear_queues()

        for _ in range(frames):

            self.world.tick()

        self.clear_queues()

        print(
            "Warmup finished."
        )

    # ==================================================================
    # 获取Sensor Frame
    # ==================================================================

    def get_sensor_frame(
        self,
        sensor_queue,
        target_frame,
        timeout=2.0
    ):

        start_time = time.time()

        while (
            time.time() - start_time
            < timeout
        ):

            try:

                image = sensor_queue.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            if image.frame < target_frame:

                continue

            if image.frame == target_frame:

                return image

            if image.frame > target_frame:

                return image

        return None

    # ==================================================================
    # 获取同步图像
    # ==================================================================

    def get_synchronized_images(
        self,
        frame_id
    ):

        rgb = self.get_sensor_frame(
            self.image_queue,
            frame_id
        )

        instance = self.get_sensor_frame(
            self.instance_queue,
            frame_id
        )

        if (
            rgb is None
            or instance is None
        ):

            return None, None

        return rgb, instance

    # ==================================================================
    # CARLA Image -> BGR
    # ==================================================================

    def image_to_bgr(self, image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = array.reshape(
            (
                image.height,
                image.width,
                4
            )
        )

        return array[:, :, :3]

    # ==================================================================
    # Instance Segmentation解码
    # ==================================================================

    def decode_instance_image(
        self,
        image
    ):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = array.reshape(
            (
                image.height,
                image.width,
                4
            )
        )

        # CARLA instance segmentation:
        #
        # R -> semantic ID
        # G/B -> instance ID
        #
        semantic_ids = array[:, :, 2]

        instance_ids = (
            array[:, :, 1].astype(np.uint16)
            +
            (
                array[:, :, 0].astype(np.uint16)
                << 8
            )
        )

        return (
            semantic_ids,
            instance_ids
        )

    # ==================================================================
    # Camera Projection Matrix
    # ==================================================================

    def build_projection_matrix(self):

        focal = (
            self.image_width
            /
            (
                2.0
                *
                math.tan(
                    math.radians(
                        self.camera_fov
                    )
                    /
                    2.0
                )
            )
        )

        K = np.identity(
            3,
            dtype=np.float32
        )

        K[0, 0] = focal
        K[1, 1] = focal

        K[0, 2] = (
            self.image_width / 2.0
        )

        K[1, 2] = (
            self.image_height / 2.0
        )

        return K

    # ==================================================================
    # World -> Camera
    # ==================================================================

    def world_to_camera(self, point):

        transform = (
            self.camera.get_transform()
        )

        matrix = np.array(
            transform.get_inverse_matrix()
        )

        point_world = np.array([
            point.x,
            point.y,
            point.z,
            1.0
        ])

        point_camera = (
            matrix.dot(
                point_world
            )
        )

        return np.array([
            point_camera[1],
            -point_camera[2],
            point_camera[0]
        ])

    # ==================================================================
    # 投影到图像
    # ==================================================================

    def project_point(self, point):

        camera_point = (
            self.world_to_camera(
                point
            )
        )

        if camera_point[2] <= 0.1:

            return None

        K = (
            self.build_projection_matrix()
        )

        projected = (
            K.dot(camera_point)
        )

        projected[0] /= projected[2]
        projected[1] /= projected[2]

        return (
            float(projected[0]),
            float(projected[1]),
            float(camera_point[2])
        )

    # ==================================================================
    # Actor 3D Bounding Box -> 2D
    # ==================================================================

    def project_actor_bbox(
        self,
        actor
    ):

        try:

            vertices = (
                actor.bounding_box.get_world_vertices(
                    actor.get_transform()
                )
            )

            projected = []

            for vertex in vertices:

                p = self.project_point(
                    vertex
                )

                if p is not None:

                    projected.append(p)

            if len(projected) < 2:

                return None

            xs = [
                p[0]
                for p in projected
            ]

            ys = [
                p[1]
                for p in projected
            ]

            depths = [
                p[2]
                for p in projected
            ]

            return {

                "x_min": min(xs),
                "y_min": min(ys),

                "x_max": max(xs),
                "y_max": max(ys),

                "depth_min": min(depths),
                "depth_max": max(depths)
            }

        except Exception:

            return None

    # ==================================================================
    # Clip Bounding Box
    # ==================================================================

    def clip_bbox(self, bbox):

        x1 = bbox["x_min"]
        y1 = bbox["y_min"]

        x2 = bbox["x_max"]
        y2 = bbox["y_max"]

        full_width = (
            x2 - x1
        )

        full_height = (
            y2 - y1
        )

        if (
            full_width <= 0
            or full_height <= 0
        ):

            return None

        full_area = (
            full_width
            *
            full_height
        )

        cx1 = max(
            0.0,
            x1
        )

        cy1 = max(
            0.0,
            y1
        )

        cx2 = min(
            float(
                self.image_width - 1
            ),
            x2
        )

        cy2 = min(
            float(
                self.image_height - 1
            ),
            y2
        )

        if (
            cx2 <= cx1
            or cy2 <= cy1
        ):

            return None

        clipped_width = (
            cx2 - cx1
        )

        clipped_height = (
            cy2 - cy1
        )

        clipped_area = (
            clipped_width
            *
            clipped_height
        )

        truncation_ratio = (
            1.0
            -
            clipped_area / full_area
        )

        return {

            "x1": cx1,
            "y1": cy1,

            "x2": cx2,
            "y2": cy2,

            "full_area": full_area,
            "clipped_area": clipped_area,

            "truncation_ratio":
                truncation_ratio
        }

    # ==================================================================
    # 获取类别
    # ==================================================================

    def get_class_id(self, actor):

        type_id = (
            actor.type_id.lower()
        )

        if type_id.startswith(
            "walker.pedestrian"
        ):

            return 0

        if "bicycle" in type_id:

            return 1

        if "motorcycle" in type_id:

            return 3

        if "bus" in type_id:

            return 5

        if (
            "truck" in type_id
            or "carlacola" in type_id
            or "firetruck" in type_id
        ):

            return 7

        if type_id.startswith(
            "vehicle."
        ):

            return 2

        return None

    # ==================================================================
    # 获取Actor可见Mask
    # ==================================================================

    def get_actor_visible_mask(
        self,
        actor,
        instance_ids
    ):

        actor_id = int(
            actor.id
        )

        return (
            instance_ids == actor_id
        )

    # ==================================================================
    # Mask -> Bounding Box
    # ==================================================================

    def mask_to_bbox(
        self,
        mask
    ):

        ys, xs = np.where(
            mask
        )

        if len(xs) == 0:

            return None

        x1 = int(
            xs.min()
        )

        x2 = int(
            xs.max()
        )

        y1 = int(
            ys.min()
        )

        y2 = int(
            ys.max()
        )

        width = (
            x2 - x1 + 1
        )

        height = (
            y2 - y1 + 1
        )

        area = len(xs)

        bbox_area = (
            width * height
        )

        if width < self.min_bbox_width:

            return None

        if height < self.min_bbox_height:

            return None

        if area < self.min_visible_pixels:

            return None

        return {

            "x1": x1,
            "y1": y1,

            "x2": x2,
            "y2": y2,

            "width": width,
            "height": height,

            "visible_pixels": area,
            "bbox_area": bbox_area
        }

    # ==================================================================
    # 截断检查
    # ==================================================================

    def check_truncation(
        self,
        projected_bbox
    ):

        clipped = (
            self.clip_bbox(
                projected_bbox
            )
        )

        if clipped is None:

            return None

        truncation_ratio = (
            clipped["truncation_ratio"]
        )

        if (
            truncation_ratio
            >
            self.max_truncation_ratio
        ):

            return None

        return clipped

    # ==================================================================
    # 距离检查
    # ==================================================================

    def check_distance(
        self,
        actor
    ):

        if self.camera is None:

            return False

        camera_location = (
            self.camera
            .get_transform()
            .location
        )

        actor_location = (
            actor
            .get_transform()
            .location
        )

        distance = (
            camera_location.distance(
                actor_location
            )
        )

        if (
            distance
            <
            self.min_object_distance
        ):

            return False

        if (
            distance
            >
            self.max_object_distance
        ):

            return False

        return True

    # ==================================================================
    # 生成Labels
    # ==================================================================

    def collect_labels(
        self,
        instance_image
    ):

        semantic_ids, instance_ids = (
            self.decode_instance_image(
                instance_image
            )
        )

        labels = []

        vehicle_count = 0
        pedestrian_count = 0

        actors = []

        actors.extend(
            self.vehicles
        )

        actors.extend(
            self.walkers
        )

        for actor in actors:

            try:

                if not actor.is_alive:

                    continue

            except Exception:

                continue

            class_id = (
                self.get_class_id(
                    actor
                )
            )

            if class_id is None:

                continue

            # ----------------------------------------------------------
            # 距离过滤
            # ----------------------------------------------------------

            if not self.check_distance(
                actor
            ):

                continue

            # ----------------------------------------------------------
            # 3D Bounding Box投影
            # ----------------------------------------------------------

            projected_bbox = (
                self.project_actor_bbox(
                    actor
                )
            )

            if projected_bbox is None:

                continue

            # ----------------------------------------------------------
            # 截断过滤
            # ----------------------------------------------------------

            clipped = (
                self.check_truncation(
                    projected_bbox
                )
            )

            if clipped is None:

                continue

            # ----------------------------------------------------------
            # Instance Segmentation
            # ----------------------------------------------------------

            mask = (
                self.get_actor_visible_mask(
                    actor,
                    instance_ids
                )
            )

            visible_bbox = (
                self.mask_to_bbox(
                    mask
                )
            )

            if visible_bbox is None:

                continue

            visible_pixels = (
                visible_bbox[
                    "visible_pixels"
                ]
            )

            # ----------------------------------------------------------
            # 可见比例
            # ----------------------------------------------------------

            visible_ratio = (
                visible_pixels
                /
                clipped["clipped_area"]
            )

            if (
                visible_ratio
                <
                self.min_visible_ratio
            ):

                continue

            # ----------------------------------------------------------
            # Mask填充比例
            # ----------------------------------------------------------

            mask_fill_ratio = (
                visible_pixels
                /
                visible_bbox["bbox_area"]
            )

            if (
                mask_fill_ratio
                <
                self.min_mask_fill_ratio
            ):

                continue

            # ----------------------------------------------------------
            # 最小Bounding Box
            # ----------------------------------------------------------

            if (
                visible_bbox["width"]
                <
                self.min_bbox_width
            ):

                continue

            if (
                visible_bbox["height"]
                <
                self.min_bbox_height
            ):

                continue

            # ----------------------------------------------------------
            # 统计类别
            # ----------------------------------------------------------

            if class_id in (
                1,
                2,
                3,
                5,
                7
            ):

                vehicle_count += 1

            elif class_id == 0:

                pedestrian_count += 1

            # ----------------------------------------------------------
            # 保存Label
            # ----------------------------------------------------------

            labels.append({

                "class_id": class_id,

                "x1": visible_bbox["x1"],
                "y1": visible_bbox["y1"],

                "x2": visible_bbox["x2"],
                "y2": visible_bbox["y2"],

                "visible_pixels":
                    visible_pixels,

                "visible_ratio":
                    visible_ratio,

                "truncation_ratio":
                    clipped[
                        "truncation_ratio"
                    ],

                "mask_fill_ratio":
                    mask_fill_ratio
            })

        return (
            labels,
            vehicle_count,
            pedestrian_count
        )

    # ==================================================================
    # BBox -> YOLO格式
    # ==================================================================

    def bbox_to_yolo(
        self,
        label
    ):

        x1 = label["x1"]
        y1 = label["y1"]

        x2 = label["x2"]
        y2 = label["y2"]

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        width = (
            x2 - x1
        )

        height = (
            y2 - y1
        )

        center_x /= (
            self.image_width
        )

        center_y /= (
            self.image_height
        )

        width /= (
            self.image_width
        )

        height /= (
            self.image_height
        )

        return (
            f"{label['class_id']} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # ==================================================================
    # 保存图片和Label
    # ==================================================================

    def save_sample(
        self,
        image,
        labels,
        map_name,
        weather_name,
        traffic_name
    ):

        self.image_count += 1

        filename = (
            f"{self.image_count:06d}"
        )

        image_path = os.path.join(
            self.image_dir,
            filename + ".jpg"
        )

        label_path = os.path.join(
            self.label_dir,
            filename + ".txt"
        )

        frame = (
            self.image_to_bgr(
                image
            )
        )

        success = cv2.imwrite(

            image_path,
            frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:

            self.image_count -= 1

            return False

        # --------------------------------------------------------------
        # YOLO Label
        # --------------------------------------------------------------

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            for label in labels:

                f.write(
                    self.bbox_to_yolo(
                        label
                    )
                    +
                    "\n"
                )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        with open(
            self.metadata_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(
                f
            )

            writer.writerow([

                filename + ".jpg",

                filename + ".txt",

                map_name,

                weather_name,

                traffic_name,

                image.frame,

                len(labels)
            ])

        print(
            f"[{self.image_count:03d}/"
            f"{self.target_images}] "
            f"{filename}.jpg | "
            f"objects={len(labels)} | "
            f"frame={image.frame}"
        )

        for i, label in enumerate(
            labels
        ):

            print(

                f"    obj{i}: "
                f"class={label['class_id']} "
                f"visible="
                f"{label['visible_ratio']:.2f} "
                f"trunc="
                f"{label['truncation_ratio']:.2f} "
                f"fill="
                f"{label['mask_fill_ratio']:.2f}"
            )

        return True

    # ==================================================================
    # 运行一个场景
    # ==================================================================

    def run_scene(
        self,
        map_name,
        weather_name,
        weather,
        traffic_name,
        traffic_config
    ):

        print("-" * 70)

        print(
            f"Map     : {map_name}"
        )

        print(
            f"Weather : {weather_name}"
        )

        print(
            f"Traffic : {traffic_name}"
        )

        print("-" * 70)

        # --------------------------------------------------------------
        # 重置场景统计
        # --------------------------------------------------------------

        self.scene_frames = 0
        self.scene_image_count = 0
        self.last_saved_frame = None

        # --------------------------------------------------------------
        # 设置天气
        # --------------------------------------------------------------

        self.set_weather(
            weather
        )

        # --------------------------------------------------------------
        # Ego
        # --------------------------------------------------------------

        if not self.spawn_ego():

            raise RuntimeError(
                "Failed to spawn ego."
            )

        # --------------------------------------------------------------
        # Camera
        # --------------------------------------------------------------

        camera_transform = (
            self.spawn_camera()
        )

        self.spawn_instance_camera(
            camera_transform
        )

        # --------------------------------------------------------------
        # Traffic
        # --------------------------------------------------------------

        self.spawn_vehicles(
            traffic_config["vehicles"]
        )

        self.spawn_walkers(
            traffic_config["walkers"]
        )

        # --------------------------------------------------------------
        # Warmup
        # --------------------------------------------------------------

        self.warmup(
            frames=80
        )

        # --------------------------------------------------------------
        # 获取当前CARLA frame
        # --------------------------------------------------------------

        current_frame = (
            self.world
            .get_snapshot()
            .frame
        )

        # --------------------------------------------------------------
        # 第一张图片也需要等待至少500 frame
        # --------------------------------------------------------------

        self.next_save_frame = (
            current_frame
            +
            self.min_frame_gap
        )

        print(
            f"Sampling starts at "
            f"frame >= {self.next_save_frame}"
        )

        # ==============================================================
        # 场景循环
        # ==============================================================

        while (

            self.scene_frames
            <
            self.max_scene_frames

            and

            self.image_count
            <
            self.target_images

            and

            self.scene_image_count
            <
            self.max_images_per_scene
        ):

            # ----------------------------------------------------------
            # CARLA推进一帧
            # ----------------------------------------------------------

            current_frame = (
                self.world.tick()
            )

            self.scene_frames += 1

            # ----------------------------------------------------------
            # 还没有达到下一次允许采样的frame
            # ----------------------------------------------------------

            if (
                current_frame
                <
                self.next_save_frame
            ):

                continue

            # ----------------------------------------------------------
            # 获取RGB + Instance
            # ----------------------------------------------------------

            rgb_image, instance_image = (
                self.get_synchronized_images(
                    current_frame
                )
            )

            if (
                rgb_image is None
                or
                instance_image is None
            ):

                print(
                    "Warning: "
                    "sensor synchronization "
                    f"timeout at frame "
                    f"{current_frame}"
                )

                continue

            # ----------------------------------------------------------
            # 生成Label
            # ----------------------------------------------------------

            (
                labels,
                vehicle_count,
                pedestrian_count
            ) = self.collect_labels(
                instance_image
            )

            # ----------------------------------------------------------
            # 没有有效目标
            #
            # 注意：
            # 这里不修改next_save_frame
            #
            # 因为500帧间隔是针对“成功保存图片”
            # 而不是针对“尝试采样”
            # ----------------------------------------------------------

            if (
                len(labels)
                <
                self.min_objects
            ):

                continue

            # ----------------------------------------------------------
            # 要求至少有一个车辆
            # ----------------------------------------------------------

            if (
                self.require_vehicle
                and
                vehicle_count == 0
            ):

                continue

            # ----------------------------------------------------------
            # 再次检查上一张成功保存图片的frame
            #
            # 这是最后一道保险。
            # ----------------------------------------------------------

            if (
                self.last_saved_frame
                is not None
            ):

                frame_gap = (
                    current_frame
                    -
                    self.last_saved_frame
                )

                if (
                    frame_gap
                    <
                    self.min_frame_gap
                ):

                    continue

            # ----------------------------------------------------------
            # 保存
            # ----------------------------------------------------------

            success = self.save_sample(

                rgb_image,

                labels,

                map_name,

                weather_name,

                traffic_name
            )

            # ----------------------------------------------------------
            # 保存成功
            # ----------------------------------------------------------

            if success:

                self.scene_image_count += 1

                # ------------------------------------------------------
                # 记录成功保存的frame
                # ------------------------------------------------------

                self.last_saved_frame = (
                    current_frame
                )

                # ------------------------------------------------------
                # 下一张至少500帧
                #
                # 实际随机：
                #
                # 500 ~ 700 frame
                # ------------------------------------------------------

                frame_gap = random.randint(

                    self.min_frame_gap,

                    self.max_frame_gap
                )

                self.next_save_frame = (

                    current_frame
                    +
                    frame_gap
                )

                print(
                    f"  Saved frame: "
                    f"{current_frame}"
                )

                print(
                    f"  Gap to next "
                    f"sample: "
                    f"{frame_gap} frames"
                )

                print(
                    f"  Next sampling "
                    f"frame: "
                    f"{self.next_save_frame}"
                )

                print(
                    f"  Scene progress: "
                    f"{self.scene_image_count}/"
                    f"{self.max_images_per_scene}"
                )

        print(
            f"Scene finished. "
            f"Scene images: "
            f"{self.scene_image_count}"
        )

        if (
            self.last_saved_frame
            is not None
        ):

            print(
                f"Last saved frame: "
                f"{self.last_saved_frame}"
            )

    # ==================================================================
    # 清理当前场景
    # ==================================================================

    def cleanup_scene(self):

        print(
            "Cleaning scene..."
        )

        # --------------------------------------------------------------
        # 停止Sensor
        # --------------------------------------------------------------

        for sensor in [

            self.camera,
            self.instance_camera

        ]:

            if sensor is not None:

                try:

                    sensor.stop()

                except Exception:

                    pass

        # --------------------------------------------------------------
        # 收集Actor
        # --------------------------------------------------------------

        actors = []

        if self.camera is not None:

            actors.append(
                self.camera
            )

        if self.instance_camera is not None:

            actors.append(
                self.instance_camera
            )

        if self.ego_vehicle is not None:

            actors.append(
                self.ego_vehicle
            )

        actors.extend(
            self.vehicles
        )

        actors.extend(
            self.walker_controllers
        )

        actors.extend(
            self.walkers
        )

        # --------------------------------------------------------------
        # 去重
        # --------------------------------------------------------------

        unique = {}

        for actor in actors:

            if actor is not None:

                unique[
                    actor.id
                ] = actor

        actors = list(
            unique.values()
        )

        # --------------------------------------------------------------
        # 批量销毁
        # --------------------------------------------------------------

        if actors:

            try:

                commands = [

                    carla.command.DestroyActor(
                        actor.id
                    )

                    for actor in actors
                ]

                self.client.apply_batch(
                    commands
                )

            except Exception as e:

                print(
                    f"Destroy warning: "
                    f"{e}"
                )

        # --------------------------------------------------------------
        # 清空引用
        # --------------------------------------------------------------

        self.camera = None
        self.instance_camera = None
        self.ego_vehicle = None

        self.vehicles.clear()
        self.walkers.clear()
        self.walker_controllers.clear()

        # --------------------------------------------------------------
        # 清空Queue
        # --------------------------------------------------------------

        self.clear_queues()

        # --------------------------------------------------------------
        # 推进一步
        # --------------------------------------------------------------

        try:

            self.world.tick()

        except Exception:

            pass

    # ==================================================================
    # 恢复CARLA
    # ==================================================================

    def restore_world(self):

        print(
            "Restoring CARLA..."
        )

        try:

            # ----------------------------------------------------------
            # Traffic Manager必须先退出同步模式
            # ----------------------------------------------------------

            if (
                self.traffic_manager
                is not None
            ):

                try:

                    self.traffic_manager.\
                        set_synchronous_mode(
                            False
                        )

                except Exception:

                    pass

            # ----------------------------------------------------------
            # CARLA退出同步模式
            # ----------------------------------------------------------

            if self.world is not None:

                settings = (
                    self.world.get_settings()
                )

                settings.synchronous_mode = (
                    False
                )

                settings.fixed_delta_seconds = (
                    None
                )

                self.world.apply_settings(
                    settings
                )

        except Exception as e:

            print(
                f"Restore warning: "
                f"{e}"
            )

    # ==================================================================
    # 主运行函数
    # ==================================================================

    def run(self):

        try:

            # ----------------------------------------------------------
            # 连接
            # ----------------------------------------------------------

            self.connect()

            # ----------------------------------------------------------
            # 检测地图
            # ----------------------------------------------------------

            if not self.detect_maps():

                return

            # ----------------------------------------------------------
            # 同步模式
            # ----------------------------------------------------------

            self.configure_sync()

            # ----------------------------------------------------------
            # 创建场景列表
            # ----------------------------------------------------------

            scenarios = []

            for map_name in self.selected_maps:

                # 每张地图随机选择3种天气
                selected_weather = random.sample(

                    self.weather_presets,

                    3
                )

                for weather_name, weather in (
                    selected_weather
                ):

                    for (
                        traffic_name,
                        traffic_config
                    ) in (
                        self.traffic_levels.items()
                    ):

                        scenarios.append(

                            (
                                map_name,

                                weather_name,

                                weather,

                                traffic_name,

                                traffic_config
                            )
                        )

            # ----------------------------------------------------------
            # 场景随机打乱
            # ----------------------------------------------------------

            random.shuffle(
                scenarios
            )

            print("=" * 70)

            print(
                f"Total scenarios: "
                f"{len(scenarios)}"
            )

            print(
                f"Target images: "
                f"{self.target_images}"
            )

            print(
                f"Max images per scene: "
                f"{self.max_images_per_scene}"
            )

            print(
                f"Frame gap: "
                f"{self.min_frame_gap}"
                f"~"
                f"{self.max_frame_gap}"
            )

            print("=" * 70)

            # ==========================================================
            # 开始场景
            # ==========================================================

            for index, scene in enumerate(

                scenarios,
                start=1
            ):

                # ------------------------------------------------------
                # 总图片达到目标
                # ------------------------------------------------------

                if (
                    self.image_count
                    >=
                    self.target_images
                ):

                    break

                print(
                    f"########## Scene "
                    f"{index}/"
                    f"{len(scenarios)} "
                    f"##########"
                )

                (
                    map_name,
                    weather_name,
                    weather,
                    traffic_name,
                    traffic_config
                ) = scene

                try:

                    # --------------------------------------------------
                    # 加载地图
                    # --------------------------------------------------

                    if not self.load_map(
                        map_name
                    ):

                        continue

                    # --------------------------------------------------
                    # 运行场景
                    # --------------------------------------------------

                    self.run_scene(

                        map_name,

                        weather_name,

                        weather,

                        traffic_name,

                        traffic_config
                    )

                except KeyboardInterrupt:

                    raise

                except Exception as e:

                    print(
                        f"Scene error: "
                        f"{e}"
                    )

                    import traceback

                    traceback.print_exc()

                finally:

                    # --------------------------------------------------
                    # 清理场景
                    # --------------------------------------------------

                    self.cleanup_scene()

            # ==========================================================
            # 完成
            # ==========================================================

            print("=" * 70)

            print(
                "DATASET COLLECTION FINISHED"
            )

            print("=" * 70)

            print(
                f"Images: "
                f"{self.image_count}"
            )

            print(
                "Images:",
                os.path.abspath(
                    self.image_dir
                )
            )

            print(
                "Labels:",
                os.path.abspath(
                    self.label_dir
                )
            )

            print(
                "Metadata:",
                os.path.abspath(
                    self.metadata_path
                )
            )

        except KeyboardInterrupt:

            print(
                "Interrupted by user."
            )

        except Exception as e:

            print(
                f"Fatal error: "
                f"{e}"
            )

            import traceback

            traceback.print_exc()

        finally:

            # ----------------------------------------------------------
            # 最终清理
            # ----------------------------------------------------------

            self.cleanup_scene()

            self.restore_world()

            print(
                "CARLA restored safely."
            )


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

    collector = CarlaYOLODatasetCollector(

        host="127.0.0.1",

        port=2000,

        output_dir="Calibration",

        target_images=500
    )

    collector.run()
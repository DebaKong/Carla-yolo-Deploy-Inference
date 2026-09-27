import carla

class CarlaWorld:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000):
        self.host = host
        self.port = port

        self.client = None
        self.world = None
        self.vehicle = None

    def connect(self):
        self.client = carla.Client(
            self.host,
            self.port
        )

        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        print(
            f"[CARLA] Connected:"
            f"{self.host}:{self.port}"
        )

        return self.world

    def spawn_vehicle(self):
        if self.world is None:
            raise RuntimeError("World is not connected.")

        blueprint_library = self.world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter(
            "vehicle.tesla.model3"
        )[0]

        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No spawn points available.")

        self.vehicle = self.world.spawn_actor(
            vehicle_bp,
            spawn_points[0]
        )

        print(
            f"[CARLA] Vehicle spawned:"
            f"{self.vehicle.id}"
        )

        return self.vehicle

    def cleanup(self):
        if self.vehicle is not None:
            self.vehicle.destroy()  
            print(f"[CARLA] Vehicle destroyed: {self.vehicle.id}")
            self.vehicle = None

        if self.client is not None:
            self.client = None
            print("[CARLA] Client disconnected.")
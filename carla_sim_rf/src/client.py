import random
import carla
import numpy as np



class Client:
    client: carla.Client
    world: carla.World
    map: carla.Map
    spectator: carla.Actor
    blueprints: carla.BlueprintLibrary
    traffic_manager: carla.TrafficManager
    sync: bool
    VehicleNPC : "list[carla.Vehicle]"
    PedestrianNPC : "list[carla.Walker]" 

    def __init__(self, sync=True, seed: "int|None" = None, dt=0.1, phys_dt=0.01, phys_substeps=10):
        Client.sync = sync
        self.seed = seed
        self.dt = dt
        self.phys_dt = phys_dt
        self.phys_substeps = phys_substeps
        self.VehicleNPC = []
        self.PedestrianNPC = []
        self._connect()
        self.removeStandByVehicle()
        if seed:  # for reproducibility
            random.seed(seed)
            np.random.seed(seed)
        if sync:
            self._synchronize_world(seed)

    def disable_sync(self):
        """Disable synchronous mode but keep settings"""
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        Client.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(settings)

    def _connect(self):
        Client.client = carla.Client("localhost", 2000)
        Client.client.set_timeout(20)
        Client.world = Client.client.get_world()
        Client.map = Client.world.get_map()
        Client.spectator = Client.world.get_spectator()
        Client.blueprints = Client.world.get_blueprint_library()
        Client.traffic_manager = Client.client.get_trafficmanager(8000)

    def _synchronize_world(self, seed: "int|None"):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.substepping = True
        settings.max_substep_delta_time = self.phys_dt
        settings.max_substeps = self.phys_substeps

        Client.traffic_manager.set_synchronous_mode(True)
        if seed:
            Client.traffic_manager.set_random_device_seed(seed)
        Client.world.apply_settings(settings)

    def weather(self,type:str):
        if(type == "ClearSunset"):
            Client.world.set_weather(carla.WeatherParameters.ClearSunset)
        elif(type == "CloudyNoon"):
            Client.world.set_weather(carla.WeatherParameters.CloudyNoon)
        elif(type == "MidRainyNoon"):
            Client.world.set_weather(carla.WeatherParameters.MidRainyNoon)
    
    def removeStandByVehicle(self):
        vehicle_labels = [carla.CityObjectLabel.Car, carla.CityObjectLabel.Motorcycle]
        for vehicle_label in vehicle_labels:
            all_vehicles_in_map = Client.world.get_environment_objects(vehicle_label)
            print(len(all_vehicles_in_map))
            processed_vehicles = set()
            for single_vehicle in all_vehicles_in_map:
                processed_vehicles.add(single_vehicle.id)
            Client.world.enable_environment_objects(processed_vehicles, False)

    def destroyNPC(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.VehicleNPC])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.PedestrianNPC])
        
    def reload(self):
        self.destroyNPC()
        Client.client.reload_world()

def simulate_ticks(ticks: int):
    """Simulate given number of ticks(frames) if in synchronous mode."""
    if Client.sync:
        for tick in range(ticks):
            Client.world.tick()

def get_distance_traveled(previous_location: carla.Location, current_location: carla.Location) -> float:
    return np.sqrt(
        (current_location.x - previous_location.x) ** 2
        + (current_location.y - previous_location.y) ** 2
        + (current_location.z - previous_location.z) ** 2
    )  # type: ignore
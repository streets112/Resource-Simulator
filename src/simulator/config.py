from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class SimConfig:
    width: int
    height: int
    timestep_ms: int
    max_steps: int
    simulation_speed: float = 1.0


@dataclass
class TerrainConfig:
    color: tuple[int, int, int]
    resource_regen_rate: float
    resource_capacity: float
    passable: bool
    mobility_prey: float = 1.0
    mobility_predator: float = 1.0
    visibility_prey: float = 1.0
    visibility_predator: float = 1.0


@dataclass
class EnvironmentConfig:
    terrain_types: dict[str, TerrainConfig]
    terrain_distribution: dict[str, float]
    resource_diffusion_rate: float
    ocean_border_width: int = 5
    grass_regrowth_rate: float = 0.02


@dataclass
class GrazerConfig:
    initial_count: int
    max_energy: float
    energy_per_step: float
    energy_from_resource: float
    reproduction_min_age: int
    reproduction_min_energy: float
    reproduction_energy_cost: float
    offspring_initial_energy: float
    move_speed: int
    gradient_weight: float
    gradient_weight_std: float
    vision_radius: int
    max_fat: float = 500.0


@dataclass
class RabbitConfig:
    initial_count: int
    max_energy: float
    energy_per_step: float
    energy_from_resource: float
    reproduction_min_age: int
    reproduction_min_energy: float
    reproduction_energy_cost: float
    offspring_initial_energy: float
    move_speed: int
    gradient_weight: float
    gradient_weight_std: float
    vision_radius: int


@dataclass
class PredatorConfig:
    initial_count: int
    max_energy: float
    energy_per_step: float
    energy_from_prey: float
    reproduction_min_age: int
    reproduction_min_energy: float
    reproduction_energy_cost: float
    offspring_initial_energy: float
    move_speed: int
    chase_radius: int
    resource_sense_radius: int


@dataclass
class VisualizationConfig:
    cell_size: int
    show_grid: bool
    show_stats: bool
    fps: int
    colors: dict[str, list[int]]
    window_width: int = 1024
    window_height: int = 768


@dataclass
class OutputConfig:
    headless: bool
    export_interval: int
    export_path: str
    log_interval: int


@dataclass
class Config:
    simulation: SimConfig
    environment: EnvironmentConfig
    grazer: GrazerConfig
    rabbit: RabbitConfig
    predator: PredatorConfig
    visualization: VisualizationConfig
    output: OutputConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    sim = SimConfig(**raw["simulation"])

    env_terrains = {
        k: TerrainConfig(**v) for k, v in raw["environment"]["terrain_types"].items()
    }
    env = EnvironmentConfig(
        terrain_types=env_terrains,
        terrain_distribution=raw["environment"]["terrain_distribution"],
        resource_diffusion_rate=raw["environment"]["resource_diffusion_rate"],
        ocean_border_width=raw["environment"].get("ocean_border_width", 5),
    )

    grazer = GrazerConfig(**raw["grazer"])
    rabbit = RabbitConfig(**raw["rabbit"])
    predator = PredatorConfig(**raw["predator"])
    viz = VisualizationConfig(**raw["visualization"])
    output = OutputConfig(**raw["output"])

    return Config(
        simulation=sim,
        environment=env,
        grazer=grazer,
        rabbit=rabbit,
        predator=predator,
        visualization=viz,
        output=output,
    )


def save_config(config: Config, path: str | Path = "config.yaml") -> None:
    raw = {
        "simulation": config.simulation.__dict__,
        "environment": {
            "terrain_types": {
                k: v.__dict__ for k, v in config.environment.terrain_types.items()
            },
            "terrain_distribution": config.environment.terrain_distribution,
            "resource_diffusion_rate": config.environment.resource_diffusion_rate,
        },
        "grazer": config.grazer.__dict__,
        "rabbit": config.rabbit.__dict__,
        "predator": config.predator.__dict__,
        "visualization": config.visualization.__dict__,
        "output": config.output.__dict__,
    }
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False)
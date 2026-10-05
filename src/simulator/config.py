from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_OCEAN_BORDER = 5


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
    # Fraction of the post-metabolism surplus a prey actually banks here.
    resource_productivity: float = 1.0


@dataclass
class EnvironmentConfig:
    terrain_types: dict[str, TerrainConfig]
    terrain_distribution: dict[str, float]
    resource_diffusion_rate: float
    ocean_border_width: int = DEFAULT_OCEAN_BORDER
    regen_base_factor: float = 0.025
    depletion_threshold: float = 0.6
    depletion_suppression: float = 2.2
    min_regen_suppression: float = 0.25
    depletion_signal_smoothing: float = 0.7


@dataclass
class PreyConfig:
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
    herd_radius: int = 8
    separation_radius: int = 2
    herd_follow_chance: float = 0.7
    cohesion_weight: float = 0.4
    alignment_weight: float = 0.3
    separation_weight: float = 0.3
    flee_radius: int = 5
    flee_weight: float = 3.0
    min_efficiency: float = 0.3
    spawn_group_size: int = 14
    spawn_group_radius: int = 6

    # Stop-and-go foraging. When migratory is set, the species alternates between
    # travelling (moving while feeding at a reduced rate) and standing still to
    # feed heavily, instead of grazing continuously as it walks.
    migratory: bool = False
    moving_efficiency: float = 0.25
    grazing_efficiency: float = 2.5
    graze_ratio_threshold: float = 0.45
    migrate_ratio_threshold: float = 0.12
    graze_energy_target: float = 1.0
    graze_resume_energy: float = 0.7
    wander_energy_threshold: float = 0.95
    migrating_herd_weight: float = 2.5

    # Movement metabolism, charged per cell advanced on top of energy_per_step.
    movement_energy_cost: float = 0.6
    flee_speed_multiplier: float = 1.3
    flee_cost_multiplier: float = 2.5
    grazing_move_multiplier: float = 0.667


@dataclass
class GrazerConfig(PreyConfig):
    pass


@dataclass
class RabbitConfig(PreyConfig):
    pass


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
    spawn_group_size: int = 6
    spawn_group_radius: int = 8
    pack_radius: int = 10
    pack_hunt_chance: float = 0.6
    flank_chance: float = 0.4
    pack_wander_chance: float = 0.5
    pack_score_bonus: float = 3.0
    depleted_ratio_threshold: float = 0.3
    investigate_steps: int = 12
    investigate_threshold: float = 0.02
    vision_boost_on_investigate: float = 1.5
    movement_energy_cost: float = 0.6
    chase_speed_multiplier: float = 1.2
    chase_cost_multiplier: float = 2.0


@dataclass
class CarcassConfig:
    decay_rate: float = 0.98
    min_energy: float = 1.0
    carcass_energy_fraction: float = 0.5
    consumption_divisor: float = 10.0
    feed_radius: int = 1


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
    carcass: CarcassConfig
    visualization: VisualizationConfig
    output: OutputConfig


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a mapping, ignoring keys the dataclass does not declare."""
    raw = raw or {}
    accepted = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in accepted})


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    sim_raw = dict(raw.get("simulation", {}))
    sim_raw.setdefault("timestep_ms", 100)
    sim_raw.setdefault("max_steps", 0)
    sim = _build(SimConfig, sim_raw)

    env_raw = dict(raw.get("environment", {}))
    terrains = {
        name: _build(TerrainConfig, body)
        for name, body in env_raw.get("terrain_types", {}).items()
    }
    env = _build(
        EnvironmentConfig,
        {
            "terrain_types": terrains,
            "terrain_distribution": env_raw.get("terrain_distribution", {}),
            "resource_diffusion_rate": env_raw.get("resource_diffusion_rate", 0.1),
            "ocean_border_width": env_raw.get("ocean_border_width", DEFAULT_OCEAN_BORDER),
        },
    )

    grazer = _build(GrazerConfig, raw.get("grazer"))
    rabbit = _build(RabbitConfig, raw.get("rabbit"))
    predator = _build(PredatorConfig, raw.get("predator"))
    carcass = _build(CarcassConfig, raw.get("carcass"))
    viz = _build(VisualizationConfig, raw.get("visualization"))
    output = _build(OutputConfig, raw.get("output"))

    return Config(
        simulation=sim,
        environment=env,
        grazer=grazer,
        rabbit=rabbit,
        predator=predator,
        carcass=carcass,
        visualization=viz,
        output=output,
    )


def _dump(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {name: _dump(getattr(obj, name)) for name in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    return obj


def save_config(config: Config, path: str | Path = "config.yaml") -> None:
    payload = {
        "simulation": _dump(config.simulation),
        "environment": {
            "terrain_types": _dump(config.environment.terrain_types),
            "terrain_distribution": config.environment.terrain_distribution,
            "resource_diffusion_rate": config.environment.resource_diffusion_rate,
            "ocean_border_width": config.environment.ocean_border_width,
            "regen_base_factor": config.environment.regen_base_factor,
            "depletion_threshold": config.environment.depletion_threshold,
            "depletion_suppression": config.environment.depletion_suppression,
            "min_regen_suppression": config.environment.min_regen_suppression,
            "depletion_signal_smoothing": config.environment.depletion_signal_smoothing,
        },
        "grazer": _dump(config.grazer),
        "rabbit": _dump(config.rabbit),
        "predator": _dump(config.predator),
        "carcass": _dump(config.carcass),
        "visualization": _dump(config.visualization),
        "output": _dump(config.output),
    }
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
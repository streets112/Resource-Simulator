import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Optional, Protocol

_IDS = count(1)


class EntityPool:
    """Object pool for entity reuse to avoid allocation churn."""

    def __init__(self, factory, max_size: int = 10000):
        self._factory = factory
        self._available = deque()
        self._max_size = max_size
        self._total_created = 0

    def acquire(self, *args, **kwargs):
        if self._available:
            entity = self._available.popleft()
            # Reinitialize with new parameters
            self._factory(entity, *args, **kwargs)
            return entity
        else:
            self._total_created += 1
            return self._factory(None, *args, **kwargs)

    def release(self, entity):
        if entity is not None and len(self._available) < self._max_size:
            # Reset state for reuse
            entity.alive = False
            entity.energy = 0.0
            entity.age = 0
            entity.x = 0.0
            entity.y = 0.0
            entity.last_dy = 0
            entity.last_dx = 0
            entity.move_budget = 0.0
            # Reset species-specific state
            if hasattr(entity, "foraging_state"):
                entity.foraging_state = "migrating"
            if hasattr(entity, "eating_state"):
                entity.eating_state = "moving"
            if hasattr(entity, "behavior_state"):
                entity.behavior_state = "wandering"
            if hasattr(entity, "investigate_timer"):
                entity.investigate_timer = 0
                entity.investigate_y = None
                entity.investigate_x = None
            if hasattr(entity, "consumed_by"):
                entity.consumed_by.clear()
            self._available.append(entity)

    def __len__(self):
        return len(self._available)

    @property
    def total_created(self):
        return self._total_created


# Global pools
_GRAZER_POOL = None
_RABBIT_POOL = None
_PREDATOR_POOL = None
_CARCASS_POOL = None


def _init_pools():
    """Initialize global entity pools."""
    global _GRAZER_POOL, _RABBIT_POOL, _PREDATOR_POOL, _CARCASS_POOL

    def _make_grazer(existing, x, y, energy, config, age, gradient_weight, inherited_gradient_weight):
        if existing is None:
            grazer = Grazer(
                x=float(x),
                y=float(y),
                energy=energy,
                config=config,
                age=age,
                gradient_weight=gradient_weight,
                inherited_gradient_weight=inherited_gradient_weight,
            )
            # Add initial variation to heritable traits
            grazer.speed_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.metabolism_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.vision_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.grazing_efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.moving_efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.forage_vision_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            grazer.gradient_weight = gradient_weight
            grazer.inherited_gradient_weight = inherited_gradient_weight
            return grazer
        else:
            existing.x = float(x)
            existing.y = float(y)
            existing.energy = energy
            existing.config = config
            existing.age = age
            existing.gradient_weight = gradient_weight
            existing.inherited_gradient_weight = inherited_gradient_weight
            existing.alive = True
            return existing

    def _make_rabbit(existing, x, y, energy, config, age, gradient_weight, inherited_gradient_weight):
        if existing is None:
            rabbit = Rabbit(
                x=float(x),
                y=float(y),
                energy=energy,
                config=config,
                age=age,
                gradient_weight=gradient_weight,
                inherited_gradient_weight=inherited_gradient_weight,
            )
            # Add initial variation to heritable traits
            rabbit.speed_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.metabolism_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.vision_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.grazing_efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.moving_efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.forage_vision_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            rabbit.gradient_weight = gradient_weight
            rabbit.inherited_gradient_weight = inherited_gradient_weight
            return rabbit
        else:
            existing.x = float(x)
            existing.y = float(y)
            existing.energy = energy
            existing.config = config
            existing.age = age
            existing.gradient_weight = gradient_weight
            existing.inherited_gradient_weight = inherited_gradient_weight
            existing.alive = True
            return existing

    def _make_predator(existing, x, y, energy, config, age):
        if existing is None:
            predator = Predator(x=float(x), y=float(y), energy=energy, config=config, age=age)
            # Add initial variation to heritable traits
            predator.speed_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            predator.metabolism_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            predator.chase_speed_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            predator.hunt_efficiency_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            predator.scent_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            predator.vision_factor = float(np.clip(np.random.normal(1.0, 0.05), 0.5, 1.5))
            return predator
        else:
            existing.x = float(x)
            existing.y = float(y)
            existing.energy = energy
            existing.config = config
            existing.age = age
            existing.alive = True
            return existing

    def _make_carcass(existing, x, y, energy, max_energy, config):
        if existing is None:
            return Carcass(x=float(x), y=float(y), energy=energy, max_energy=max_energy, config=config)
        else:
            existing.x = float(x)
            existing.y = float(y)
            existing.energy = energy
            existing.max_energy = max_energy
            existing.config = config
            existing.age = 0
            existing.alive = True
            existing.consumed_by.clear()
            return existing

    _GRAZER_POOL = EntityPool(_make_grazer)
    _RABBIT_POOL = EntityPool(_make_rabbit)
    _PREDATOR_POOL = EntityPool(_make_predator)
    _CARCASS_POOL = EntityPool(_make_carcass)


def get_grazer_pool() -> EntityPool:
    if _GRAZER_POOL is None:
        _init_pools()
    return _GRAZER_POOL


def get_rabbit_pool() -> EntityPool:
    if _RABBIT_POOL is None:
        _init_pools()
    return _RABBIT_POOL


def get_predator_pool() -> EntityPool:
    if _PREDATOR_POOL is None:
        _init_pools()
    return _PREDATOR_POOL


def get_carcass_pool() -> EntityPool:
    if _CARCASS_POOL is None:
        _init_pools()
    return _CARCASS_POOL


class HasResourceField(Protocol):
    def get_resource_ratio(self, y: int, x: int) -> float: ...
    def consume_resource(self, y: int, x: int, amount: float) -> float: ...


@dataclass
class Entity:
    x: float
    y: float
    energy: float
    config: object
    age: int = 0
    alive: bool = True
    id: int = field(default_factory=lambda: next(_IDS))
    last_dy: int = 0
    last_dx: int = 0
    move_budget: float = 0.0
    # Individual variation traits (heritable with mutation)
    speed_factor: float = 1.0
    metabolism_factor: float = 1.0
    efficiency_factor: float = 1.0
    vision_factor: float = 1.0

    @property
    def max_energy(self) -> float:
        return float(self.config.max_energy)

    @property
    def energy_ratio(self) -> float:
        ceiling = self.max_energy
        return self.energy / ceiling if ceiling > 0 else 0.0

    @property
    def move_speed(self) -> int:
        return int(self.config.move_speed * self.speed_factor)

    def step_energy(self) -> None:
        self.energy -= float(self.config.energy_per_step) * self.metabolism_factor
        self.age += 1
        if self.energy <= 0:
            self.energy = 0.0
            self.alive = False

    def add_energy(self, amount: float) -> float:
        """Credit energy, respecting the species cap. Returns the amount actually taken."""
        before = self.energy
        self.energy = min(self.max_energy, self.energy + amount)
        return self.energy - before

    def record_move(self, dy: int, dx: int) -> None:
        self.last_dy = dy
        self.last_dx = dx

    def movement_allowance(self, mobility: float) -> int:
        """Cells to advance this step, carrying the fractional remainder.

        Rounding each step to a whole number of cells with a floor of 1 makes
        terrain mobility meaningless for slow-moving species: a grazer with
        move_speed 1 on rock (mobility 0.5) computed max(1, int(0.5)) == 1, so it
        moved exactly as fast on rock as on plains. Accumulating the budget
        instead makes rock genuinely cost time -- one cell every two steps --
        which is what lets ridges channel movement while staying passable.
        """
        self.move_budget += self.move_speed * mobility * self.speed_factor
        steps = int(self.move_budget)
        self.move_budget -= steps
        return steps


@dataclass
class Prey(Entity):
    """Shared prey mechanics. Grazers and rabbits differ only by config values."""

    # Terrain affinity key: selects the per-species productivity
    # override on the terrain being eaten.
    species_key = "prey"
    gradient_weight: float = 0.7
    inherited_gradient_weight: float = 0.7
    vision_radius: int = 5
    foraging_state: str = "migrating"
    # Heritable traits with mutation
    grazing_efficiency_factor: float = 1.0
    moving_efficiency_factor: float = 1.0
    forage_vision_factor: float = 1.0

    @property
    def vision(self) -> int:
        return int(self.config.vision_radius * self.forage_vision_factor)

    def mutate_gradient_weight(self, rng) -> None:
        std = float(self.config.gradient_weight_std)
        self.gradient_weight = rng.normal(self.inherited_gradient_weight, std)
        self.gradient_weight = float(min(1.0, max(0.0, self.gradient_weight)))

    def mutate_traits(self, rng) -> None:
        """Mutate heritable traits with small random variations."""
        std = 0.05  # 5% standard deviation
        self.speed_factor = float(np.clip(rng.normal(self.speed_factor, std), 0.5, 1.5))
        self.metabolism_factor = float(np.clip(rng.normal(self.metabolism_factor, std), 0.5, 1.5))
        self.efficiency_factor = float(np.clip(rng.normal(self.efficiency_factor, std), 0.5, 1.5))
        self.vision_factor = float(np.clip(rng.normal(self.vision_factor, std), 0.5, 1.5))
        self.grazing_efficiency_factor = float(np.clip(rng.normal(self.grazing_efficiency_factor, std), 0.5, 1.5))
        self.moving_efficiency_factor = float(np.clip(rng.normal(self.moving_efficiency_factor, std), 0.5, 1.5))
        self.forage_vision_factor = float(np.clip(rng.normal(self.forage_vision_factor, std), 0.5, 1.5))
        self.gradient_weight = rng.normal(self.inherited_gradient_weight, float(self.config.gradient_weight_std))
        self.gradient_weight = float(min(1.0, max(0.0, self.gradient_weight)))
        self.inherited_gradient_weight = self.gradient_weight

    def eat(self, env: HasResourceField, efficiency_scale: float = 1.0) -> float:
        """Resource consumption with exponential extraction decay.

        Extraction rate decays exponentially from 100% at full capacity to
        40% near zero, so depleted cells yield progressively less.
        """
        y, x = int(self.y), int(self.x)
        ratio = env.get_resource_ratio(y, x)
        if ratio <= 0:
            return 0.0
        # Exponential decay: 100% at ratio=1, 40% at ratio=0
        extraction = 0.4 + 0.6 * math.exp(-5.0 * (1.0 - ratio))
        efficiency = max(float(self.config.min_efficiency), extraction) * efficiency_scale * self.efficiency_factor
        taken = env.consume_resource(y, x, float(self.config.energy_from_resource) * efficiency)
        gained = self.add_energy(taken)

        productivity = getattr(env, "productivity", None)
        if productivity is not None:
            share = productivity(y, x, self.species_key)
            if share != 1.0:
                surplus = gained - float(self.config.energy_per_step)
                if surplus > 0.0:
                    # Terrain productivity scales the banked surplus:
                    # preferred ground pays a bonus, poor ground taxes
                    # it. share < 1 is the original tax; share > 1
                    # rewards the species that thrives here.
                    adjustment = surplus * (share - 1.0)
                    before = self.energy
                    self.energy = min(self.max_energy, max(0.0, self.energy + adjustment))
                    gained += self.energy - before
        return gained


@dataclass
class Grazer(Prey):
    species_key = "grazer"


@dataclass
class Rabbit(Prey):
    species_key = "rabbit"


@dataclass
class Predator(Entity):
    chase_radius: int = 7
    resource_sense_radius: int = 10
    eating_state: str = "moving"
    behavior_state: str = "wandering"
    investigate_y: Optional[int] = None
    investigate_x: Optional[int] = None
    investigate_timer: int = 0
    eating_timer: int = 0
    # Heritable traits with mutation
    chase_speed_factor: float = 1.0
    hunt_efficiency_factor: float = 1.0
    scent_factor: float = 1.0
    vision_factor: float = 1.0

    @property
    def vision(self) -> int:
        return int(self.config.chase_radius * self.vision_factor)

    @property
    def investigating(self) -> bool:
        return self.investigate_timer > 0

    def begin_investigation(self, y: int, x: int, steps: int) -> None:
        self.investigate_y = int(y)
        self.investigate_x = int(x)
        self.investigate_timer = int(steps)

    def clear_investigation(self) -> None:
        self.investigate_y = None
        self.investigate_x = None
        self.investigate_timer = 0

    def mutate_traits(self, rng) -> None:
        """Mutate heritable traits with small random variations."""
        std = 0.05  # 5% standard deviation
        self.speed_factor = float(np.clip(rng.normal(self.speed_factor, 0.05), 0.5, 1.5))
        self.metabolism_factor = float(np.clip(rng.normal(self.metabolism_factor, 0.05), 0.5, 1.5))
        self.chase_speed_factor = float(np.clip(rng.normal(self.chase_speed_factor, 0.05), 0.5, 1.5))
        self.hunt_efficiency_factor = float(np.clip(rng.normal(self.hunt_efficiency_factor, 0.05), 0.5, 1.5))
        self.scent_factor = float(np.clip(rng.normal(self.scent_factor, 0.05), 0.5, 1.5))

    def begin_investigation(self, y: int, x: int, steps: int) -> None:
        self.investigate_y = int(y)
        self.investigate_x = int(x)
        self.investigate_timer = int(steps)

    def clear_investigation(self) -> None:
        self.investigate_y = None
        self.investigate_x = None
        self.investigate_timer = 0


@dataclass
class Carcass:
    x: float
    y: float
    energy: float
    max_energy: float
    config: object
    age: int = 0
    id: int = field(default_factory=lambda: next(_IDS))
    consumed_by: list = field(default_factory=list)

    def step(self) -> None:
        self.age += 1
        self.energy *= float(self.config.decay_rate)
        if self.energy < float(self.config.min_energy):
            self.energy = 0.0

    @property
    def provides_energy(self) -> bool:
        return self.energy > 0

    def take_energy(self, amount: float) -> float:
        taken = min(amount, self.energy)
        self.energy -= taken
        return taken

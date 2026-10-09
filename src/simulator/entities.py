from dataclasses import dataclass, field
from itertools import count
from typing import Optional, Protocol
import math

_IDS = count(1)


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

    @property
    def max_energy(self) -> float:
        return float(self.config.max_energy)

    @property
    def energy_ratio(self) -> float:
        ceiling = self.max_energy
        return self.energy / ceiling if ceiling > 0 else 0.0

    @property
    def move_speed(self) -> int:
        return int(self.config.move_speed)

    def step_energy(self) -> None:
        self.energy -= float(self.config.energy_per_step)
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
        self.move_budget += self.move_speed * mobility
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

    @property
    def vision(self) -> int:
        return int(self.config.vision_radius)

    def mutate_gradient_weight(self, rng) -> None:
        std = float(self.config.gradient_weight_std)
        self.gradient_weight = rng.normal(self.inherited_gradient_weight, std)
        self.gradient_weight = float(min(1.0, max(0.0, self.gradient_weight)))

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
        efficiency = max(float(self.config.min_efficiency), extraction) * efficiency_scale
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
                    self.energy = min(
                        self.max_energy, max(0.0, self.energy + adjustment)
                    )
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

    @property
    def vision(self) -> int:
        return int(self.config.chase_radius)

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
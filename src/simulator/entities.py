import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from simulator.config import GrazerConfig, PredatorConfig, RabbitConfig


@dataclass
class Entity:
    x: float
    y: float
    energy: float
    max_energy: float
    age: int = 0
    id: int = field(default_factory=lambda: Entity._next_id())
    alive: bool = True
    size: float = 1.0

    _id_counter: int = field(default=0, init=False, repr=False)

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def step_energy(self, cost: float) -> None:
        self.energy -= cost
        self.age += 1
        if self.energy <= 0:
            self.alive = False

    def add_energy(self, amount: float) -> float:
        old = self.energy
        self.energy = min(self.max_energy, self.energy + amount)
        return self.energy - old


@dataclass
class Grazer(Entity):
    """Deer - can eat while moving at 60% efficiency, gains 110% when stopped, stores fat"""
    gradient_weight: float = 0.7
    vision_radius: int = 5
    move_speed: int = 1
    fat: float = 0.0  # Stored fat reserves
    max_fat: float = 500.0  # Maximum fat storage
    eating_state: str = "moving"  # "moving", "grazing"
    grazing_timer: int = 0
    config: Optional['GrazerConfig'] = None

    def __post_init__(self):
        if self.config:
            self.max_energy = self.config.max_energy
            self.gradient_weight = self.config.gradient_weight
            self.vision_radius = self.config.vision_radius
            self.move_speed = self.config.move_speed
            self.max_fat = self.config.max_fat

    def eat_while_moving(self, amount: float) -> float:
        """Eat while moving - 60% efficiency"""
        efficiency = 0.6
        eaten = min(amount * 0.6, self.max_energy - self.energy)
        self.energy += eaten
        return eaten

    def eat_while_stopped(self, amount: float) -> float:
        """Eat while stopped - 110% efficiency, stores excess as fat"""
        efficiency = 1.1
        gained = amount * 1.1
        # First fill energy to max
        energy_needed = self.max_energy - self.energy
        if gained <= energy_needed:
            self.energy += gained
        else:
            self.energy = self.max_energy
            # Excess goes to fat
            excess = gained - energy_needed
            self.fat = min(self.max_fat, self.fat + excess)
        return min(gained, self.max_energy - self.energy + self.max_fat - self.fat)

    def use_fat(self, amount: float) -> float:
        """Use stored fat for energy when needed"""
        used = min(amount, self.fat)
        self.fat -= used
        self.energy = min(self.max_energy, self.energy + used)
        return used


@dataclass
class Rabbit(Entity):
    """Rabbit - fast breeding, low energy, high herding, stops to eat"""
    gradient_weight: float = 0.9  # Very high herding
    vision_radius: int = 4  # Slightly smaller vision
    move_speed: int = 1
    eating_state: str = "moving"  # "moving", "eating", "grazing"
    grazing_timer: int = 0  # Timer for grazing duration
    config: Optional['RabbitConfig'] = None

    def __post_init__(self):
        if self.config:
            self.max_energy = self.config.max_energy
            self.gradient_weight = self.config.gradient_weight
            self.vision_radius = self.config.vision_radius
            self.move_speed = self.config.move_speed

    def eat_while_moving(self, amount: float) -> float:
        """Eat while moving - only 20% efficiency"""
        efficiency = 0.2
        eaten = min(amount * 0.2, self.max_energy - self.energy)
        self.energy += eaten
        return eaten

    def eat_while_stopped(self, amount: float) -> float:
        """Eat while stopped - gains lots of energy"""
        efficiency = 3.0  # Gains 3x when stopped
        gained = amount * 3.0
        gained = min(gained, self.max_energy - self.energy)
        self.energy += gained
        return gained


@dataclass
class Predator(Entity):
    chase_radius: int = 5
    resource_sense_radius: int = 8
    move_speed: int = 2
    eating_state: str = "moving"  # "moving", "eating_carcass", "eating_prey"
    eating_timer: int = 0
    config: Optional['PredatorConfig'] = None

    def __post_init__(self):
        if self.config:
            self.max_energy = self.config.max_energy
            self.chase_radius = self.config.chase_radius
            self.resource_sense_radius = self.config.resource_sense_radius
            self.move_speed = self.config.move_speed

    def can_eat_carcass(self) -> bool:
        """Can eat carcass in adjacent cell without moving"""
        return True

    def can_eat_prey(self) -> bool:
        """Can only eat prey when in same cell and stopped"""
        return True


@dataclass
class Carcass:
    x: float
    y: float
    energy: float
    max_energy: float
    age: int = 0
    id: int = field(default_factory=lambda: Carcass._next_id())
    consumed_by: list = field(default_factory=list)

    _id_counter: int = field(default=0, init=False, repr=False)

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def step(self) -> None:
        self.age += 1
        self.energy *= 0.98
        if self.energy < 1:
            self.energy = 0

    def can_provide_energy(self) -> bool:
        return self.energy > 0

    def take_energy(self, amount: float) -> float:
        taken = min(amount, self.energy)
        self.energy -= taken
        return taken
"""Reproduction system for entity reproduction and population dynamics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.config import PredatorConfig, PreyConfig
    from simulator.entities import Grazer, Predator, Rabbit
    from simulator.simulation import Simulation
else:
    from simulator.entities import Grazer, Predator, Rabbit


class ReproductionSystem:
    """Handles all reproduction logic for grazers, rabbits, and predators."""

    def __init__(self, sim) -> None:
        self.sim = sim

    def reproduce(self) -> None:
        """Process reproduction for all species."""
        self._reproduce_species(
            self.sim.grazers,
            self.sim.config.grazer,
            "grazer",
            lambda p, c: Grazer(x=float(0), y=float(0), energy=0, config=self.sim.config.grazer),
        )
        self._reproduce_species(
            self.sim.rabbits,
            self.sim.config.rabbit,
            "rabbit",
            lambda p, c: Rabbit(x=float(0), y=float(0), energy=0, config=self.sim.config.rabbit),
        )
        self._reproduce_species(
            self.sim.predators,
            self.sim.config.predator,
            "predator",
            lambda p, c: Predator(x=float(0), y=float(0), energy=0, config=self.sim.config.predator),
        )

    def _reproduce_species(self, flock, cfg, kind, child_factory):
        """Generic reproduction logic for a species."""
        offspring_list = []

        for parent in flock:
            if not parent.alive:
                continue
            if parent.age < cfg.reproduction_min_age or parent.energy < cfg.reproduction_min_energy:
                continue

            cost = parent.energy * cfg.reproduction_energy_cost
            parent.energy -= cost

            py, px = int(parent.y), int(parent.x)
            oy = py + int(self.sim.rng.integers(-1, 2))
            ox = px + int(self.sim.rng.integers(-1, 2))
            oy, ox = self.sim.env.clamp(oy, ox)

            energy = float(cfg.offspring_initial_energy)
            child = Grazer(x=float(0), y=float(0), energy=energy, config=cfg)  # placeholder

            if kind == "grazer":
                child = Grazer(x=float(0), y=float(0), energy=energy, config=self.sim.config.grazer)
                child.inherited_gradient_weight = parent.inherited_gradient_weight
                child.mutate_gradient_weight(self.sim.rng)
                self.sim.counters.births_grazer += 1
            elif kind == "rabbit":
                child = Rabbit(x=float(0), y=float(0), energy=energy, config=self.sim.config.rabbit)
                child.inherited_gradient_weight = parent.inherited_gradient_weight
                child.mutate_gradient_weight(self.sim.rng)
                self.sim.counters.births_rabbit += 1
            else:  # predator
                child = Predator(x=float(0), y=float(0), energy=energy, config=self.sim.config.predator)
                self.sim.counters.births_predator += 1

            # Set actual position
            py, px = int(parent.y), int(parent.x)
            oy = py + int(self.sim.rng.integers(-1, 2))
            ox = px + int(self.sim.rng.integers(-1, 2))
            oy, ox = self.sim.env.clamp(oy, ox)

            child.x = float(ox)
            child.y = float(oy)
            child.energy = energy

            if isinstance(parent, Grazer):
                child.inherited_gradient_weight = parent.inherited_gradient_weight
                child.mutate_gradient_weight(self.sim.rng)
                self.sim.counters.births_grazer += 1
            elif isinstance(parent, Rabbit):
                child.inherited_gradient_weight = parent.inherited_gradient_weight
                child.mutate_gradient_weight(self.sim.rng)
                self.sim.counters.births_rabbit += 1
            else:
                self.sim.counters.births_predator += 1

            offspring_list.append(child)

        if kind == "grazer":
            self.sim.grazers.extend(offspring_list)
        elif kind == "rabbit":
            self.sim.rabbits.extend(offspring_list)
        else:
            self.sim.predators.extend(offspring_list)

    def _try_reproduce(self, parent, cfg):
        """SPECIFICATION.md: cost is a fraction of the parent's *current* energy."""
        if parent.age < cfg.reproduction_min_age or parent.energy < cfg.reproduction_min_energy:
            return None
        cost = parent.energy * cfg.reproduction_energy_cost
        parent.energy -= cost

        py, px = int(parent.y), int(parent.x)
        oy = py + int(self.sim.rng.integers(-1, 2))
        ox = px + int(self.sim.rng.integers(-1, 2))
        oy, ox = self.sim.env.clamp(oy, ox)

        energy = float(cfg.offspring_initial_energy)
        if isinstance(parent, Grazer):
            child = Grazer(x=float(ox), y=float(oy), energy=energy, config=cfg)
            child.inherited_gradient_weight = parent.inherited_gradient_weight
            child.mutate_gradient_weight(self.sim.rng)
            self.sim.counters.births_grazer += 1
        elif isinstance(parent, Rabbit):
            child = Rabbit(x=float(ox), y=float(oy), energy=energy, config=cfg)
            child.inherited_gradient_weight = parent.inherited_gradient_weight
            child.mutate_gradient_weight(self.sim.rng)
            self.sim.counters.births_rabbit += 1
        else:
            child = Predator(x=float(ox), y=float(oy), energy=energy, config=cfg)
            self.sim.counters.births_predator += 1
        return child

"""Movement system for entity movement and pathfinding."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.config import PredatorConfig, PreyConfig
    from simulator.entities import Predator, Prey
    from simulator.environment import Environment
    from simulator.simulation import Simulation


class MovementSystem:
    """Handles all entity movement logic for prey and predators."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim

    # --- Prey movement ---

    def _threat_vector(self, prey, radius: int):
        """Weighted escape direction away from predators inside flee_radius."""
        if radius <= 0:
            return None
        py, px = int(prey.y), int(prey.x)
        away_y = away_x = 0.0
        found = False
        for threat in self.sim._predator_index.query(py - radius, py + radius, px - radius, px + radius):
            if not threat.alive:
                continue
            ty, tx = int(threat.y), int(threat.x)
            gap = max(abs(ty - py), abs(tx - px))
            if gap > radius or gap == 0:
                continue
            weight = 1.0 / gap
            away_y += _sign(py - ty) * weight
            away_x += _sign(px - tx) * weight
            found = True
        return (away_y, away_x) if found else None

    def move_prey(self, prey, cfg, herd, target):
        """Advance a prey. Returns True if the movement cost starved it."""
        py, px = int(prey.y), int(prey.x)
        threat = self._threat_vector(prey, cfg.flee_radius)
        fleeing = threat is not None

        speed_multiplier = cfg.flee_speed_multiplier if fleeing else 1.0
        if not cfg.migratory:
            speed_multiplier *= cfg.grazing_move_multiplier
        cost_multiplier = cfg.flee_cost_multiplier if fleeing else 1.0

        herd_weighted = None
        if herd is not None and self.sim.rng.random() < cfg.herd_follow_chance:
            herd_weighted = [
                herd[0] * cfg.cohesion_weight + herd[2] * cfg.alignment_weight,
                herd[1] * cfg.cohesion_weight + herd[3] * cfg.alignment_weight,
                herd[4] * cfg.separation_weight,
                herd[5] * cfg.separation_weight,
            ]
            if cfg.migratory and prey.foraging_state == "migrating":
                boost = cfg.migrating_herd_weight
                herd_weighted[0] *= boost
                herd_weighted[1] *= boost

        gradient_bias = float(np.clip(prey.gradient_weight, 0.0, 1.0))

        mobility = self.sim.env.mobility_prey[py, px]
        allowance = prey.movement_allowance(mobility * speed_multiplier)
        moved = 0

        for _ in range(allowance):
            ty, tx = target
            current_to_target = self.sim.distance(py, px, ty, tx)
            best = None
            best_score = -np.inf

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    ny, nx = self.sim.env.clamp(ny, nx)
                    if not self.sim.env.passable[ny, nx]:
                        continue

                    approach = current_to_target - self.sim.distance(ny, nx, ty, tx)
                    score = approach * gradient_bias

                    if herd_weighted is not None:
                        score += (dy * herd_weighted[0] + dx * herd_weighted[1]) * (1.0 - gradient_bias) * 0.5
                        score += dy * herd_weighted[2] + dx * herd_weighted[3]

                    if threat is not None:
                        score += (dy * threat[0] + dx * threat[1]) * cfg.flee_weight

                    if prey.energy_ratio > 0.95:
                        score += (1.0 - self.sim.env.mobility_prey[ny, nx]) * 1.5

                    if score > best_score:
                        best_score = score
                        best = (ny, nx, dy, dx)

            if best is None:
                break

            ny, nx, dy, dx = best
            moved_y = int(np.sign(ny - py))
            moved_x = int(np.sign(nx - px))
            prey.x = float(nx)
            prey.y = float(ny)
            prey.record_move(moved_y, moved_x)
            py, px = ny, nx
            moved += 1

            self.sim.env._add_scent(py, px, 0.15)

        prey.energy -= moved * cfg.movement_energy_cost * cost_multiplier
        if prey.energy <= 0:
            prey.energy = 0.0
            prey.alive = False
            return True
        return False

    # --- Predator movement ---

    def move_predator_toward(self, predator, ty: int, tx: int, chasing: bool = False) -> None:
        """Advance toward a goal. Chasing costs double movement metabolism."""
        cfg = self.sim.config.predator
        py, px = int(predator.y), int(predator.x)
        mobility = self.sim.env.mobility_predator[py, px]

        speed = self.sim.config.predator.chase_speed_multiplier if chasing else 1.0
        cost = self.sim.config.predator.chase_cost_multiplier if chasing else 1.0
        allowance = predator.movement_allowance(mobility * speed)
        moved = 0

        for _ in range(allowance):
            gap = self.sim.distance(py, px, ty, tx)
            if gap == 0:
                break
            best = None
            best_gap = gap
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = self.sim.env.clamp(py + dy, px + dx)
                    if not self.sim.env.passable[ny, nx]:
                        continue
                    candidate_gap = self.sim.distance(ny, nx, ty, tx)
                    if candidate_gap < best_gap:
                        best_gap = candidate_gap
                        best = (ny, nx)
                if best is None:
                    break
                moved_y = int(np.sign(best[0] - py))
                moved_x = int(np.sign(best[1] - px))
                predator.y = float(best[0])
                predator.x = float(best[1])
                predator.record_move(moved_y, moved_x)
                py, px = best
                moved += 1

        predator.energy -= moved * self.sim.config.predator.movement_energy_cost * cost
        if predator.energy <= 0:
            predator.energy = 0.0
            predator.alive = False

    def predator_wander(self, predator) -> None:
        py, px = int(predator.y), int(predator.x)
        mobility = self.sim.env.mobility_predator[py, px]
        steps = predator.movement_allowance(mobility)

        for _ in range(steps):
            best = None
            best_score = -float("inf")
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = self.sim.env.clamp(py + dy, px + dx)
                    if not self.sim.env.passable[ny, nx]:
                        continue
                    centre = max(abs(ny - self.sim.map_height // 2), abs(nx - self.sim.map_width // 2))
                    score = self.sim._ratios[ny, nx] - centre * 0.01
                    if score > best_score:
                        best_score = score
                        best = (ny, nx)
            if best is None:
                break
            moved_y = int(np.sign(best[0] - py))
            moved_x = int(np.sign(best[1] - px))
            predator.y = float(best[0])
            predator.x = float(best[1])
            predator.record_move(moved_y, moved_x)
            py, px = best


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

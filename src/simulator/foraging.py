"""Foraging system for prey movement and foraging behavior."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.config import PreyConfig
    from simulator.entities import Prey
    from simulator.environment import Environment
    from simulator.simulation import Simulation


class ForagingSystem:
    """Handles all prey foraging behavior and movement decisions."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        # Pre-allocated arrays for hot paths to avoid repeated allocation
        self._forage_score = np.empty((0, 0), dtype=np.float64)
        self._forage_reachable = np.empty((0, 0), dtype=bool)

    # --- Foraging state ---

    def foraging_state(self, prey, cfg) -> str:
        """Decide whether this prey feeds where it stands or travels."""
        if not cfg.migratory:
            return "grazing"

        ratio = self.sim.env.get_resource_ratio(int(prey.y), int(prey.x))
        energy_ratio = prey.energy_ratio

        if prey.foraging_state == "grazing":
            # Stay until full, or until there is nothing left here worth eating.
            if (
                energy_ratio >= cfg.graze_energy_target
                or ratio <= cfg.migrate_ratio_threshold
                or energy_ratio >= cfg.wander_energy_threshold
            ):
                return "migrating"
            # If stationary intake has dropped below what moving would give
            # on a full cell, leave before the cell is stripped.
            stationary = (
                cfg.energy_from_resource * cfg.grazing_efficiency * (0.4 + 0.6 * math.exp(-5.0 * (1.0 - ratio)))
            )
            moving_baseline = cfg.energy_from_resource * cfg.moving_efficiency
            if stationary < moving_baseline and energy_ratio > cfg.graze_resume_energy:
                return "migrating"
            return "grazing"

        # Starvation escape: if critically low, settle on ANY passable cell
        # rather than burning energy travelling on bare ground.
        if energy_ratio <= cfg.graze_resume_energy * 0.5:
            return "grazing"

        # Travelling: only settle once genuinely hungry, and only on good ground.
        if energy_ratio <= cfg.graze_resume_energy and ratio >= cfg.graze_ratio_threshold:
            return "grazing"

        # High resource opportunity: if local resources are abundant,
        # stop and graze even if not critically hungry
        if ratio >= cfg.graze_ratio_threshold * 1.5 and energy_ratio > cfg.graze_resume_energy:
            return "grazing"

        return "migrating"

    def _species_affinity(self, species: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Per-cell foraging bonus from terrain the species thrives in."""
        terrains = self.sim.config.environment.terrain_types
        lut = np.zeros(len(self.sim.env.terrain_names))
        for i, name in enumerate(self.sim.env.terrain_names):
            terrain_cfg = terrains.get(name)
            if terrain_cfg is None:
                continue
            value = getattr(terrain_cfg, f"{species}_productivity", None)
            if value is not None:
                lut[i] = (float(value) - 1.0) * BIOME_AFFINITY_WEIGHT
        return lut[self.sim.env.terrain[y0:y1, x0:x1]]

    def forage_target(self, prey, cfg) -> tuple[int, int]:
        """Best cell inside the prey's vision disc, scored on resource availability."""
        py, px = int(prey.y), int(prey.x)
        radius = max(1, int(prey.vision))
        y0, y1 = max(0, py - radius), min(self.sim.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.sim.map_width, px + radius + 1)

        h, w = y1 - y0, x1 - x0

        # Resize pre-allocated arrays if needed
        if self._forage_score.shape != (h, w):
            self._forage_score = np.empty((h, w), dtype=np.float64)
            self._forage_reachable = np.empty((h, w), dtype=bool)

        window = self.sim._ratios[y0:y1, x0:x1]
        reachable = self.sim.env.passable[y0:y1, x0:x1]

        # Use pre-allocated arrays
        self._forage_reachable[:] = reachable
        self._forage_score[:] = np.where(reachable, window, -np.inf)
        score = self._forage_score

        # Species affinity: ground where this prey extracts more energy
        # scores higher, so grazers drift toward meadows and rabbits
        # toward scrub instead of both species competing on the same
        # cells. The bias is small enough that resource availability
        # still dominates.
        affinity = self._species_affinity(prey.species_key, y0, y1, x0, x1)
        score += np.where(reachable, affinity, 0.0)

        # Well-fed prey can afford to prospect away from rich ground, but a
        # migratory species is looking for food, so prospecting is counterproductive.
        if prey.energy_ratio > 0.8 and not cfg.migratory:
            score += np.where(reachable, (1.0 - window) * 0.3, 0.0)

        # Distance penalty - compute inline to avoid allocation
        rows = np.arange(h)[:, None] + y0
        cols = np.arange(w)[None, :] + x0
        score -= 0.02 * np.maximum(np.abs(rows - py), np.abs(cols - px))

        best = int(np.argmax(score))
        if not np.isfinite(score.flat[best]):
            return py, px
        return y0 + best // w, x0 + best % w

    def threat_vector(self, prey, radius: int):
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

    def move_prey(self, prey, cfg, herd, target) -> bool:
        """Advance a prey. Returns True if the movement cost starved it."""
        py, px = int(prey.y), int(prey.x)
        threat = self.threat_vector(prey, cfg.flee_radius)
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
                # Travelling as a pack: cohesion and alignment outweigh the
                # individual gradient, so the herd moves as one body.
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

            # Deposit scent as prey moves
            self.sim.env._add_scent(py, px, 0.15)

        prey.energy -= moved * cfg.movement_energy_cost * cost_multiplier
        if prey.energy <= 0:
            prey.energy = 0.0
            prey.alive = False
            return True
        return False

    def update_prey(self, flock, cfg, kind: str) -> None:
        """Update all prey in a flock (grazers or rabbits)."""
        for prey in flock:
            if not prey.alive:
                continue

            was_alive = prey.alive
            prey.step_energy()
            if was_alive and not prey.alive:
                self.sim._record_death(kind)

            if not prey.alive:
                continue

            state = self.foraging_state(prey, cfg)

            if cfg.migratory:
                prey.foraging_state = state
                if state == "grazing":
                    # Standing still: feed hard, do not move.
                    gained = prey.eat(self.sim.env, cfg.grazing_efficiency)
                    if gained > 0:
                        self.sim._mark_grazed(prey)
                    continue
                # Travelling: feed on the move, at a rate that cannot cover
                # metabolism, so the journey costs energy.
                prey.eat(self.sim.env, cfg.moving_efficiency)
                herd = self.herd_vectors(prey, self.sim._prey_index, cfg)
                target = self.forage_target(prey, cfg)
                if self.move_prey(prey, cfg, herd, target):
                    self.sim._record_death(kind)
                continue

            herd = self.herd_vectors(prey, self.sim._prey_index, cfg)
            target = self.forage_target(prey, cfg)
            if self.move_prey(prey, cfg, herd, target):
                self.sim._record_death(kind)
            if prey.eat(self.sim.env) > 0:
                self.sim._mark_grazed(prey)

    def _mark_grazed(self, prey) -> None:
        """Record that this cell is being fed on, so a camped herd strips it."""
        y, x = int(prey.y), int(prey.x)
        if 0 <= y < self.sim.map_height and 0 <= x < self.sim.map_width:
            self.sim._grazed[y, x] += 1

    def herd_vectors(self, prey, index: _SpatialIndex, cfg) -> tuple | None:
        """Cohesion, alignment and separation from neighbours inside herd_radius."""
        py, px = int(prey.y), int(prey.x)
        radius = cfg.herd_radius
        cohesion_y = cohesion_x = 0.0
        alignment_y = alignment_x = 0.0
        separation_y = separation_x = 0.0
        seen = 0

        for other in index.query(py - radius, py + radius, px - radius, px + radius):
            if other is prey or not other.alive:
                continue
            oy, ox = int(other.y), int(other.x)
            gap = max(abs(oy - py), abs(ox - px))
            if gap > radius:
                continue
            seen += 1
            dy = oy - py
            dx = ox - px
            cohesion_y += dy
            cohesion_x += dx
            alignment_y += other.last_dy
            alignment_x += other.last_dx
            if 0 < gap <= cfg.separation_radius:
                weight = cfg.separation_radius - gap + 1
                separation_y += dy * weight
                separation_x += dx * weight

        if seen == 0:
            return None
        return (
            cohesion_y / seen,
            cohesion_x / seen,
            alignment_y / seen,
            alignment_x / seen,
            separation_y,
            separation_x,
            seen,
        )


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


BIOME_AFFINITY_WEIGHT = 0.6

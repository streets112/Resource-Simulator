"""Predator behavior system for hunting, patrolling, and pack dynamics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.config import PredatorConfig
    from simulator.entities import Carcass, Predator
    from simulator.environment import Environment
    from simulator.simulation import Simulation
else:
    from simulator.entities import Carcass


class PredatorBehaviorSystem:
    """Handles all predator behavior: hunting, patrolling, pack dynamics."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim

    # --- Predator behavior ---

    def update_predators(self) -> None:
        """Update all predators: handle feeding, movement, and behavior."""
        # Update wind: slowly shift direction and intensity
        self._update_wind()

        # Decrease eating timer for predators that are eating
        for predator in self.sim.predators:
            if predator.eating_timer > 0:
                predator.eating_timer -= 1
                predator.eating_state = "feeding"
            else:
                predator.eating_state = "moving"

        # Carcass feeding is resolved as one coordinated pass before any predator
        # moves, so that every predator around the same carcass competes for one
        # shared energy store in a deterministic order.
        self._feed_on_carcasses()

        for predator in self.sim.predators:
            if not predator.alive:
                continue

            was_alive = predator.alive
            predator.step_energy()
            if was_alive and not predator.alive:
                self.sim.counters.deaths_predator += 1
            if not predator.alive:
                continue

            # Skip if currently eating
            if predator.eating_timer > 0:
                predator.eating_state = "feeding"
                continue

            # Feeding and killing are deliberately separate steps. A kill deposits
            # a carcass and grants the predator nothing directly; the energy is
            # only recoverable by scavenging that carcass on a later step. This
            # also removes the old double-count where a kill paid out both an
            # immediate energy transfer and a carcass holding part of the same
            # animal.
            #
            # Movement runs before the kill check so a predator that closes on
            # prey during its chase can kill it on the same step; otherwise it
            # sweeps over its target and the prey escapes.
            if predator.eating_state == "feeding":
                continue

            self._predator_behaviour(predator)

            if self._predator_kill_prey(predator):
                continue

    def _update_wind(self) -> None:
        """Drift the virtual wind direction and intensity."""
        self.sim._wind_dir += float(self.sim.rng.normal(0, self.sim._wind_change_rate))
        self.sim._wind_dir %= 2 * np.pi
        self.sim._wind_intensity = float(np.clip(self.sim._wind_intensity + self.sim.rng.normal(0, 0.08), 0.1, 1.0))

    def _feed_on_carcasses(self) -> None:
        """Scavenge pass over every carcass.

        All predators within feed_radius (their own cell plus the eight adjacent
        ones) draw on the same carcass, so a carcass surrounded by more predators
        is stripped proportionally faster. Feeding is ordered oldest-first, which
        decides who gets the remainder when the store runs out mid-pass.
        """
        cfg = self.sim.config.predator
        carcass_cfg = self.sim.config.carcass
        per_step = cfg.energy_from_prey / carcass_cfg.consumption_divisor

        for carcass in self.sim.carcasses:
            if not carcass.provides_energy:
                continue
            # Predators in the 3x3 neighbourhood of the carcass can feed.
            feeders = []
            cy, cx = int(carcass.y), int(carcass.x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < self.sim.map_height and 0 <= nx < self.sim.map_width):
                        continue
                    for predator in self.sim._predator_index.query(cy, cy, cx, cx):
                        if predator.alive and int(predator.y) == cy + dy and int(predator.x) == cx + dx:
                            feeders.append(predator)
            if not feeders:
                continue
            # Oldest first
            feeders.sort(key=lambda p: p.age)
            for predator in feeders:
                # Stop eating if predator is full
                if predator.energy >= predator.max_energy:
                    predator.eating_state = "moving"
                    continue
                draw = min(per_step, carcass.energy, predator.max_energy - predator.energy)
                if draw <= 0:
                    predator.eating_state = "moving"
                    continue
                carcass.energy -= draw
                predator.energy += draw
                predator.eating_state = "feeding"
                if not carcass.provides_energy:
                    predator.eating_state = "moving"
                    break

    def _predator_kill_prey(self, predator) -> bool:
        """Kill co-located prey and leave a carcass. No energy is transferred here.
        Only kills when hungry."""
        # Only kill when hungry
        cfg = self.sim.config.predator
        hungry = predator.energy < cfg.hunt_energy_threshold * predator.max_energy
        if not hungry:
            return False
            
        py, px = int(predator.y), int(predator.x)
        fraction = self.sim.config.carcass.carcass_energy_fraction
        
        for flock, kind, kill_counter in (
            (self.sim.grazers, "grazer", "kills_grazer"),
            (self.sim.rabbits, "rabbit", "kills_rabbit"),
        ):
            for prey in flock:
                if not prey.alive:
                    continue
                if int(prey.y) != py or int(prey.x) != px:
                    continue
                prey.alive = False
                self.sim._record_death(kind)
                setattr(self.sim.counters, kill_counter, getattr(self.sim.counters, kill_counter) + 1)
                self.sim.carcasses.append(
                    Carcass(
                        x=prey.x,
                        y=prey.y,
                        energy=prey.energy * fraction,
                        max_energy=prey.energy * fraction,
                        config=self.sim.config.carcass,
                    )
                )
                predator.eating_state = "feeding"
                # Eating time based on prey's energy: (1/3) * max_energy + (2/3) * current_energy
                eating_time = int((1.0/3.0) * prey.max_energy + (2.0/3.0) * prey.energy)
                predator.eating_timer = max(1, eating_time)
                predator.eating_state = "feeding"
                return True
        return False

    def _strongest_scent_cell(self, predator) -> tuple[int, int] | None:
        """Return the coordinates of the cell with the strongest scent within range."""
        py, px = int(predator.y), int(predator.x)
        radius = self.sim.config.predator.scent_radius
        y0, y1 = max(0, py - radius), min(self.sim.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.sim.map_width, px + radius + 1)

        # Find the cell with the strongest scent
        scent_slice = self.sim.env.scent[y0:y1, x0:x1]
        if scent_slice.size == 0:
            return None

        max_idx = np.argmax(scent_slice)
        if scent_slice.flat[scent_slice.argmax()] <= 0.01:  # Minimum threshold
            return None

        max_y, max_x = np.unravel_index(max_idx, scent_slice.shape)
        return y0 + max_y, x0 + max_x

    def distance(self, y1: int, x1: int, y2: int, x2: int) -> int:
        """SPECIFICATION.md bounded Chebyshev distance (toroidal)."""
        dy = abs(y1 - y2)
        dx = abs(x1 - x2)
        return int(max(min(dy, self.sim.map_height - dy), min(dx, self.sim.map_width - dx)))

    # --- Predator behavior methods ---

    def _predator_behaviour(self, predator) -> None:
        """Main predator decision-making logic."""
        cfg = self.sim.config.predator
        py, px = int(predator.y), int(predator.x)

        pack = self.sim._pack_mates(predator)
        pack_center = self.sim._centroid(pack) if pack else None

        # Split large packs first
        self.sim._maybe_split_pack(predator, self.sim._pack_mates(predator))

        # Check satiety: predators only hunt when energy is below threshold
        hungry = predator.energy < cfg.hunt_energy_threshold * predator.max_energy

        # Priority 1: Direct sight of prey -> chase
        solo_target = self._solo_target(predator)
        if solo_target is not None:
            predator.behavior_state = "chasing"
            self.sim.movement.move_predator_toward(predator, int(solo_target.y), int(solo_target.x), chasing=True)
            return

        if pack and self.sim.rng.random() < cfg.pack_hunt_chance:
            target = self._pack_target(predator, pack)
            if target is not None:
                predator.behavior_state = "pack_hunting"
                flank = self.sim.rng.random() < cfg.flank_chance
                goal = (
                    self._flank_goal(target, pack_center) if flank and pack_center else (int(target.y), int(target.x))
                )
                self.sim.movement.move_predator_toward(predator, goal[0], goal[1], chasing=True)
                return

        # Only hunt if hungry
        if hungry:
            # Priority 2: Follow scent gradient downwind
            scent_target = self._strongest_scent_cell(predator)
            if scent_target is not None:
                predator.behavior_state = "scent_tracking"
                ty, tx = scent_target
                ty, tx = self.sim.env.clamp(ty, tx)
                self.sim.movement.move_predator_toward(predator, ty, tx)
                return

            # Priority 3: Investigate depletion
            if self.sim._investigate_depletion(predator):
                return

        # Priority 3 (satiated) or 4 (hungry but no scent/resource): Patrol toward high prey density
        if cfg.patrol_toward_prey:
            prey_target = self.sim._prey_density_target(predator)
            if prey_target is not None:
                predator.behavior_state = "patrolling"
                self.sim.movement.move_predator_toward(predator, prey_target[0], prey_target[1])
                return

        # Priority 4: Wander in a random direction
        if self.sim._investigate_depletion(predator):
            predator.behavior_state = "investigating"
            return
        if (
            pack
            and self.sim._centroid(self.sim._pack_mates(predator)) is not None
            and self.sim.rng.random() < cfg.pack_wander_chance
        ):
            predator.behavior_state = "pack_wandering"
            self.sim.movement.move_predator_toward(
                predator,
                self.sim._centroid(self.sim._pack_mates(predator))[0],
                self.sim._centroid(self.sim._pack_mates(predator))[1],
            )
            return
        predator.behavior_state = "wandering"
        self.sim.movement.predator_wander(predator)

    def _solo_target(self, predator):
        """Find nearest prey within chase radius."""
        vision = self._prey_vision(predator)
        py, px = int(predator.y), int(predator.x)
        best = None
        best_distance = float("inf")
        for prey in self.sim._prey_index.query(py - vision, py + vision, px - vision, px + vision):
            if not prey.alive:
                continue
            gap = self.distance(py, px, int(prey.y), int(prey.x))
            if gap <= vision and gap < best_distance:
                best, best_distance = prey, gap
        return best

    def _pack_target(self, predator, pack):
        """SPECIFICATION.md: score = (chase_radius + 5 - distance) + nearby_predators * 3."""
        cfg = self.sim.config.predator
        py, px = int(predator.y), int(predator.x)
        vision = self._prey_vision(predator)
        pack_positions = [(int(m.y), int(m.x)) for m in pack]

        best = None
        best_score = -float("inf")
        for prey in self.sim._prey_index.query(py - vision, py + vision, px - vision, px + vision):
            if not prey.alive:
                continue
            ty, tx = int(prey.y), int(prey.x)
            gap = self.distance(py, px, ty, tx)
            if gap > vision:
                continue
            escorts = sum(1 for my, mx in pack_positions if self.distance(my, mx, ty, tx) <= cfg.pack_radius)
            score = (cfg.chase_radius + 5 - gap) + escorts * cfg.pack_score_bonus
            if score > best_score:
                best, best_score = prey, score
        return best

    def _flank_goal(self, prey, pack_center: tuple[int, int]) -> tuple[int, int]:
        """Approach from the side of the prey facing away from the pack centre."""
        ty, tx = int(prey.y), int(prey.x)
        cy, cx = pack_center
        return ty + _sign(ty - cy), tx + _sign(tx - cx)

    def _investigate_depletion(self, predator) -> bool:
        """Resource sensing: head for freshly stripped ground inside sense radius."""
        cfg = self.sim.config.predator
        py, px = int(predator.y), int(predator.x)

        if predator.investigating:
            predator.investigate_timer -= 1
            if predator.investigate_timer <= 0:
                predator.clear_investigation()
            else:
                self.sim.movement.move_predator_toward(predator, predator.investigate_y, predator.investigate_x)
                return True

        radius = max(1, int(cfg.resource_sense_radius))
        radius = int(radius * self.sim.env.visibility(py, px, predator=True))
        y0, y1 = max(0, py - radius), min(self.sim.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.sim.map_width, px + radius + 1)

        ratios = self.sim._ratios[y0:y1, x0:x1]
        depletion = self.sim.env.depletion_rate[y0:y1, x0:x1]
        reachable = self.sim.env.passable[y0:y1, x0:x1]

        evidence = np.where(reachable & (ratios <= cfg.depleted_ratio_threshold), depletion, 0.0)
        if float(evidence.max()) < cfg.investigate_threshold:
            return False

        best = int(np.argmax(evidence))
        predator.begin_investigation(
            y0 + best // evidence.shape[1],
            x0 + best % evidence.shape[1],
            cfg.investigate_steps,
        )
        self.sim.movement.move_predator_toward(predator, predator.investigate_y, predator.investigate_x)
        return True

    def _prey_density_target(self, predator) -> tuple[int, int] | None:
        """Find the cell with highest prey density within patrol range."""
        cfg = self.sim.config.predator
        py, px = int(predator.y), int(predator.x)
        radius = cfg.patrol_vision_range
        y0, y1 = max(0, py - radius), min(self.sim.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.sim.map_width, px + radius + 1)

        # Count prey in each cell
        prey_counts = np.zeros((y1 - y0, x1 - x0), dtype=int)
        for prey in self.grazers + self.rabbits:
            if prey.alive:
                ry, rx = int(prey.y), int(prey.x)
                if y0 <= ry < y1 and x0 <= rx < x1:
                    prey_counts[ry - y0, rx - x0] += 1

        if prey_counts.max() == 0:
            return None

        # Weight by distance (prefer closer high-density areas)
        y_idx, x_idx = np.unravel_index(prey_counts.argmax(), prey_counts.shape)
        return y0 + y_idx, x0 + x_idx

    def _prey_vision(self, predator) -> int:
        cfg = self.sim.config.predator
        base = cfg.chase_radius
        if predator.investigating:
            base = int(base * cfg.vision_boost_on_investigate)
        return self.sim.env.effective_vision(int(predator.y), int(predator.x), base, predator=True)

    def _pack_mates(self, predator: Predator) -> list[Predator]:
        radius = self.sim.config.predator.pack_radius
        py, px = int(predator.y), int(predator.x)
        mates = []
        for other in self.sim._predator_index.query(py - radius, py + radius, px - radius, px + radius):
            if other is predator or not other.alive:
                continue
            if self.distance(py, px, int(other.y), int(other.x)) <= radius:
                mates.append(other)
        return mates

    def _centroid(self, group: list) -> tuple[int, int]:
        if not group:
            return 0, 0
        ys = [int(m.y) for m in group]
        xs = [int(m.x) for m in group]
        return int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs)))

    def _maybe_split_pack(self, predator: Predator, pack: list[Predator]) -> None:
        """Split large packs into smaller groups that move apart.

        When a pack exceeds max_pack_size, it splits into roughly equal
        subgroups that are pushed apart from each other.
        """
        cfg = self.sim.config.predator
        all_pack = pack + [predator]
        if len(all_pack) < cfg.max_pack_size:
            return

        split_count = (len(all_pack) + cfg.max_pack_size - 1) // cfg.max_pack_size
        if split_count < 2:
            return

        # Sort by ID for deterministic splitting
        all_pack.sort(key=lambda p: p.id)

        # Distribute into split_count groups
        groups = [[] for _ in range(split_count)]
        for i, p in enumerate(all_pack):
            groups[i % split_count].append(p)

        # Find the group this predator belongs to
        my_group_idx = next(i for i, g in enumerate(groups) if predator in g)
        my_group = groups[my_group_idx]

        # Compute center of my group
        my_center_y = int(round(sum(p.y for p in my_group) / len(my_group)))
        my_center_x = int(round(sum(p.x for p in my_group) / len(my_group)))

        # Push away from other group centers
        push_dy = push_dx = 0.0
        push_dist = float(cfg.split_push_distance)
        for i, g in enumerate(groups):
            if i == my_group_idx:
                continue
            g_center_y = int(round(sum(p.y for p in g) / len(g)))
            g_center_x = int(round(sum(p.x for p in g) / len(g)))
            dy = my_center_y - g_center_y
            dx = my_center_x - g_center_x
            dist = max(1, int(np.sqrt(dy * dy + dx * dx)))
            if dist > 0:
                push_dy += push_dist * dy / dist
                push_dx += push_dist * dx / dist

        # Apply push
        target_y = int(my_center_y + push_dy)
        target_x = int(my_center_x + push_dx)
        target_y, target_x = self.sim.env.clamp(target_y, target_x)
        self.sim.movement.move_predator_toward(predator, target_y, target_x)

    def _predator_wander(self, predator) -> None:
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

    def _pack_mates(self, predator: Predator) -> list[Predator]:
        radius = self.sim.config.predator.pack_radius
        py, px = int(predator.y), int(predator.x)
        mates = []
        for other in self.sim._predator_index.query(py - radius, py + radius, px - radius, px + radius):
            if other is predator or not other.alive:
                continue
            if self.distance(py, px, int(other.y), int(other.x)) <= radius:
                mates.append(other)
        return mates

    def _centroid(self, group: list) -> tuple[int, int]:
        if not group:
            return 0, 0
        ys = [int(m.y) for m in group]
        xs = [int(m.x) for m in group]
        return int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs)))

    def _maybe_split_pack(self, predator: Predator, pack: list[Predator]) -> None:
        """Split large packs into smaller groups that move apart."""
        cfg = self.sim.config.predator
        all_pack = pack + [predator]
        if len(all_pack) < cfg.max_pack_size:
            return

        split_count = (len(all_pack) + cfg.max_pack_size - 1) // cfg.max_pack_size
        if split_count < 2:
            return

        # Sort by ID for deterministic splitting
        all_pack.sort(key=lambda p: p.id)

        # Distribute into split_count groups
        groups = [[] for _ in range(split_count)]
        for i, p in enumerate(all_pack):
            groups[i % split_count].append(p)

        # Find the group this predator belongs to
        my_group_idx = next(i for i, g in enumerate(groups) if predator in g)
        my_group = groups[my_group_idx]

        # Compute center of my group
        my_center_y = int(round(sum(p.y for p in my_group) / len(my_group)))
        my_center_x = int(round(sum(p.x for p in my_group) / len(my_group)))

        # Push away from other group centers
        push_dy = push_dx = 0.0
        push_dist = float(cfg.split_push_distance)
        for i, g in enumerate(groups):
            if i == my_group_idx:
                continue
            g_center_y = int(round(sum(p.y for p in g) / len(g)))
            g_center_x = int(round(sum(p.x for p in g) / len(g)))
            dy = my_center_y - g_center_y
            dx = my_center_x - g_center_x
            dist = max(1, int(np.sqrt(dy * dy + dx * dx)))
            if dist > 0:
                push_dy += push_dist * dy / dist
                push_dx += push_dist * dx / dist

        # Apply push
        target_y = int(my_center_y + push_dy)
        target_x = int(my_center_x + push_dx)
        target_y, target_x = self.sim.env.clamp(target_y, target_x)
        self.sim.movement.move_predator_toward(predator, target_y, target_x)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

import numpy as np
from dataclasses import dataclass, field

from simulator.config import Config
from simulator.environment import Environment
from simulator.entities import Carcass, Grazer, Predator, Rabbit


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class _SpatialIndex:
    """Cell-bucketed neighbour lookup, so perception is O(n * radius^2) not O(n^2).

    Built once per step from a snapshot of positions, which also makes agent
    perception order-independent within a step.
    """

    __slots__ = ("cells",)

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], list] = {}

    def add(self, y: int, x: int, item) -> None:
        self.cells.setdefault((y, x), []).append(item)

    def query(self, y0: int, y1: int, x0: int, x1: int):
        cells = self.cells
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                bucket = cells.get((cy, cx))
                if bucket:
                    yield from bucket


@dataclass
class SimulationStats:
    step: int = 0
    grazer_count: int = 0
    rabbit_count: int = 0
    predator_count: int = 0
    carcass_count: int = 0
    total_grazer_energy: float = 0.0
    total_rabbit_energy: float = 0.0
    total_predator_energy: float = 0.0
    total_carcass_energy: float = 0.0
    avg_grazer_energy: float = 0.0
    avg_rabbit_energy: float = 0.0
    avg_predator_energy: float = 0.0
    total_resource: float = 0.0
    births_grazer: int = 0
    births_rabbit: int = 0
    births_predator: int = 0
    deaths_grazer: int = 0
    deaths_rabbit: int = 0
    deaths_predator: int = 0
    kills_grazer: int = 0
    kills_rabbit: int = 0

    def as_record(self) -> dict:
        return {
            "step": self.step,
            "grazer_count": self.grazer_count,
            "rabbit_count": self.rabbit_count,
            "predator_count": self.predator_count,
            "carcass_count": self.carcass_count,
            "total_grazer_energy": self.total_grazer_energy,
            "total_rabbit_energy": self.total_rabbit_energy,
            "total_predator_energy": self.total_predator_energy,
            "total_carcass_energy": self.total_carcass_energy,
            "avg_grazer_energy": self.avg_grazer_energy,
            "avg_rabbit_energy": self.avg_rabbit_energy,
            "avg_predator_energy": self.avg_predator_energy,
            "total_resource": self.total_resource,
            "births_grazer": self.births_grazer,
            "births_rabbit": self.births_rabbit,
            "births_predator": self.births_predator,
            "deaths_grazer": self.deaths_grazer,
            "deaths_rabbit": self.deaths_rabbit,
            "deaths_predator": self.deaths_predator,
            "kills_grazer": self.kills_grazer,
            "kills_rabbit": self.kills_rabbit,
        }


@dataclass
class _Counters:
    births_grazer: int = 0
    births_rabbit: int = 0
    births_predator: int = 0
    deaths_grazer: int = 0
    deaths_rabbit: int = 0
    deaths_predator: int = 0
    kills_grazer: int = 0
    kills_rabbit: int = 0


class Simulation:
    def __init__(
        self,
        config: Config,
        seed: int | None = None,
        environment: Environment | None = None,
    ):
        self.config = config
        self.map_height = config.simulation.height
        self.map_width = config.simulation.width
        self.rng = np.random.default_rng(seed)

        # Shared RNG so that environment generation and agent decisions replay
        # identically for a given --seed.
        if environment is not None:
            self.env = environment
        else:
            self.env = Environment(
                config.environment, self.map_width, self.map_height, self.rng
            )

        self.grazers: list[Grazer] = []
        self.rabbits: list[Rabbit] = []
        self.predators: list[Predator] = []
        self.carcasses: list[Carcass] = []

        self.stats_history: list[SimulationStats] = []
        self.current_step = 0
        self.counters = _Counters()
        self.ended = False
        self.extinct_step: int | None = None

        self._refresh_fields()
        self._spawn_initial_entities()

    # ------------------------------------------------------------- life cycle

    def reset(self, seed: int | None = None) -> None:
        """R key: fresh populations and statistics, identical environment."""
        environment = self.env
        self.__init__(self.config, seed=seed, environment=environment)

    def reload_environment(self, seed: int | None = None) -> None:
        """E key: regenerate terrain, keep every entity and the statistics history."""
        environment = Environment(
            self.config.environment, self.map_width, self.map_height, self.rng
        )
        history = self.stats_history
        step = self.current_step
        counters = self.counters

        self.__init__(self.config, seed=seed, environment=environment)

        self.stats_history = history
        self.current_step = step
        self.counters = counters
        self.carcasses.clear()

    def _refresh_fields(self) -> None:
        """Vectorised per-step lookup tables derived from the resource field."""
        capacity = self.env.resource_capacity
        self._ratios = np.divide(
            self.env.resources,
            capacity,
            out=np.zeros_like(self.env.resources),
            where=capacity > 0,
        )

    def _build_indices(self) -> None:
        """Snapshot alive agents into spatial indices for neighbour queries."""
        prey = _SpatialIndex()
        for grazer in self.grazers:
            prey.add(int(grazer.y), int(grazer.x), grazer)
        for rabbit in self.rabbits:
            prey.add(int(rabbit.y), int(rabbit.x), rabbit)

        predators = _SpatialIndex()
        for predator in self.predators:
            predators.add(int(predator.y), int(predator.x), predator)

        self._prey_index = prey
        self._predator_index = predators

    # ------------------------------------------------------------- distance

    def distance(self, y1: int, x1: int, y2: int, x2: int) -> int:
        """SPECIFICATION.md bounded Chebyshev distance (toroidal)."""
        dy = abs(y1 - y2)
        dx = abs(x1 - x2)
        return int(max(min(dy, self.map_height - dy), min(dx, self.map_width - dx)))

    # -------------------------------------------------------------- spawning

    def _spawn_location(self) -> tuple[int, int]:
        for _ in range(200):
            y = int(self.rng.integers(0, self.map_height))
            x = int(self.rng.integers(0, self.map_width))
            if self.env.is_passable(y, x):
                return y, x
        return self.map_height // 2, self.map_width // 2

    def _spawn_locations(
        self, count: int, group_size: int, radius: int
    ) -> list[tuple[int, int]]:
        """Clustered spawn positions, so agents start as herds and packs.

        Scattering every individual uniformly left the population with no social
        structure at all: cohesion, alignment and separation had nothing to act
        on, and herds only formed later by accident. Several distinct groups are
        seeded rather than one clump, so the map does not open with a single
        obvious cluster.
        """
        if count <= 0:
            return []
        group_size = max(1, int(group_size))
        radius = max(0, int(radius))

        group_count = -(-count // group_size)  # ceiling division
        centers = [self._spawn_location() for _ in range(group_count)]

        locations: list[tuple[int, int]] = []
        for index in range(count):
            cy, cx = centers[index % group_count]
            if radius == 0:
                locations.append((cy, cx))
                continue
            for _ in range(40):
                offset_y = int(self.rng.integers(-radius, radius + 1))
                offset_x = int(self.rng.integers(-radius, radius + 1))
                ny, nx = self.env.clamp(cy + offset_y, cx + offset_x)
                if self.env.is_passable(ny, nx):
                    locations.append((ny, nx))
                    break
            else:
                locations.append((cy, cx))
        return locations

    def _spawn_initial_entities(self) -> None:
        grazer_cfg = self.config.grazer
        for y, x in self._spawn_locations(
            grazer_cfg.initial_count,
            grazer_cfg.spawn_group_size,
            grazer_cfg.spawn_group_radius,
        ):
            self._add_grazer(y, x)

        rabbit_cfg = self.config.rabbit
        for y, x in self._spawn_locations(
            rabbit_cfg.initial_count,
            rabbit_cfg.spawn_group_size,
            rabbit_cfg.spawn_group_radius,
        ):
            self._add_rabbit(y, x)

        predator_cfg = self.config.predator
        for y, x in self._spawn_locations(
            predator_cfg.initial_count,
            predator_cfg.spawn_group_size,
            predator_cfg.spawn_group_radius,
        ):
            self._add_predator(y, x)

    def _spawn_age(self, cfg) -> int:
        """Starting age drawn uniformly from 0 to maturity.

        Seeding every individual as newborn synchronises the whole population's
        first reproduction wave and its first mortality wave, which produces
        artificial boom-bust cycles that have nothing to do with the model.
        """
        return int(self.rng.integers(0, int(cfg.reproduction_min_age) + 1))

    def _add_grazer(self, y: int | None = None, x: int | None = None) -> Grazer:
        cfg = self.config.grazer
        if y is None or x is None:
            y, x = self._spawn_location()
        weight = float(np.clip(self.rng.normal(cfg.gradient_weight, cfg.gradient_weight_std), 0.0, 1.0))
        grazer = Grazer(
            x=float(x),
            y=float(y),
            energy=cfg.max_energy * float(self.rng.uniform(0.5, 1.0)),
            config=cfg,
            age=self._spawn_age(cfg),
            gradient_weight=weight,
            inherited_gradient_weight=weight,
        )
        self.grazers.append(grazer)
        return grazer

    def _add_rabbit(self, y: int | None = None, x: int | None = None) -> Rabbit:
        cfg = self.config.rabbit
        if y is None or x is None:
            y, x = self._spawn_location()
        weight = float(np.clip(self.rng.normal(cfg.gradient_weight, cfg.gradient_weight_std), 0.0, 1.0))
        rabbit = Rabbit(
            x=float(x),
            y=float(y),
            energy=cfg.max_energy * float(self.rng.uniform(0.5, 1.0)),
            config=cfg,
            age=self._spawn_age(cfg),
            gradient_weight=weight,
            inherited_gradient_weight=weight,
        )
        self.rabbits.append(rabbit)
        return rabbit

    def _add_predator(self, y: int | None = None, x: int | None = None) -> Predator:
        cfg = self.config.predator
        if y is None or x is None:
            y, x = self._spawn_location()
        predator = Predator(
            x=float(x),
            y=float(y),
            energy=cfg.max_energy * float(self.rng.uniform(0.5, 1.0)),
            config=cfg,
            age=self._spawn_age(cfg),
        )
        self.predators.append(predator)
        return predator

    def spawn_grazer_at(self, y: int, x: int) -> Grazer | None:
        if not self.env.is_passable(y, x):
            return None
        self._add_grazer(y, x)
        return self.grazers[-1]

    def spawn_predator_at(self, y: int, x: int) -> Predator | None:
        if not self.env.is_passable(y, x):
            return None
        self._add_predator(y, x)
        return self.predators[-1]

    # ------------------------------------------------------------------ loop

    def step(self) -> None:
        self.env.step()
        self._refresh_fields()
        self._build_indices()

        for carcass in self.carcasses:
            carcass.step()

        self._update_prey(self.grazers, self.config.grazer, "grazer")
        self._update_prey(self.rabbits, self.config.rabbit, "rabbit")
        self._update_predators()

        self._reproduce()
        self._cleanup()

        stats = self._record_stats()
        self.stats_history.append(stats)
        self.current_step += 1

        if not self.grazers and not self.rabbits and not self.predators:
            self.ended = True
            self.extinct_step = self.current_step

    def run(self, steps: int | None = None) -> list[SimulationStats]:
        limit = steps if steps is not None else self.config.simulation.max_steps
        if limit is None or limit <= 0:
            limit = float("inf")
        while self.current_step < limit and not self.ended:
            self.step()
        return self.stats_history

    # ------------------------------------------------------------------ prey

    def _foraging_state(self, prey, cfg) -> str:
        """Decide whether this prey feeds where it stands or travels.

        The two thresholds form a hysteresis band rather than a single setpoint.
        A feeding prey commits until it is full (or has stripped the cell), and
        once it leaves it will not settle again until it has walked down to
        ``graze_resume_energy``. With one shared threshold the species thrashed:
        it topped up to full, stepped one cell, immediately qualified as
        "hungry" again on any decent ground, and spent its life grazing in place.

        While travelling, cohesion dominates so the herd moves as a pack toward
        fresh ground instead of each individual picking its own target.
        """
        if not cfg.migratory:
            return "grazing"

        ratio = self.env.get_resource_ratio(int(prey.y), int(prey.x))
        energy_ratio = prey.energy_ratio

        if prey.foraging_state == "grazing":
            # Stay until full, or until there is nothing left here worth eating.
            if (
                energy_ratio >= cfg.graze_energy_target
                or ratio <= cfg.migrate_ratio_threshold
                or energy_ratio >= cfg.wander_energy_threshold
            ):
                return "migrating"
            return "grazing"

        # Travelling: only settle once genuinely hungry, and only on good ground.
        if (
            energy_ratio <= cfg.graze_resume_energy
            and ratio >= cfg.graze_ratio_threshold
        ):
            return "grazing"
        return "migrating"

    def _update_prey(self, flock: list, cfg, kind: str) -> None:
        for prey in flock:
            if not prey.alive:
                continue

            was_alive = prey.alive
            prey.step_energy()
            if was_alive and not prey.alive:
                self._record_death(kind)

            if not prey.alive:
                continue

            state = self._foraging_state(prey, cfg)

            if cfg.migratory:
                prey.foraging_state = state
                if state == "grazing":
                    # Standing still: feed hard, do not move.
                    prey.eat(self.env, cfg.grazing_efficiency)
                    continue
                # Travelling: feed on the move, at a rate that cannot cover
                # metabolism, so the journey costs energy.
                prey.eat(self.env, cfg.moving_efficiency)
                herd = self._herd_vectors(prey, self._prey_index, cfg)
                target = self._forage_target(prey, cfg)
                if self._move_prey(prey, cfg, herd, target):
                    self._record_death(kind)
                continue

            herd = self._herd_vectors(prey, self._prey_index, cfg)
            target = self._forage_target(prey, cfg)
            if self._move_prey(prey, cfg, herd, target):
                self._record_death(kind)
            prey.eat(self.env)

    def _herd_vectors(self, prey, index: _SpatialIndex, cfg) -> tuple | None:
        """Cohesion, alignment and separation from neighbours inside herd_radius.

        Direction vectors use plain grid deltas and plain Chebyshev gaps: the
        toroidal metric in SPECIFICATION.md is only meaningful for perception
        radii, and using it here would report herd members as adjacent across the
        impassable ocean border. Separation points *away* from a neighbour --
        SPECIFICATION.md line 102 has that sign inverted relative to its own
        stated intent, "avoid crowding".
        """
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

    def _forage_target(self, prey, cfg) -> tuple[int, int]:
        """Best cell inside the prey's vision disc, scored on resource availability."""
        py, px = int(prey.y), int(prey.x)
        radius = max(1, int(prey.vision))
        y0, y1 = max(0, py - radius), min(self.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.map_width, px + radius + 1)

        window = self._ratios[y0:y1, x0:x1]
        reachable = self.env.passable[y0:y1, x0:x1]
        score = np.where(reachable, window, -np.inf)

        # Well-fed prey can afford to prospect away from rich ground, but a
        # migratory species is looking for food, so prospecting is counterproductive.
        if prey.energy_ratio > 0.8 and not cfg.migratory:
            score = score + np.where(reachable, (1.0 - window) * 0.3, 0.0)

        rows, cols = np.indices(score.shape)
        score = score - 0.02 * np.maximum(
            np.abs(rows + y0 - py), np.abs(cols + x0 - px)
        )

        best = int(np.argmax(score))
        if not np.isfinite(score.flat[best]):
            return py, px
        return y0 + best // score.shape[1], x0 + best % score.shape[1]

    def _threat_vector(self, prey, radius: int) -> tuple[float, float] | None:
        """Weighted escape direction away from predators inside flee_radius."""
        if radius <= 0:
            return None
        py, px = int(prey.y), int(prey.x)
        away_y = away_x = 0.0
        found = False
        for threat in self._predator_index.query(py - radius, py + radius, px - radius, px + radius):
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

    def _move_prey(self, prey, cfg, herd: tuple | None, target: tuple[int, int]) -> bool:
        """Advance a prey. Returns True if the movement cost starved it.

        Terrain mobility decides how many frames a cell takes to cross, and the
        behavioural multipliers modulate both speed and its price: fleeing is
        1.3x faster at 2.5x the movement metabolism, and a continuous grazer
        shuffling along while feeding moves at two thirds speed.
        """
        py, px = int(prey.y), int(prey.x)
        threat = self._threat_vector(prey, cfg.flee_radius)
        fleeing = threat is not None

        speed_multiplier = cfg.flee_speed_multiplier if fleeing else 1.0
        if not cfg.migratory:
            speed_multiplier *= cfg.grazing_move_multiplier
        cost_multiplier = cfg.flee_cost_multiplier if fleeing else 1.0

        herd_weighted = None
        if herd is not None and self.rng.random() < cfg.herd_follow_chance:
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

        mobility = self.env.mobility_prey[py, px]
        allowance = prey.movement_allowance(mobility * speed_multiplier)
        moved = 0

        for _ in range(allowance):
            ty, tx = target
            current_to_target = self.distance(py, px, ty, tx)
            best = None
            best_score = -np.inf

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    ny, nx = self.env.clamp(ny, nx)
                    if not self.env.passable[ny, nx]:
                        continue

                    # Gradient-following pulls toward the forage target; the
                    # individual's gradient_weight scales how strongly.
                    approach = current_to_target - self.distance(ny, nx, ty, tx)
                    score = approach * gradient_bias

                    if herd_weighted is not None:
                        score += (
                            dy * herd_weighted[0] + dx * herd_weighted[1]
                        ) * (1.0 - gradient_bias) * 0.5
                        score += dy * herd_weighted[2] + dx * herd_weighted[3]

                    # Fleeing outweighs foraging: without this, predators that
                    # close at 2 cells/step against prey's 1 have no way to lose
                    # a chase, and prey are harvested to extinction.
                    if threat is not None:
                        score += (dy * threat[0] + dx * threat[1]) * cfg.flee_weight

                    # Peak-condition prey tolerates rough, slow terrain.
                    if prey.energy_ratio > 0.95:
                        score += (1.0 - self.env.mobility_prey[ny, nx]) * 1.5

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

        prey.energy -= moved * cfg.movement_energy_cost * cost_multiplier
        if prey.energy <= 0:
            prey.energy = 0.0
            prey.alive = False
            return True
        return False

    # -------------------------------------------------------------- predators

    def _update_predators(self) -> None:
        # Reset the transient behaviour label; "feeding" only suppresses movement
        # for the single step in which a predator actually drew on a carcass.
        for predator in self.predators:
            predator.eating_state = "moving"

        # Carcass feeding is resolved as one coordinated pass before any predator
        # moves, so that every predator around the same carcass competes for one
        # shared energy store in a deterministic order.
        self._feed_on_carcasses()

        for predator in self.predators:
            if not predator.alive:
                continue

            was_alive = predator.alive
            predator.step_energy()
            if was_alive and not predator.alive:
                self.counters.deaths_predator += 1
            if not predator.alive:
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

    def _feed_on_carcasses(self) -> None:
        """Scavenge pass over every carcass.

        All predators within feed_radius (their own cell plus the eight adjacent
        ones) draw on the same carcass, so a carcass surrounded by more predators
        is stripped proportionally faster. Feeding is ordered oldest-first, which
        decides who gets the remainder when the store runs out mid-pass.
        """
        cfg = self.config.carcass
        per_step = self.config.predator.energy_from_prey / cfg.consumption_divisor
        radius = max(0, int(cfg.feed_radius))

        for carcass in self.carcasses:
            if not carcass.provides_energy:
                continue
            cy, cx = int(carcass.y), int(carcass.x)
            feeders = [
                predator
                for predator in self._predator_index.query(
                    cy - radius, cy + radius, cx - radius, cx + radius
                )
                if predator.alive
                and self.distance(int(predator.y), int(predator.x), cy, cx) <= radius
            ]
            if not feeders:
                continue

            # Oldest first; id breaks ties so the outcome is reproducible.
            feeders.sort(key=lambda p: (-p.age, p.id))

            for predator in feeders:
                if not carcass.provides_energy:
                    break
                taken = carcass.take_energy(per_step)
                if taken <= 0:
                    break
                predator.add_energy(taken)
                carcass.consumed_by.append(predator.id)
                predator.eating_state = "feeding"

    def _predator_kill_prey(self, predator: Predator) -> bool:
        """Kill co-located prey and leave a carcass. No energy is transferred here."""
        py, px = int(predator.y), int(predator.x)
        cfg = self.config.predator
        fraction = self.config.carcass.carcass_energy_fraction

        for flock, kind, kill_counter in (
            (self.grazers, "grazer", "kills_grazer"),
            (self.rabbits, "rabbit", "kills_rabbit"),
        ):
            for prey in flock:
                if not prey.alive:
                    continue
                if int(prey.y) != py or int(prey.x) != px:
                    continue
                prey.alive = False
                self._record_death(kind)
                setattr(self.counters, kill_counter, getattr(self.counters, kill_counter) + 1)
                self.carcasses.append(
                    Carcass(
                        x=prey.x,
                        y=prey.y,
                        energy=prey.energy * fraction,
                        max_energy=prey.energy * fraction,
                        config=self.config.carcass,
                    )
                )
                predator.eating_state = "killing"
                return True
        return False

    def _predator_behaviour(self, predator: Predator) -> None:
        cfg = self.config.predator
        py, px = int(predator.y), int(predator.x)

        pack = self._pack_mates(predator)
        pack_center = self._centroid(pack) if pack else None

        mode = "wander"
        if pack and self.rng.random() < cfg.pack_hunt_chance:
            mode = "pack_hunt"
        elif self._solo_target(predator) is not None:
            mode = "solo"

        if mode == "pack_hunt":
            target = self._pack_target(predator, pack)
            if target is not None:
                flank = pack_center is not None and self.rng.random() < cfg.flank_chance
                goal = self._flank_goal(target, pack_center) if flank else (
                    int(target.y),
                    int(target.x),
                )
                self._move_predator_toward(predator, goal[0], goal[1], chasing=True)
                return

        if mode == "solo":
            target = self._solo_target(predator)
            if target is not None:
                self._move_predator_toward(predator, int(target.y), int(target.x), chasing=True)
                return

        # No prey in sight: infer their presence from stripped ground.
        if self._investigate_depletion(predator):
            return

        if pack and pack_center is not None and self.rng.random() < cfg.pack_wander_chance:
            self._move_predator_toward(predator, pack_center[0], pack_center[1])
            return

        self._predator_wander(predator)

    def _pack_mates(self, predator: Predator) -> list[Predator]:
        radius = self.config.predator.pack_radius
        py, px = int(predator.y), int(predator.x)
        mates = []
        for other in self._predator_index.query(py - radius, py + radius, px - radius, px + radius):
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

    def _prey_vision(self, predator: Predator) -> int:
        cfg = self.config.predator
        base = cfg.chase_radius
        if predator.investigating:
            base = int(base * cfg.vision_boost_on_investigate)
        return self.env.effective_vision(
            int(predator.y), int(predator.x), base, predator=True
        )

    def _solo_target(self, predator: Predator):
        vision = self._prey_vision(predator)
        py, px = int(predator.y), int(predator.x)
        best = None
        best_distance = float("inf")
        for prey in self._prey_index.query(py - vision, py + vision, px - vision, px + vision):
            if not prey.alive:
                continue
            gap = self.distance(py, px, int(prey.y), int(prey.x))
            if gap <= vision and gap < best_distance:
                best, best_distance = prey, gap
        return best

    def _pack_target(self, predator: Predator, pack: list):
        """SPECIFICATION.md: score = (chase_radius + 5 - distance) + nearby_predators * 3."""
        cfg = self.config.predator
        py, px = int(predator.y), int(predator.x)
        vision = self._prey_vision(predator)
        pack_positions = [(int(m.y), int(m.x)) for m in pack]

        best = None
        best_score = -float("inf")
        for prey in self._prey_index.query(py - vision, py + vision, px - vision, px + vision):
            if not prey.alive:
                continue
            ty, tx = int(prey.y), int(prey.x)
            gap = self.distance(py, px, ty, tx)
            if gap > vision:
                continue
            escorts = sum(
                1
                for my, mx in pack_positions
                if self.distance(my, mx, ty, tx) <= cfg.pack_radius
            )
            score = (cfg.chase_radius + 5 - gap) + escorts * cfg.pack_score_bonus
            if score > best_score:
                best, best_score = prey, score
        return best

    def _flank_goal(self, prey, pack_center: tuple[int, int]) -> tuple[int, int]:
        """Approach from the side of the prey facing away from the pack centre."""
        ty, tx = int(prey.y), int(prey.x)
        cy, cx = pack_center
        return ty + _sign(ty - cy), tx + _sign(tx - cx)

    def _investigate_depletion(self, predator: Predator) -> bool:
        """Resource sensing: head for freshly stripped ground inside sense radius."""
        cfg = self.config.predator
        py, px = int(predator.y), int(predator.x)

        if predator.investigating:
            predator.investigate_timer -= 1
            if predator.investigate_timer <= 0:
                predator.clear_investigation()
            else:
                self._move_predator_toward(
                    predator, predator.investigate_y, predator.investigate_x
                )
                return True

        radius = max(1, int(cfg.resource_sense_radius))
        radius = int(radius * self.env.visibility(py, px, predator=True))
        y0, y1 = max(0, py - radius), min(self.map_height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.map_width, px + radius + 1)

        ratios = self._ratios[y0:y1, x0:x1]
        depletion = self.env.depletion_rate[y0:y1, x0:x1]
        reachable = self.env.passable[y0:y1, x0:x1]

        evidence = np.where(
            reachable & (ratios <= cfg.depleted_ratio_threshold), depletion, 0.0
        )
        if float(evidence.max()) < cfg.investigate_threshold:
            return False

        best = int(np.argmax(evidence))
        predator.begin_investigation(
            y0 + best // evidence.shape[1],
            x0 + best % evidence.shape[1],
            cfg.investigate_steps,
        )
        self._move_predator_toward(
            predator, predator.investigate_y, predator.investigate_x
        )
        return True

    def _move_predator_toward(
        self, predator: Predator, ty: int, tx: int, chasing: bool = False
    ) -> None:
        """Advance toward a goal. Chasing costs double movement metabolism."""
        cfg = self.config.predator
        py, px = int(predator.y), int(predator.x)
        mobility = self.env.mobility_predator[py, px]

        speed = cfg.chase_speed_multiplier if chasing else 1.0
        cost = cfg.chase_cost_multiplier if chasing else 1.0
        allowance = predator.movement_allowance(mobility * speed)
        moved = 0

        for _ in range(allowance):
            gap = self.distance(py, px, ty, tx)
            if gap == 0:
                break
            best = None
            best_gap = gap
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = self.env.clamp(py + dy, px + dx)
                    if not self.env.passable[ny, nx]:
                        continue
                    candidate_gap = self.distance(ny, nx, ty, tx)
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

        predator.energy -= moved * cfg.movement_energy_cost * cost
        if predator.energy <= 0:
            predator.energy = 0.0
            predator.alive = False

    def _predator_wander(self, predator: Predator) -> None:
        py, px = int(predator.y), int(predator.x)
        mobility = self.env.mobility_predator[py, px]
        steps = predator.movement_allowance(mobility)

        for _ in range(steps):
            best = None
            best_score = -float("inf")
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = self.env.clamp(py + dy, px + dx)
                    if not self.env.passable[ny, nx]:
                        continue
                    centre = max(
                        abs(ny - self.map_height // 2), abs(nx - self.map_width // 2)
                    )
                    score = self._ratios[ny, nx] - centre * 0.01
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

    # ------------------------------------------------------------ reproduction

    def _reproduce(self) -> None:
        self.grazers.extend(
            offspring
            for parent in list(self.grazers)
            if parent.alive
            for offspring in [self._try_reproduce(parent, self.config.grazer)]
            if offspring is not None
        )
        self.rabbits.extend(
            offspring
            for parent in list(self.rabbits)
            if parent.alive
            for offspring in [self._try_reproduce(parent, self.config.rabbit)]
            if offspring is not None
        )
        self.predators.extend(
            offspring
            for parent in list(self.predators)
            if parent.alive
            for offspring in [self._try_reproduce(parent, self.config.predator)]
            if offspring is not None
        )

    def _try_reproduce(self, parent, cfg):
        """SPECIFICATION.md: cost is a fraction of the parent's *current* energy."""
        if parent.age < cfg.reproduction_min_age or parent.energy < cfg.reproduction_min_energy:
            return None
        cost = parent.energy * cfg.reproduction_energy_cost
        parent.energy -= cost

        py, px = int(parent.y), int(parent.x)
        oy = py + int(self.rng.integers(-1, 2))
        ox = px + int(self.rng.integers(-1, 2))
        oy, ox = self.env.clamp(oy, ox)

        energy = float(cfg.offspring_initial_energy)
        if isinstance(parent, Grazer):
            child = Grazer(x=float(ox), y=float(oy), energy=energy, config=cfg)
            child.inherited_gradient_weight = parent.inherited_gradient_weight
            child.mutate_gradient_weight(self.rng)
            self.counters.births_grazer += 1
        elif isinstance(parent, Rabbit):
            child = Rabbit(x=float(ox), y=float(oy), energy=energy, config=cfg)
            child.inherited_gradient_weight = parent.inherited_gradient_weight
            child.mutate_gradient_weight(self.rng)
            self.counters.births_rabbit += 1
        else:
            child = Predator(x=float(ox), y=float(oy), energy=energy, config=cfg)
            self.counters.births_predator += 1
        return child

    def _record_death(self, kind: str) -> None:
        if kind == "grazer":
            self.counters.deaths_grazer += 1
        else:
            self.counters.deaths_rabbit += 1

    def _cleanup(self) -> None:
        self.grazers = [g for g in self.grazers if g.alive]
        self.rabbits = [r for r in self.rabbits if r.alive]
        self.predators = [p for p in self.predators if p.alive]
        self.carcasses = [c for c in self.carcasses if c.provides_energy]

    # -------------------------------------------------------------- reporting

    def _record_stats(self) -> SimulationStats:
        grazers, rabbits, predators = self.grazers, self.rabbits, self.predators
        live_carcasses = [c for c in self.carcasses if c.provides_energy]

        stats = SimulationStats(
            step=self.current_step,
            grazer_count=len(grazers),
            rabbit_count=len(rabbits),
            predator_count=len(predators),
            carcass_count=len(live_carcasses),
            births_grazer=self.counters.births_grazer,
            births_rabbit=self.counters.births_rabbit,
            births_predator=self.counters.births_predator,
            deaths_grazer=self.counters.deaths_grazer,
            deaths_rabbit=self.counters.deaths_rabbit,
            deaths_predator=self.counters.deaths_predator,
            kills_grazer=self.counters.kills_grazer,
            kills_rabbit=self.counters.kills_rabbit,
        )
        stats.total_grazer_energy = float(sum(g.energy for g in grazers))
        stats.total_rabbit_energy = float(sum(r.energy for r in rabbits))
        stats.total_predator_energy = float(sum(p.energy for p in predators))
        stats.total_carcass_energy = float(sum(c.energy for c in live_carcasses))
        stats.total_resource = self.env.total_resource()

        if grazers:
            stats.avg_grazer_energy = stats.total_grazer_energy / len(grazers)
        if rabbits:
            stats.avg_rabbit_energy = stats.total_rabbit_energy / len(rabbits)
        if predators:
            stats.avg_predator_energy = stats.total_predator_energy / len(predators)
        return stats

    def get_state(self) -> dict:
        terrain, resources = self.env.as_lists()
        return {
            "step": self.current_step,
            "grazers": [
                {"id": g.id, "x": g.x, "y": g.y, "energy": g.energy, "age": g.age}
                for g in self.grazers
            ],
            "rabbits": [
                {"id": r.id, "x": r.x, "y": r.y, "energy": r.energy, "age": r.age}
                for r in self.rabbits
            ],
            "predators": [
                {"id": p.id, "x": p.x, "y": p.y, "energy": p.energy, "age": p.age}
                for p in self.predators
            ],
            "carcasses": [
                {"id": c.id, "x": c.x, "y": c.y, "energy": c.energy}
                for c in self.carcasses
                if c.provides_energy
            ],
            "resources": resources,
            "terrain": terrain,
        }

    def get_statistics(self) -> dict:
        return {
            "grazers": len(self.grazers),
            "rabbits": len(self.rabbits),
            "predators": len(self.predators),
            "carcasses": len([c for c in self.carcasses if c.provides_energy]),
            "total_resource": self.env.total_resource(),
            "grazers_list": list(self.grazers),
            "rabbits_list": list(self.rabbits),
            "predators_list": list(self.predators),
        }

    def extinction_summary(self) -> str:
        peak = {
            "grazer": max((s.grazer_count for s in self.stats_history), default=0),
            "rabbit": max((s.rabbit_count for s in self.stats_history), default=0),
            "predator": max((s.predator_count for s in self.stats_history), default=0),
        }
        return "\n".join(
            [
                "=== SIMULATION ENDED (all species extinct) ===",
                f"Final step: {self.current_step}",
                f"Births   grazer={self.counters.births_grazer} "
                f"rabbit={self.counters.births_rabbit} "
                f"predator={self.counters.births_predator}",
                f"Deaths   grazer={self.counters.deaths_grazer} "
                f"rabbit={self.counters.deaths_rabbit} "
                f"predator={self.counters.deaths_predator}",
                f"Kills    grazer={self.counters.kills_grazer} "
                f"rabbit={self.counters.kills_rabbit}",
                f"Peak     grazer={peak['grazer']} "
                f"rabbit={peak['rabbit']} "
                f"predator={peak['predator']}",
            ]
        )
        """Reset the simulation to initial state."""        self.__init__(self.config, seed=self.rng.integers(0, 2**32-1))    def reset(self) -> None:import numpy as np
from dataclasses import dataclass
from typing import Optional
from simulator.config import Config, GrazerConfig, RabbitConfig, PredatorConfig
from simulator.environment import Environment
from simulator.entities import Grazer, Rabbit, Predator, Carcass, Entity


def _sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


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
    avg_grazer_energy: float = 0.0
    avg_rabbit_energy: float = 0.0
    avg_predator_energy: float = 0.0
    grazer_births: int = 0
    rabbit_births: int = 0
    predator_births: int = 0
    grazer_deaths: int = 0
    rabbit_deaths: int = 0
    predator_deaths: int = 0


class Simulation:
    def __init__(self, config: Config, seed: int | None = None):
        self.config = config
        self.map_width = config.simulation.width
        self.map_height = config.simulation.height
        self._ocean_border = config.environment.ocean_border_width
        self.rng = np.random.default_rng(seed)

        self.env = Environment(config.environment, self.map_width, self.map_height, seed)

        self.grazers: list[Grazer] = []
        self.rabbits: list[Rabbit] = []
        self.predators: list[Predator] = []
        self.carcasses: list[Carcass] = []

        self.stats_history: list[SimulationStats] = []
        self.current_step = 0

        self._spawn_initial_entities()

    def _spawn_initial_entities(self) -> None:
        for _ in range(self.config.grazer.initial_count):
            self._spawn_grazer()

        for _ in range(self.config.rabbit.initial_count):
            self._spawn_rabbit()

        for _ in range(self.config.predator.initial_count):
            self._spawn_predator()

    def _spawn_grazer(self) -> None:
        y, x = self._find_spawn_location()
        grazer = Grazer(
            x=float(x),
            y=float(y),
            energy=self.config.grazer.max_energy * self.rng.uniform(0.5, 1.0),
            max_energy=self.config.grazer.max_energy,
            config=self.config.grazer,
        )
        grazer.gradient_weight = self.rng.normal(
            self.config.grazer.gradient_weight, self.config.grazer.gradient_weight_std
        )
        grazer.gradient_weight = np.clip(grazer.gradient_weight, 0.1, 1.0)
        self.grazers.append(grazer)

    def _spawn_rabbit(self) -> None:
        y, x = self._find_spawn_location()
        rabbit = Rabbit(
            x=float(x),
            y=float(y),
            energy=self.config.rabbit.max_energy * self.rng.uniform(0.5, 1.0),
            max_energy=self.config.rabbit.max_energy,
            config=self.config.rabbit,
        )
        rabbit.gradient_weight = self.rng.normal(
            self.config.rabbit.gradient_weight, self.config.rabbit.gradient_weight_std
        )
        rabbit.gradient_weight = np.clip(rabbit.gradient_weight, 0.1, 1.0)
        self.rabbits.append(rabbit)

    def _spawn_predator(self) -> None:
        y, x = self._find_spawn_location()
        predator = Predator(
            x=float(x),
            y=float(y),
            energy=self.config.predator.max_energy * self.rng.uniform(0.5, 1.0),
            max_energy=self.config.predator.max_energy,
            config=self.config.predator,
        )
        self.predators.append(predator)

    def _find_spawn_location(self) -> tuple[int, int]:
        for _ in range(100):
            y = self.rng.integers(self._ocean_border, self.map_height - self._ocean_border)
            x = self.rng.integers(self._ocean_border, self.map_width - self._ocean_border)
            if self.env.is_passable(y, x):
                return y, x
        return self.map_height // 2, self.map_width // 2

    def _clamp(self, value: float, min_val: int, max_val: int) -> float:
        return max(min_val, min(max_val, value))

    def _grazer_move(self, g: Grazer) -> None:
        y, x = int(g.y), int(g.x)
        min_y = self._ocean_border
        max_y = self.map_height - 1 - self._ocean_border
        min_x = self._ocean_border
        max_x = self.map_width - 1 - self._ocean_border

        current_terrain = self.env.get_terrain_at(y, x)
        current_mobility = self.env.get_mobility_prey(current_terrain)

        herd_radius = 6
        separation_radius = 2

        neighbors = []
        for other in self.grazers:
            if other is g or not other.alive:
                continue
            dy = abs(int(other.y) - y)
            dx = abs(int(other.x) - x)
            dist = max(dy, dx)
            if dist <= herd_radius:
                neighbors.append((other, dist))

        cohesion_dy, cohesion_dx = 0, 0
        separation_dy, separation_dx = 0, 0

        if neighbors:
            for other, dist in neighbors:
                dy = int(other.y) - int(g.y)
                dx = int(other.x) - int(g.x)
                cohesion_dy += dy
                cohesion_dx += dx

                if dist <= separation_radius:
                    separation_dy -= dy * (separation_radius - dist + 1)
                    separation_dx -= dx * (separation_radius - dist + 1)

            n = len(neighbors)
            cohesion_dy, cohesion_dx = cohesion_dy / n, cohesion_dx / n

        best_dy, best_dx = 0, 0

        for _ in range(max(1, int(g.move_speed * current_mobility))):
            candidates = []
            energy_ratio = g.energy / g.max_energy if g.max_energy > 0 else 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny = self._clamp(int(g.y) + dy, 0, self.map_height - 1)
                    nx = self._clamp(int(g.x) + dx, 0, self.map_width - 1)
                    if not self.env.is_passable(ny, nx):
                        continue
                    target_terrain = self.env.get_terrain_at(ny, nx)
                    target_mobility = self.env.get_mobility_prey(target_terrain)
                    resource_ratio = self.env.get_resource_ratio(ny, nx)
                    candidates.append((dy, dx, resource_ratio, target_mobility, target_terrain))

            if not candidates:
                break

            use_gradient = self.rng.random() < g.gradient_weight

            if use_gradient and candidates:
                energy_ratio = g.energy / g.max_energy if g.max_energy > 0 else 0
                scored_candidates = []
                for dy, dx, resource_ratio, target_mobility, target_terrain in candidates:
                    score = resource_ratio
                    if energy_ratio > 0.7:
                        score += (1.0 - resource_ratio) * 0.3
                    scored_candidates.append((dy, dx, score))

                scored_candidates.sort(key=lambda c: c[2], reverse=True)
                best_dy, best_dx, _ = scored_candidates[0]
            else:
                scored_candidates = []
                for dy, dx, resource_ratio, target_mobility, target_terrain in candidates:
                    score = resource_ratio
                    if energy_ratio > 0.7:
                        score += (1.0 - resource_ratio) * 0.2
                    scored_candidates.append((dy, dx, score))

                total_score = sum(c[2] for c in scored_candidates)
                if total_score > 0:
                    r_val = self.rng.uniform(0, total_score)
                    cumsum = 0
                    for dy, dx, score in scored_candidates:
                        cumsum += score
                        if cumsum >= r_val:
                            best_dy, best_dx = dy, dx
                            break
                else:
                    best_dy, best_dx, _, _, _ = candidates[self.rng.integers(0, len(candidates))]

                if neighbors and self.rng.random() < 0.7:
                    target_dy = int(_sign(cohesion_dy)) if cohesion_dy != 0 else 0
                    target_dx = int(_sign(cohesion_dx)) if cohesion_dx != 0 else 0
                    if target_dy != 0 or target_dx != 0:
                        for dy, dx, _, _, _ in candidates:
                            if dy == target_dy and dx == target_dx:
                                best_dy, best_dx = dy, dx
                                break

            g.y = self._clamp(g.y + best_dy, 0, self.map_height - 1)
            g.x = self._clamp(g.x + best_dx, 0, self.map_width - 1)

    def _grazer_eat(self, g: Grazer) -> None:
        y, x = int(g.y), int(g.x)
        ratio = self.env.get_resource_ratio(y, x)
        if ratio <= 0:
            return

        if g.energy >= g.max_energy * 0.9:
            return

        amount = self.config.grazer.energy_from_resource

        if g.eating_state != "grazing":
            eaten = g.eat_while_moving(amount)
            if ratio > 0.5 and self.rng.random() < 0.3:
                g.eating_state = "grazing"
                g.grazing_timer = 3
        else:
            eaten = g.eat_while_stopped(amount)
            g.grazing_timer -= 1
            if g.grazing_timer <= 0:
                g.eating_state = "moving"

        if g.fat > 0 and g.energy < g.max_energy * 0.3:
            g.use_fat(min(g.fat, 20.0))

    def _update_grazers(self) -> None:
        for g in self.grazers:
            if not g.alive:
                continue
            g.step_energy(self.config.grazer.energy_per_step)
            if g.alive:
                self._grazer_move(g)
                self._grazer_eat(g)

    def _rabbit_move(self, r: Rabbit) -> None:
        y, x = int(r.y), int(r.x)
        min_y = self._ocean_border
        max_y = self.map_height - 1 - self._ocean_border
        min_x = self._ocean_border
        max_x = self.map_width - 1 - self._ocean_border

        current_terrain = self.env.get_terrain_at(y, x)
        current_mobility = self.env.get_mobility_prey(current_terrain)

        herd_radius = 6
        separation_radius = 2
        flee_radius = 5

        neighbors = []
        for other in self.rabbits:
            if other is r or not other.alive:
                continue
            dy = abs(int(other.y) - y)
            dx = abs(int(other.x) - x)
            dist = max(dy, dx)
            if dist <= herd_radius:
                neighbors.append((other, dist))

        nearby_predators = []
        for p in self.predators:
            if not p.alive:
                continue
            dist = max(abs(int(p.y) - y), abs(int(p.x) - x))
            if dist <= flee_radius:
                nearby_predators.append((p, dist))

        cohesion_dy, cohesion_dx = 0, 0
        separation_dy, separation_dx = 0, 0

        if neighbors:
            for other, dist in neighbors:
                dy = int(other.y) - int(r.y)
                dx = int(other.x) - int(r.x)
                cohesion_dy += dy
                cohesion_dx += dx

                if dist <= separation_radius:
                    separation_dy -= dy * (separation_radius - dist + 1)
                    separation_dx -= dx * (separation_radius - dist + 1)

            n = len(neighbors)
            cohesion_dy, cohesion_dx = cohesion_dy / n, cohesion_dx / n

        best_dy, best_dx = 0, 0

        for _ in range(max(1, int(r.move_speed * current_mobility))):
            candidates = []
            energy_ratio = r.energy / r.max_energy if r.max_energy > 0 else 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny = self._clamp(int(r.y) + dy, 0, self.map_height - 1)
                    nx = self._clamp(int(r.x) + dx, 0, self.map_width - 1)
                    if not self.env.is_passable(ny, nx):
                        continue
                    target_terrain = self.env.get_terrain_at(ny, nx)
                    target_mobility = self.env.get_mobility_prey(target_terrain)
                    resource_ratio = self.env.get_resource_ratio(ny, nx)
                    candidates.append((dy, dx, resource_ratio, target_mobility, target_terrain))

            if not candidates:
                break

            use_gradient = self.rng.random() < r.gradient_weight

            if use_gradient and candidates:
                energy_ratio = r.energy / r.max_energy if r.max_energy > 0 else 0
                scored_candidates = []
                for dy, dx, resource_ratio, target_mobility, target_terrain in candidates:
                    score = resource_ratio
                    if energy_ratio > 0.7:
                        score += (1.0 - resource_ratio) * 0.4
                    scored_candidates.append((dy, dx, score))

                scored_candidates.sort(key=lambda c: c[2], reverse=True)
                best_dy, best_dx, _ = scored_candidates[0]
            else:
                scored_candidates = []
                for dy, dx, resource_ratio, target_mobility, target_terrain in candidates:
                    score = resource_ratio
                    if r.energy / r.max_energy > 0.7:
                        score += (1.0 - resource_ratio) * 0.3
                    scored_candidates.append((dy, dx, score))

                total_score = sum(c[2] for c in scored_candidates)
                if total_score > 0:
                    r_val = self.rng.uniform(0, total_score)
                    cumsum = 0
                    for dy, dx, score in scored_candidates:
                        cumsum += score
                        if cumsum >= r_val:
                            best_dy, best_dx = dy, dx
                            break
                else:
                    best_dy, best_dx, _, _, _ = candidates[self.rng.integers(0, len(candidates))]

                if nearby_predators and self.rng.random() < 0.9:
                    total_flee_dy, total_flee_dx = 0, 0
                    for p, dist in nearby_predators:
                        weight = 1.0 / max(1, dist)
                        total_flee_dy += -int(_sign(p.y - r.y)) * weight
                        total_flee_dx += -int(_sign(p.x - r.x)) * weight

                    target_dy = int(_sign(total_flee_dy))
                    target_dx = int(_sign(total_flee_dx))
                    for dy, dx, _, _, _ in candidates:
                        if dy == target_dy and dx == target_dx:
                            best_dy, best_dx = dy, dx
                            break
                elif neighbors:
                    target_dy = int(_sign(cohesion_dy)) if cohesion_dy != 0 else 0
                    target_dx = int(_sign(cohesion_dx)) if cohesion_dx != 0 else 0
                    if target_dy != 0 or target_dx != 0:
                        for dy, dx, _, _, _ in candidates:
                            if dy == target_dy and dx == target_dx:
                                best_dy, best_dx = dy, dx
                                break

            r.y = self._clamp(r.y + best_dy, 0, self.map_height - 1)
            r.x = self._clamp(r.x + best_dx, 0, self.map_width - 1)

    def _rabbit_eat(self, r: Rabbit) -> None:
        y, x = int(r.y), int(r.x)
        ratio = self.env.get_resource_ratio(y, x)
        if r.eating_state == "moving":
            efficiency = 0.2 * max(0.3, ratio)
            eaten = self.env.consume_resource(y, x, self.config.rabbit.energy_from_resource * efficiency)
            r.add_energy(eaten)
            if ratio > 0.5 and self.rng.random() < 0.3:
                r.eating_state = "grazing"
                r.grazing_timer = 3
        else:
            efficiency = 3.0 * max(0.3, ratio)
            eaten = self.env.consume_resource(y, x, self.config.rabbit.energy_from_resource * efficiency)
            r.add_energy(eaten)
            r.grazing_timer -= 1
            if r.grazing_timer <= 0:
                r.eating_state = "moving"

    def _update_rabbits(self) -> None:
        for r in self.rabbits:
            if not r.alive:
                continue
            r.step_energy(self.config.rabbit.energy_per_step)
            if r.alive:
                self._rabbit_move(r)
                self._rabbit_eat(r)

    def _find_prey(self, predator: Predator, prey_type: str) -> Optional[Entity]:
        vision = self.env.get_effective_vision_predator(int(predator.y), int(predator.x), predator.chase_radius)
        best_prey = None
        best_dist = float('inf')

        prey_list = self.grazers if prey_type == "grazer" else self.rabbits

        for prey in prey_list:
            if not prey.alive:
                continue
            dy = abs(int(prey.y) - int(predator.y))
            dx = abs(int(prey.x) - int(predator.x))
            dist = max(dy, dx)
            if dist <= vision and dist < best_dist:
                terrain = self.env.get_terrain_at(int(prey.y), int(prey.x))
                if self.env.can_see_through(int(predator.y), int(predator.x), int(prey.y), int(prey.x), True):
                    best_prey = prey
                    best_dist = dist

        return best_prey

    def _find_pack_target(self, predator: Predator) -> Optional[Entity]:
        vision = self.env.get_effective_vision_predator(int(predator.y), int(predator.x), predator.chase_radius)
        best_prey = None
        best_dist = float('inf')

        for grazer in self.grazers:
            if not grazer.alive:
                continue
            dy = abs(int(grazer.y) - int(predator.y))
            dx = abs(int(grazer.x) - int(predator.x))
            dist = max(dy, dx)
            if dist <= vision and dist < best_dist:
                if self.env.can_see_through(int(predator.y), int(predator.x), int(grazer.y), int(grazer.x), True):
                    best_prey = grazer
                    best_dist = dist

        for rabbit in self.rabbits:
            if not rabbit.alive:
                continue
            dy = abs(int(rabbit.y) - int(predator.y))
            dx = abs(int(rabbit.x) - int(predator.x))
            dist = max(dy, dx)
            if dist <= vision and dist < best_dist:
                if self.env.can_see_through(int(predator.y), int(predator.x), int(rabbit.y), int(rabbit.x), True):
                    best_prey = rabbit
                    best_dist = dist

        return best_prey

    def _predator_chase(self, predator: Predator, target: Entity) -> None:
        y, x = int(predator.y), int(predator.x)
        ty, tx = int(target.y), int(target.x)

        current_terrain = self.env.get_terrain_at(y, x)
        mobility = self.env.get_mobility_predator(current_terrain)

        best_dy, best_dx = 0, 0
        best_dist = float('inf')

        for _ in range(max(1, int(predator.move_speed * mobility))):
            candidates = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny = self._clamp(y + dy, 0, self.map_height - 1)
                    nx = self._clamp(x + dx, 0, self.map_width - 1)
                    if not self.env.is_passable(ny, nx):
                        continue
                    candidates.append((dy, dx, ny, nx))

            if not candidates:
                break

            scored = []
            for dy, dx, ny, nx in candidates:
                dist = max(abs(ny - ty), abs(nx - tx))
                scored.append((dy, dx, dist))

            scored.sort(key=lambda s: s[2])
            best_dy, best_dx, best_dist = scored[0]

            y, x = int(predator.y) + best_dy, int(predator.x) + best_dx
            predator.y = self._clamp(predator.y + best_dy, 0, self.map_height - 1)
            predator.x = self._clamp(predator.x + best_dx, 0, self.map_width - 1)

    def _predator_eat_carcass(self, predator: Predator) -> bool:
        y, x = int(predator.y), int(predator.x)
        for carcass in self.carcasses:
            if carcass.energy <= 0:
                continue
            dy = abs(int(carcass.y) - y)
            dx = abs(int(carcass.x) - x)
            if max(dy, dx) <= 1:
                eaten = carcass.take_energy(self.config.predator.energy_from_prey * 0.5)
                if eaten > 0:
                    predator.add_energy(eaten)
                    predator.eating_state = "eating_carcass"
                    predator.eating_timer = 2
                    return True
        return False

    def _predator_eat_prey(self, predator: Predator) -> bool:
        y, x = int(predator.y), int(predator.x)

        for grazer in self.grazers:
            if not grazer.alive:
                continue
            if int(grazer.y) == y and int(grazer.x) == x:
                grazer.alive = False
                energy_gained = min(self.config.predator.energy_from_prey, grazer.energy + grazer.fat)
                predator.add_energy(energy_gained)
                self.carcasses.append(Carcass(
                    x=float(x), y=float(y),
                    energy=grazer.energy * 0.5,
                    max_energy=grazer.energy * 0.5
                ))
                predator.eating_state = "eating_prey"
                predator.eating_timer = 3
                return True

        for rabbit in self.rabbits:
            if not rabbit.alive:
                continue
            if int(rabbit.y) == y and int(rabbit.x) == x:
                rabbit.alive = False
                energy_gained = min(self.config.predator.energy_from_prey, rabbit.energy)
                predator.add_energy(energy_gained)
                self.carcasses.append(Carcass(
                    x=float(x), y=float(y),
                    energy=rabbit.energy * 0.5,
                    max_energy=rabbit.energy * 0.5
                ))
                predator.eating_state = "eating_prey"
                predator.eating_timer = 3
                return True

        return False

    def _update_predators(self) -> None:
        for p in self.predators:
            if not p.alive:
                continue

            p.step_energy(self.config.predator.energy_per_step)
            if not p.alive:
                continue

            if p.eating_state == "eating_carcass":
                p.eating_timer -= 1
                if p.eating_timer <= 0:
                    p.eating_state = "moving"
                continue
            elif p.eating_state == "eating_prey":
                p.eating_timer -= 1
                if p.eating_timer <= 0:
                    p.eating_state = "moving"
                continue

            if self._predator_eat_carcass(p):
                continue

            if self._predator_eat_prey(p):
                continue

            target = self._find_pack_target(p)
            if target:
                self._predator_chase(p, target)
            else:
                self._predator_wander(p)

    def _predator_wander(self, predator: Predator) -> None:
        y, x = int(predator.y), int(predator.x)
        current_terrain = self.env.get_terrain_at(y, x)
        mobility = self.env.get_mobility_predator(current_terrain)

        best_dy, best_dx = 0, 0
        best_score = -1

        for _ in range(max(1, int(predator.move_speed * mobility))):
            candidates = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny = self._clamp(y + dy, 0, self.map_height - 1)
                    nx = self._clamp(x + dx, 0, self.map_width - 1)
                    if not self.env.is_passable(ny, nx):
                        continue
                    resource_ratio = self.env.get_resource_ratio(ny, nx)
                    candidates.append((dy, dx, resource_ratio, ny, nx))

            if not candidates:
                break

            scored = []
            for dy, dx, ratio, ny, nx in candidates:
                score = ratio
                dist_to_center = max(abs(ny - self.map_height // 2), abs(nx - self.map_width // 2))
                score -= dist_to_center * 0.01
                scored.append((dy, dx, score))

            scored.sort(key=lambda s: s[2], reverse=True)
            best_dy, best_dx, best_score = scored[0]

            y, x = int(predator.y) + best_dy, int(predator.x) + best_dx
            predator.y = self._clamp(predator.y + best_dy, 0, self.map_height - 1)
            predator.x = self._clamp(predator.x + best_dx, 0, self.map_width - 1)

    def _handle_reproduction(self) -> None:
        new_grazers = []
        for g in self.grazers:
            if not g.alive:
                continue
            if (g.age >= self.config.grazer.reproduction_min_age and
                g.energy >= self.config.grazer.reproduction_min_energy):
                cost = g.max_energy * self.config.grazer.reproduction_energy_cost
                if g.energy >= cost:
                    g.energy -= cost
                    offspring = Grazer(
                        x=g.x + self.rng.uniform(-1, 1),
                        y=g.y + self.rng.uniform(-1, 1),
                        energy=self.config.grazer.offspring_initial_energy,
                        max_energy=self.config.grazer.max_energy,
                        config=self.config.grazer,
                    )
                    offspring.gradient_weight = self.rng.normal(
                        self.config.grazer.gradient_weight, self.config.grazer.gradient_weight_std
                    )
                    offspring.gradient_weight = np.clip(offspring.gradient_weight, 0.1, 1.0)
                    new_grazers.append(offspring)

        new_rabbits = []
        for r in self.rabbits:
            if not r.alive:
                continue
            if (r.age >= self.config.rabbit.reproduction_min_age and
                r.energy >= self.config.rabbit.reproduction_min_energy):
                cost = r.max_energy * self.config.rabbit.reproduction_energy_cost
                if r.energy >= cost:
                    r.energy -= cost
                    offspring = Rabbit(
                        x=r.x + self.rng.uniform(-1, 1),
                        y=r.y + self.rng.uniform(-1, 1),
                        energy=self.config.rabbit.offspring_initial_energy,
                        max_energy=self.config.rabbit.max_energy,
                        config=self.config.rabbit,
                    )
                    offspring.gradient_weight = self.rng.normal(
                        self.config.rabbit.gradient_weight, self.config.rabbit.gradient_weight_std
                    )
                    offspring.gradient_weight = np.clip(offspring.gradient_weight, 0.1, 1.0)
                    new_rabbits.append(offspring)

        new_predators = []
        for p in self.predators:
            if not p.alive:
                continue
            if (p.age >= self.config.predator.reproduction_min_age and
                p.energy >= self.config.predator.reproduction_min_energy):
                cost = p.max_energy * self.config.predator.reproduction_energy_cost
                if p.energy >= cost:
                    p.energy -= cost
                    offspring = Predator(
                        x=p.x + self.rng.uniform(-1, 1),
                        y=p.y + self.rng.uniform(-1, 1),
                        energy=self.config.predator.offspring_initial_energy,
                        max_energy=self.config.predator.max_energy,
                        config=self.config.predator,
                    )
                    new_predators.append(offspring)

        self.grazers.extend(new_grazers)
        self.rabbits.extend(new_rabbits)
        self.predators.extend(new_predators)

    def _cleanup_dead(self) -> None:
        self.grazers = [g for g in self.grazers if g.alive]
        self.rabbits = [r for r in self.rabbits if r.alive]
        self.predators = [p for p in self.predators if p.alive]
        self.carcasses = [c for c in self.carcasses if c.can_provide_energy()]

    def _record_stats(self) -> SimulationStats:
        stats = SimulationStats()
        stats.step = self.current_step

        alive_grazers = [g for g in self.grazers if g.alive]
        alive_rabbits = [r for r in self.rabbits if r.alive]
        alive_predators = [p for p in self.predators if p.alive]

        stats.grazer_count = len(alive_grazers)
        stats.rabbit_count = len(alive_rabbits)
        stats.predator_count = len(alive_predators)
        stats.carcass_count = len([c for c in self.carcasses if c.can_provide_energy()])

        stats.total_grazer_energy = sum(g.energy for g in alive_grazers)
        stats.total_rabbit_energy = sum(r.energy for r in alive_rabbits)
        stats.total_predator_energy = sum(p.energy for p in alive_predators)

        stats.avg_grazer_energy = stats.total_grazer_energy / stats.grazer_count if stats.grazer_count > 0 else 0
        stats.avg_rabbit_energy = stats.total_rabbit_energy / stats.rabbit_count if stats.rabbit_count > 0 else 0
        stats.avg_predator_energy = stats.total_predator_energy / stats.predator_count if stats.predator_count > 0 else 0

        return stats

    def step(self) -> None:
        self.env.step()

        for c in self.carcasses:
            c.step()

        self._update_grazers()
        self._update_rabbits()
        self._update_predators()

        self._handle_reproduction()

        self._cleanup_dead()

        stats = self._record_stats()
        self.stats_history.append(stats)

        self.current_step += 1

    def run(self, steps: int | None = None) -> list[SimulationStats]:
        max_steps = steps if steps is not None else self.config.simulation.max_steps
        if max_steps <= 0:
            max_steps = float('inf')

        while self.current_step < max_steps:
            if not self.grazers and not self.rabbits and not self.predators:
                break
            self.step()

        return self.stats_history

    def get_state(self) -> dict:
        return {
            "step": self.current_step,
            "grazers": [
                {"id": g.id, "x": g.x, "y": g.y, "energy": g.energy, "fat": g.fat, "alive": g.alive}
                for g in self.grazers if g.alive
            ],
            "rabbits": [
                {"id": r.id, "x": r.x, "y": r.y, "energy": r.energy, "eating_state": r.eating_state, "alive": r.alive}
                for r in self.rabbits if r.alive
            ],
            "predators": [
                {"id": p.id, "x": p.x, "y": p.y, "energy": p.energy, "eating_state": p.eating_state, "alive": p.alive}
                for p in self.predators if p.alive
            ],
            "carcasses": [
                {"id": c.id, "x": c.x, "y": c.y, "energy": c.energy}
                for c in self.carcasses if c.can_provide_energy()
            ],
            "resources": self.env.resources.tolist(),
            "terrain": self.env.terrain.tolist(),
        }

    def get_statistics(self) -> dict:
        """Get simulation statistics for UI display."""
        return {
            "grazers": len([g for g in self.grazers if g.alive]),
            "rabbits": len([r for r in self.rabbits if r.alive]),
            "predators": len([p for p in self.predators if p.alive]),
            "total_grass": float(np.sum(self.env.resources)),
            "grazers_list": [g for g in self.grazers if g.alive],
            "rabbits_list": [r for r in self.rabbits if r.alive],
            "predators_list": [p for p in self.predators if p.alive],
        }
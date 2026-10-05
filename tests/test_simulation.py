"""Regression tests for behaviour that was previously broken.

Each test here corresponds to a defect found while reconciling the implementation
with SPECIFICATION.md / USER_STORIES.md.
"""

import numpy as np
import pytest

from simulator.config import load_config
from simulator.environment import Environment
from simulator.entities import Grazer, Prey, Predator
from simulator.simulation import Simulation


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


@pytest.fixture
def sim(config):
    return Simulation(config, seed=7)


def clear_agents(sim):
    sim.grazers.clear()
    sim.rabbits.clear()
    sim.predators.clear()
    sim.carcasses.clear()


def place_plain(sim, y, x):
    sim.env.set_terrain(y, x, "plains")
    sim.env.refresh_terrain_properties()


# --------------------------------------------------------------------- config


def test_config_exposes_spec_parameters(config):
    assert config.environment.ocean_border_width == 5
    # SPECIFICATION.md terrain table: rock is passable slow terrain.
    assert config.environment.terrain_types["rock"].passable is True
    assert config.grazer.herd_radius == 8
    assert config.grazer.reproduction_energy_cost == pytest.approx(0.45)
    assert config.carcass.feed_radius == 1


# ----------------------------------------------------------------- environment


def test_ocean_border_is_impassable_and_empty(config):
    env = Environment(config.environment, 60, 60, seed=3)
    border = env.ocean_border
    assert not env.is_passable(0, 0)
    assert not env.is_passable(border - 1, 30)
    assert env.get_resource_at(0, 0) == 0.0
    assert env.get_resource_at(2, 2) == 0.0
    assert env.is_passable(border + 5, 30)


def test_terrain_generation_uses_all_types(config):
    env = Environment(config.environment, 80, 80, seed=5)
    assert set(env.terrain_names) == {"ocean", "rock", "plains", "forest"}
    # Rock must not be a solid mass; passable interior must exist.
    assert env.passable[env.terrain == env._terrain_index["forest"]].any()


def test_regeneration_slows_but_never_stops(config):
    """Suppression is floored at min_regen_suppression, so a stripped world recovers."""
    floor = config.environment.min_regen_suppression
    assert floor == pytest.approx(0.25)
    env = Environment(config.environment, 40, 40, seed=1)
    env.resources[:] = 0.0
    before = float(env.resources.sum())
    for _ in range(400):
        env.step()
    assert float(env.resources.sum()) > before, "regeneration halted on a bare map"


def test_depletion_signal_tracks_fresh_grazing(config):
    env = Environment(config.environment, 40, 40, seed=1)
    y, x = 20, 20
    place_plain_via_env(env, y, x)
    capacity = float(env.resource_capacity[y, x])
    env.resources[y, x] = capacity
    env._snapshot_resources()
    env.consume_resource(y, x, capacity * 0.5)
    env.step()
    assert env.depletion_rate[y, x] > 0.0

    # Once grazing stops, regeneration refills the cell and the "fresh change"
    # signal decays rather than latching on permanently.
    peak = float(env.depletion_rate[y, x])
    for _ in range(80):
        env.step()
    assert float(env.depletion_rate[y, x]) < peak
    assert env.get_resource_ratio(y, x) > float(env.depletion_rate[y, x])


def place_plain_via_env(env, y, x):
    env.set_terrain(y, x, "plains")


# ----------------------------------------------------------------- consumption


def test_prey_consumption_actually_depletes_resources(config, sim):
    """Grazers previously called eat_while_moving() and never touched the grid."""
    clear_agents(sim)
    place_plain(sim, 50, 50)
    sim.env.resources[50, 50] = 90.0
    before = float(sim.env.resources[50, 50])

    grazer = Grazer(x=50.0, y=50.0, energy=500.0, config=config.grazer)
    grazer.eat(sim.env)

    assert sim.env.resources[50, 50] < before, "grazing did not remove any resource"
    assert grazer.energy > 500.0


def test_efficiency_has_a_floor_but_still_outlives_metabolism(config):
    """Worst-case intake must exceed metabolic cost or prey can never recover."""
    for name in ("grazer", "rabbit"):
        cfg = getattr(config, name)
        floor_gain = cfg.energy_from_resource * cfg.min_efficiency
        assert floor_gain > cfg.energy_per_step, f"{name} starves in depleted terrain"


# ------------------------------------------------------------------- predation


def test_kill_creates_carcass_and_transfers_no_energy(config, sim):
    clear_agents(sim)
    place_plain(sim, 60, 60)
    prey = Grazer(x=60.0, y=60.0, energy=400.0, config=config.grazer)
    sim.grazers.append(prey)
    killer = Predator(x=60.0, y=60.0, energy=0.0, config=config.predator)
    sim.predators.append(killer)

    assert sim._predator_kill_prey(killer) is True
    assert prey.alive is False
    assert killer.energy == 0.0, "kill must not pay out energy directly"
    assert len(sim.carcasses) == 1
    assert sim.carcasses[0].energy == pytest.approx(
        400.0 * config.carcass.carcass_energy_fraction
    )


def test_adjacent_predators_share_one_carcass_store(config, sim):
    clear_agents(sim)
    place_plain(sim, 70, 70)
    sim.grazers.append(Grazer(x=70.0, y=70.0, energy=400.0, config=config.grazer))
    predators = []
    for dy, dx in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1)]:
        predators.append(
            Predator(x=float(70 + dx), y=float(70 + dy), energy=0.0, config=config.predator)
        )
    sim.predators.extend(predators)
    sim._build_indices()

    sim._predator_kill_prey(predators[0])
    carcass = sim.carcasses[0]
    start = carcass.energy

    sim._feed_on_carcasses()

    total = sum(p.energy for p in predators)
    assert total == pytest.approx(start - carcass.energy), "feeders did not share one store"
    assert carcass.energy < start
    assert all(p.energy > 0 for p in predators), "adjacent predators could not feed"


def test_more_predators_strip_a_carcass_faster(config, sim):
    def passes(feeders):
        clear_agents(sim)
        place_plain(sim, 80, 80)
        sim.grazers.append(Grazer(x=80.0, y=80.0, energy=400.0, config=config.grazer))
        spots = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        pack = [
            Predator(x=float(80 + dx), y=float(80 + dy), energy=0.0, config=config.predator)
            for dy, dx in spots[:feeders]
        ]
        sim.predators.extend(pack)
        sim._build_indices()
        sim._predator_kill_prey(pack[0])
        carcass = sim.carcasses[0]
        count = 0
        while carcass.provides_energy and count < 500:
            sim._feed_on_carcasses()
            count += 1
        return count

    assert passes(8) < passes(4) < passes(2) < passes(1)


def test_carcass_feeding_is_oldest_first(config, sim):
    clear_agents(sim)
    place_plain(sim, 90, 90)
    sim.grazers.append(Grazer(x=90.0, y=90.0, energy=40.0, config=config.grazer))
    pack = []
    for idx, (dy, dx) in enumerate([(0, 0), (-1, -1), (-1, 0)]):
        predator = Predator(
            x=float(90 + dx), y=float(90 + dy), energy=0.0, config=config.predator
        )
        predator.age = 100 - idx
        pack.append(predator)
    sim.predators.extend(pack)
    sim._build_indices()
    sim._predator_kill_prey(pack[0])

    sim._feed_on_carcasses()

    # Scarce carcass: the oldest takes the full draw, the next the remainder.
    assert pack[0].energy == pytest.approx(
        config.predator.energy_from_prey / config.carcass.consumption_divisor
    )
    assert pack[1].energy > 0.0
    assert pack[2].energy == 0.0


def test_predator_closes_and_kills_in_one_step(config, sim):
    """The kill check must run after movement or a 2-cell chase sweeps past prey."""
    clear_agents(sim)
    place_plain(sim, 100, 100)
    prey = Grazer(x=101.0, y=100.0, energy=400.0, config=config.grazer)
    sim.grazers.append(prey)
    predator = Predator(x=100.0, y=100.0, energy=0.0, config=config.predator)
    sim.predators.append(predator)
    sim._build_indices()

    sim._predator_behaviour(predator)
    assert sim._predator_kill_prey(predator) is True
    assert prey.alive is False


# ----------------------------------------------------------------- mechanics


def test_bounded_chebyshev_distance(sim):
    height = sim.map_height
    width = sim.map_width
    assert sim.distance(0, 0, 0, 3) == 3
    assert sim.distance(0, 0, 4, 4) == 4          # diagonal is not 5.66
    assert sim.distance(0, 0, 0, width - 2) == 2  # wraps at the seam
    assert sim.distance(0, 0, height - 1, 0) == 1


def test_spatial_index_matches_brute_force(sim):
    sim._build_indices()
    radius = sim.config.grazer.herd_radius
    target = sim.grazers[0]
    ty, tx = int(target.y), int(target.x)

    indexed = {
        id(other)
        for other in sim._prey_index.query(ty - radius, ty + radius, tx - radius, tx + radius)
        if other is not target
    }
    brute = {
        id(other)
        for other in (sim.grazers + sim.rabbits)
        if other is not target
        and max(abs(int(other.y) - ty), abs(int(other.x) - tx)) <= radius
    }
    assert indexed == brute


def test_reproduction_cost_is_fraction_of_current_energy(config, sim):
    clear_agents(sim)
    parent = Grazer(x=50.0, y=50.0, energy=800.0, config=config.grazer)
    parent.age = config.grazer.reproduction_min_age
    child = sim._try_reproduce(parent, config.grazer)
    assert child is not None
    assert parent.energy == pytest.approx(
        800.0 * (1.0 - config.grazer.reproduction_energy_cost)
    )
    # Not a flat charge off max_energy.
    assert parent.energy != pytest.approx(
        800.0 - config.grazer.max_energy * config.grazer.reproduction_energy_cost
    )


def test_stats_track_cumulative_births_and_deaths(sim):
    # Long enough for the youngest reproduction age (rabbit, 35) to elapse.
    for _ in range(90):
        sim.step()
    stats = sim.stats_history[-1]
    assert stats.step == 89
    assert stats.deaths_grazer + stats.deaths_rabbit + stats.deaths_predator > 0
    assert stats.births_grazer + stats.births_rabbit + stats.births_predator > 0
    # Cumulative counters must never decrease.
    assert [s.deaths_grazer for s in sim.stats_history] == sorted(
        s.deaths_grazer for s in sim.stats_history
    )


def test_reset_preserves_environment(sim):
    terrain_before = sim.env.terrain.copy()
    sim.step()
    sim.reset()
    assert np.array_equal(sim.env.terrain, terrain_before)
    assert sim.current_step == 0
    assert sim.counters.deaths_grazer == 0


def test_timestep_is_exposed_and_positive(config):
    assert config.simulation.timestep_ms > 0
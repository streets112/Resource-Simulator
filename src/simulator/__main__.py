import argparse
import json
from pathlib import Path

from simulator.config import Config, load_config
from simulator.simulation import Simulation

# Cap on catch-up steps per frame so a slow frame cannot cascade into a stall.
MAX_STEPS_PER_FRAME = 10


def run_headless(config: Config, steps: int | None, export_path: str | None, seed: int | None) -> None:
    sim = Simulation(config, seed=seed)
    sim.run(steps)

    if export_path:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump([stats.as_record() for stats in sim.stats_history], handle, indent=2)
        print(f"Exported {len(sim.stats_history)} steps to {export_path}")

    if sim.ended:
        print(sim.extinction_summary())
    else:
        final = sim.stats_history[-1] if sim.stats_history else None
        if final:
            print(
                f"Reached step {sim.current_step}: "
                f"grazers={final.grazer_count} rabbits={final.rabbit_count} "
                f"predators={final.predator_count}"
            )


def run_gui(config: Config, seed: int | None) -> None:
    try:
        from simulator.render import MainRenderer
    except ImportError as error:
        print(f"GUI not available: {error}")
        print("Install pygame first: pip install pygame-ce")
        return

    sim = Simulation(config, seed=seed)
    renderer = MainRenderer(config)
    renderer.set_simulation(sim)

    accumulator = 0.0

    while True:
        if renderer.handle_events(sim) == "quit":
            break

        elapsed_ms = renderer.tick()
        paused = renderer.paused

        if not paused and not sim.ended:
            interval_ms = max(1, config.simulation.timestep_ms) / max(
                1e-6, config.simulation.simulation_speed
            )
            accumulator += elapsed_ms
            taken = 0
            while accumulator >= interval_ms and taken < MAX_STEPS_PER_FRAME:
                sim.step()
                accumulator -= interval_ms
                taken += 1
            if accumulator > interval_ms * MAX_STEPS_PER_FRAME:
                accumulator = 0.0
            if sim.ended:
                print(sim.extinction_summary())

        renderer.draw(sim, paused)

    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource Simulator")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 = config max)")
    parser.add_argument("--export", help="Export per-step statistics to JSON")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible runs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps:
        config.simulation.max_steps = args.steps

    if args.headless:
        run_headless(config, args.steps or None, args.export, args.seed)
    else:
        run_gui(config, args.seed)


if __name__ == "__main__":
    main()
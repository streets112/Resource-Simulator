import argparse
import json
from pathlib import Path
from simulator.config import load_config, save_config, Config
from simulator.simulation import Simulation
from simulator.render import Renderer


def run_headless(config: Config, max_steps: int | None, export_path: str | None, export_interval: int) -> None:
    sim = Simulation(config)
    history = sim.run(max_steps)

    if export_path:
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "step": int(s.step),
                "grazer_count": int(s.grazer_count),
                "predator_count": int(s.predator_count),
"carcass_count": int(s.carcass_count),
            "total_grazer_energy": float(s.total_grazer_energy),
            "total_rabbit_energy": float(s.total_rabbit_energy),
            "total_predator_energy": float(s.total_predator_energy),
            "avg_grazer_energy": float(s.avg_grazer_energy),
            "avg_rabbit_energy": float(s.avg_rabbit_energy),
            "avg_predator_energy": float(s.avg_predator_energy),
            "births_grazer": int(s.grazer_births),
            "births_rabbit": int(s.rabbit_births),
            "births_predator": int(s.predator_births),
            "deaths_grazer": int(s.grazer_deaths),
            "deaths_rabbit": int(s.rabbit_deaths),
            "deaths_predator": int(s.predator_deaths),
            }
            for s in history
        ]
        with open(export_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} steps to {export_path}")

    alive_grazers = len([g for g in sim.grazers if g.alive])
    alive_rabbits = len([r for r in sim.rabbits if r.alive])
    alive_predators = len([p for p in sim.predators if p.alive])
    if alive_grazers == 0 and alive_rabbits == 0 and alive_predators == 0:
        print("\n=== SIMULATION ENDED (all species extinct) ===")
        print(f"Final Step: {sim.current_step}")
        if sim.stats_history:
            final_stats = sim.stats_history[-1]
            print(f"Total Grazer Births: {final_stats.grazer_births}")
            print(f"Total Rabbit Births: {final_stats.rabbit_births}")
            print(f"Total Predator Births: {final_stats.predator_births}")
            print(f"Total Grazer Deaths: {final_stats.grazer_deaths}")
            print(f"Total Rabbit Deaths: {final_stats.rabbit_deaths}")
            print(f"Total Predator Deaths: {final_stats.predator_deaths}")


def run_gui(config: Config) -> None:
    try:
        from simulator.render import MainRenderer
    except ImportError as e:
        print(f"GUI not available: {e}")
        print("Install pygame for visualization: pip install pygame")
        return

    sim = Simulation(config)
    renderer = MainRenderer(config.visualization, config.simulation.width, config.simulation.height)
    renderer.set_simulation(sim, config)

    running = True
    paused = False
    ended = False

    while running:
        result = renderer.handle_events()
        if result is False:
            running = False
            break
        elif result == "pause":
            paused = not paused
        elif result == "reset":
            sim = Simulation(config)
            renderer.set_simulation(sim, config)
            ended = False
        elif result == "reload_env":
            # Keep same entities, reload environment
            old_grazers = sim.grazers
            old_predators = sim.predators
            sim = Simulation(config)
            sim.grazers = old_grazers
            sim.predators = old_predators
            renderer.set_simulation(sim, config)
            ended = False

        if not paused and not ended:
            sim.step()
            renderer.dirty = True

            alive_grazers = len([g for g in sim.grazers if g.alive])
            alive_predators = len([p for p in sim.predators if p.alive])
            if alive_grazers == 0 and alive_predators == 0:
                ended = True
                sim.ended = True
                print("\n=== SIMULATION ENDED ===")
                print(f"Step: {sim.step_count}")
                print(f"Grazers: 0")
                print(f"Predators: 0")
                print(f"Total Grazer Births: {sim.stats.births_grazer}")
                print(f"Total Predator Births: {sim.stats.births_predator}")
                print(f"Total Grazer Deaths: {sim.stats.deaths_grazer}")
                print(f"Total Predator Deaths: {sim.stats.deaths_predator}")
                print(f"Max Grazer Population: {max(s.grazer_count for s in sim.stats_history) if sim.stats_history else 0}")
                print(f"Max Predator Population: {max(s.predator_count for s in sim.stats_history) if sim.stats_history else 0}")

        renderer.draw(sim, paused)
        renderer.update_config_window()
        renderer.tick()

    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource Simulator")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 = infinite)")
    parser.add_argument("--export", help="Export data to JSON file")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--export-interval", type=int, default=1000, help="Export interval")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config.simulation.max_steps = args.steps or config.simulation.max_steps

    if args.headless:
        run_headless(config, args.steps or None, args.export, args.export_interval)
    else:
        run_gui(config)


if __name__ == "__main__":
    main()
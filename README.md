# Resource Simulator

A 2D grid-based ecosystem simulator modelling resource-predator-prey dynamics on a
procedurally generated landscape.

- **Resources** regenerate per terrain type and diffuse between cells
- **Grazers / rabbits** consume resources, herd, flee, reproduce and starve
- **Predators** hunt solo or in packs, scavenge carcasses, and infer prey presence
  from freshly depleted ground

## Requirements

Python 3.10+, and:

```bash
pip install numpy scipy pyyaml pygame-ce
```

`pygame-ce` is required rather than `pygame`: the upstream `pygame` project ships no
wheels for Python 3.13+ and its sdist fails to build, because `distutils` was removed
from the standard library.

## Install

```bash
pip install -e .
```

The project uses a `src/` layout, so an editable install (or `PYTHONPATH=src`) is
required before `python -m simulator` will resolve.

## Running

Interactive GUI:

```bash
python -m simulator
```

Headless, with per-step statistics exported to JSON:

```bash
python -m simulator --headless --steps 1000 --export output/run.json --seed 42
```

Options: `--config`, `--headless`, `--steps`, `--export`, `--seed`.

### Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `Esc` | Quit |
| `R` | Reset populations, keep terrain and resources |
| `E` | Regenerate terrain, keep entities and statistics |
| `0` | Reset zoom and pan |
| Mouse wheel / middle drag | Zoom (cursor-anchored) / pan |
| `Ctrl` `+` / `Ctrl` `-` | Keyboard zoom |
| `F1`-`F4` | Toggle resources / grazers / predators / carcasses |
| `G` / `P` / `T` | Same layer toggles, mnemonically |
| `F5` | Toggle the population chart |
| `M` | Creator mode |
| `1` `2` `3` | Creator: rock / plains / forest brush |
| `+` / `-` | Creator: brush size |
| `Ctrl`+click | Creator: spawn grazer |
| `Shift`+click | Creator: spawn predator |

The HUD shows smoothed FPS and simulation steps/second, which are decoupled: the
simulation advances on a `timestep_ms` accumulator, so the step rate can differ
substantially from the frame rate.

## Architecture

| Module | Responsibility |
|--------|----------------|
| `config.py` | Dataclasses loaded from `config.yaml`; unknown YAML keys are ignored so configs stay forward/backward compatible |
| `environment.py` | Perlin/ridge terrain generation, resource regeneration with neighbour-depletion suppression, diffusion, consumption, depletion signal |
| `entities.py` | `Entity`, `Prey`, `Grazer`, `Rabbit`, `Predator`, `Carcass` |
| `simulation.py` | The step orchestrator: boids, foraging, fleeing, four predator modes, scavenging, reproduction, statistics |
| `render.py` | pygame presentation: cached world layer, entity layers, HUD, chart, creator mode |

### Simulation order

```
environment.step()      # regeneration + diffusion
carcass.step()          # decay
grazers / rabbits       # metabolism -> herd/flee/forage move -> eat
predators               # metabolism -> scavenge -> move -> kill
reproduce()
cleanup()
record statistics
extinction check
```

Predator kills and feeding are deliberately separate: a kill deposits a carcass and
transfers no energy directly, so the energy is only recoverable by scavenging on a
later step. All predators within `feed_radius` of a carcass draw on that one shared
store, so a carcass with more predators around it is stripped proportionally faster;
when the store runs out mid-pass, older predators feed first.

## Design notes and known gaps

This implementation was reconciled against `requirements-and-specifications/`. Where
the documents conflicted, the resolution was:

- **Rock is passable.** `SPECIFICATION.md`'s terrain table says "No", but
  `USER_STORIES.md`'s table and the shipped config both say yes. Passable was chosen so
  that rock's mobility multipliers (0.5 prey / 0.3 predator) mean "slow" rather than
  "absolute wall", and so the generated mountain passes matter.
- **Reproduction cost is a fraction of current energy**, per `SPECIFICATION.md`, not of
  `max_energy` as the old code did.
- **Separation points away from a neighbour.** The formula in `SPECIFICATION.md:102`
  has the sign inverted relative to its own stated intent, "avoid crowding".
- **Bounded Chebyshev distance** is used for predator perception radii. Herd vectors
  use plain grid deltas, because a toroidal metric would report herd members as
  adjacent across the impassable ocean border.
- **Regeneration suppression is floored** at `min_regen_suppression` (0.25) rather than
  allowed to reach zero, so a stripped map slows instead of entering a permanent dead
  state.

Not implemented, despite being described in the documents:

- Inspect mode (`I`), follow mode (`F`), and the keyboard-driven config window (`C`).
- Predators cannot pass vision or sense obstacles; line of sight is not modelled, only
  radius.
- `timestep_ms` is honoured by the GUI accumulator but is not a discrete simulation
  parameter: nothing is normalised by it, so changing it changes how fast the same
  step sequence is emitted, not the dynamics.
- Rabbits exist in `config.yaml` but appear in neither specification document, so their
  behaviour follows the grazer pattern rather than a documented spec.

**Population balance is not yet calibrated.** The structural defects are fixed and the
suite in `tests/` locks them in, but predator recruitment currently outpaces prey
reproduction over a few hundred steps. Treat the current numbers in `config.yaml` as a
starting point rather than a converged equilibrium.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_simulation.py` covers regressions for each defect found during the rewrite:
prey that consumed without depleting the grid, prey unable to out-earn metabolism in
depleted terrain, predators sweeping past prey during a chase, a kill that paid out
energy twice, regeneration that could halt permanently, and cumulative statistics that
were never populated.
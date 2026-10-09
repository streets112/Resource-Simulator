# Resource Simulator - Requirements and Specifications

## Overview
A 2D grid-based ecosystem simulator modeling resource-predator-prey dynamics with:
- **Resources**: Environmental food sources that regenerate based on terrain type
- **Grazers (Prey)**: Consume resources, reproduce, die from starvation
- **Predators**: Hunt grazers, reproduce, die from starvation

---

## Environment System

### Grid Structure
- **Dimensions**: 200×200 cells (configurable)
- **Coordinate System**: (x, y) with origin at top-left, bounded by ocean borders (no wrapping)
- **Cell Size**: 8 pixels (configurable)

### Terrain Types
 
| Terrain | Color (RGB) | Regen Rate | Capacity | Passable | Mobility (Prey/Pred) | Visibility (Prey/Pred) | Grazer Prod | Rabbit Prod |
|---------|-------------|------------|----------|----------|----------------------|------------------------|-------------|-------------|
| Ocean   | (30, 60, 180) | 0.0        | 0        | No       | 0.0/0.0              | 0.0/0.0                | 1.0         | 1.0         |
| Rock    | (80, 80, 80) | 0.0        | 0        | Yes      | 0.5/0.3              | 1.0/1.0                | 1.0         | 1.0         |
| Plains  | (180, 200, 100) | 0.6      | 110      | Yes      | 1.0/1.0              | 1.0/1.0                | 1.0         | 0.75        |
| Forest  | (60, 140, 60) | 0.10       | 160      | Yes      | 0.6/0.7              | 0.4/0.6                | 0.7         | 1.1         |
| Meadow  | (100, 200, 90) | 0.5      | 150      | Yes      | 1.0/1.0              | 1.0/1.0                | 1.3         | 0.5         |
| Scrub   | (120, 125, 65) | 0.25     | 130      | Yes      | 0.8/0.9              | 0.7/0.8                | 0.5         | 1.3         |
 
**Terrain Characteristics:**
- **Plains**: Fast regeneration, low capacity — grazers at home, rabbits exposed
- **Forest**: Slow regeneration, high capacity — cover favors rabbits
- **Meadow**: Grazer stronghold — fast regen, high capacity, grazers +30% surplus
- **Scrub**: Rabbit stronghold — cover with rich pickings for rabbits, poor for grazers
- **Rock**: Slow passable terrain, no resources — forms mountain ridges
- **Ocean**: Impassable border — 5 cells wide around the map

### Biome Generation (Procedural)
Uses multi-octave Perlin noise with ridge noise for mountains:
1. **Mountain Ridges**: Ridge noise `(1 - |perlin|)²` with directional bias creates long connected ridges
2. **Natural Passes**: Carved at thinnest points of large ridge systems (>200 cells) using distance transform
3. **Lowland Allocation**: Non-mountain areas split among Forest, Meadow, Scrub, Plains via sequential quantile allocation on per-biome noise fields, honoring `terrain_distribution` weights
4. **Organic Expansion**: Forests expand organically into adjacent ground (3 iterations)
5. **Ocean Border**: 5-cell wide impassable border around the entire map

### Resource Dynamics

#### Resource Regeneration (per timestep)
```
deficit(x, y) = capacity(x, y) - resources(x, y)
base_regen(x, y) = regen_rate(x, y) × deficit(x, y) × 0.025
suppression(x, y) = clip(1.0 - neighbor_depletion(x, y) × 2.2, 0.0, 1.0)
resources(x, y) ← clip(resources(x, y) + base_regen × suppression, 0, capacity(x, y))
```

Neighbor depletion suppresses regrowth when surrounding cells are below 60% capacity.

#### Resource Diffusion (per timestep)
```
diffusion_kernel = D × [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
diff(x, y) = Σ diffusion_kernel[i,j] × resources(x+i, y+j)
resources(x, y) ← clip(resources(x, y) + diff(x, y), 0, capacity(x, y))
```
Where D = 0.1 (configurable `resource_diffusion_rate`)

#### Resource Consumption
```
ratio = resources(y, x) / capacity(y, x)
efficiency = max(0.3, ratio)
eaten = consume_resource(y, x, energy_from_resource × efficiency)
energy ← min(max_energy, energy + eaten)
```
Grazers eat less efficiently in depleted cells.

---

## Grazer (Prey) Specifications

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Count | 200 | Starting population |
| Max Energy | 1000 | Energy storage cap |
| Energy/Step | 10 | Metabolic cost per timestep |
| Energy/Resource | 18 | Energy gained per resource unit consumed |
| Reproduction Min Age | 55 | Minimum age (timesteps) to reproduce |
| Reproduction Min Energy | 650 | Minimum energy to reproduce |
| Reproduction Cost | 0.45 | Fraction of energy transferred to offspring |
| Offspring Initial Energy | 200 | Starting energy for newborn |
| Move Speed | 1 | Cells per timestep |
| Gradient Weight | 0.7 | Bias toward gradient vs random (0-1) |
| Gradient Weight Std | 0.2 | Individual variation in gradient weight |
| Vision Radius | 5 | Cells to scan for resources |

### Herd Behavior (Boids-style)
Each grazer detects neighbors within 8 cells and computes:

**Cohesion**: Move toward center of mass of nearby herd members
**Alignment**: Match average movement direction of neighbors
**Separation**: Avoid crowding within 2 cells

```
for each neighbor within herd_radius:
    cohesion += neighbor_position - my_position
    alignment += neighbor.velocity
    if distance ≤ separation_radius:
        separation -= (my_position - neighbor_position) * (separation_radius - distance + 1)

movement_score = cohesion × 0.4 + alignment × 0.3 + separation × 0.3
```

70% chance to follow herd direction when neighbors present, otherwise gradient/random walk.

### Feeding Behavior
```
ratio = get_resource_ratio(y, x)
efficiency = max(0.3, ratio)
eaten = consume_resource(y, x, energy_from_resource × efficiency)
energy ← min(max_energy, energy + eaten)
```

### Energy Update (per timestep)
```
energy ← energy - energy_per_step
age ← age + 1
if energy ≤ 0: die
```

### Reproduction Conditions
```
can_reproduce = (age ≥ reproduction_min_age) AND (energy ≥ reproduction_min_energy)
energy_cost = energy × reproduction_energy_cost
energy ← energy - energy_cost
offspring inherits gradient_weight ~ N(parent.gradient_weight, gradient_weight_std) clipped to [0,1]
```

---

## Predator Specifications

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Count | 30 | Starting population |
| Max Energy | 800 | Energy storage cap |
| Energy/Step | 3 | Metabolic cost per timestep |
| Energy/Prey | 700 | Energy gained from consuming grazer |
| Reproduction Min Age | 65 | Minimum age (timesteps) to reproduce |
| Reproduction Min Energy | 250 | Minimum energy to reproduce |
| Reproduction Cost | 0.3 | Fraction of energy transferred to offspring |
| Offspring Initial Energy | 200 | Starting energy for newborn |
| Move Speed | 2 | Cells per timestep (2× grazer speed) |
| Chase Radius | 7 | Detection range for direct pursuit |
| Resource Sense Radius | 10 | Range to detect depleted resource patches |

### Behavior States

#### 1. Pack Hunting Mode
When ≥1 predator within 10 cells, 60% chance to coordinate:
- Target grazers with most nearby predators (flanking)
- Flank from direction of pack center (40% chance per step)

#### 2. Solo Pursuit Mode
Direct chase of closest grazer within chase_radius (Chebyshev distance)

#### 3. Resource Sensing Mode
Scan for depleted resource patches (<30% capacity) within resource_sense_radius

#### 4. Pack Wander Mode
When no targets, move toward center of nearby predator group (≤10 cells) with 50% probability

### Energy Update (per timestep)
```
energy ← energy - energy_per_step
age ← age + 1
if energy ≤ 0: die
```

### Reproduction
```
can_reproduce = (age ≥ reproduction_min_age) AND (energy ≥ reproduction_min_energy)
energy_cost = energy × reproduction_energy_cost
offspring at parent position with offspring_initial_energy
```

---

## Simulation Loop

```
while running:
    environment.step()          # Regenerate + diffuse resources
    for each grazer:
        grazer.step_energy()
        if alive: grazer.move() with herd behavior; grazer.eat()
    for each predator:
        predator.step_energy()
        if alive: predator.update_behavior() with pack behavior
    handle_reproduction()
    cleanup_dead()
    record_statistics()
    check_extinction()
```

Extinction check: if both populations reach 0, simulation ends with summary statistics.

### Timestep
- Default: 100ms per step (10 steps/second)
- Configurable via `timestep_ms`

---

## Configuration System

All parameters exposed in `config.yaml` with live-reload via **Config Window** (keyboard-driven):
- Simulation parameters (grid size, timestep)
- Terrain types and biome generation weights
- Resource dynamics (regen, diffusion, capacity per terrain)
- Grazer parameters (physiological + herd behavior weights)
- Predator parameters (physiological + pack behavior weights)
- Visualization settings (colors, cell size, FPS)
- Output settings (headless mode, export intervals)

---

## Visualization System

### 4×4 Per-Cell Rendering
Each simulation cell renders as 4×4 subpixels (8×8 pixels total at cell_size=8):

| Quadrant | Shows |
|----------|-------|
| Top-Left | Terrain type (rock=gray, plains=yellow-green, meadow=bright green, scrub=olive, forest=dark green) |
| Top-Right | Resource level (dark gray→bright green gradient) |
| Bottom-Left | Predator presence (orange) / Carcass (red) |
| Bottom-Right | Grazer presence (yellow) |

### Controls
| Key | Action |
|-----|--------|
| **Space** | Pause/Resume |
| **Esc** | Quit |
| **G** | Toggle grazers visibility |
| **P** | Toggle predators visibility |
| **T** | Toggle carcasses visibility |
| **R** | Toggle resources visibility |
| **C** | Open Config Window (keyboard-driven) |
| **R** | Reset simulation (keep environment) |
| **E** | Reload environment (keep entities) |
| **M** | Toggle Creator Mode |
| **I** | Toggle Inspect Mode |
| **F** | Toggle Follow Mode |
| **0** | Reset zoom/pan |
| **F1-F4** | Toggle layers (Resources, Grazers, Predators, Carcasses) |
| **Ctrl+=/-** | Zoom in/out |
| **Middle drag** | Pan |

### Config Window (C → G/P/T/S/M)
Keyboard-driven hierarchical menu:
1. Press **C** → Main menu with 4 categories
2. Press **G/P/T/S** → Grazer/Predator/Terrain/Simulation settings
3. Adjust sliders with mouse
4. Press **M** → Return to main menu
5. Press **ESC** → Close (from main) / Back to menu (from sub)

### Inspect Mode (I)
Keyboard-driven inspection:
- Press **I** → Enter inspect mode at current mouse position (or map center)
- **Arrow keys** (← → ↑ ↓) → Move inspection box 1 cell per press
- **5×5 grid** displayed at top of tooltip, details at bottom
- Yellow box rendered around 5×5 inspection region in world view
- **I** again → Exit inspect mode
### Follow Mode (F)

**As a researcher, I want to track individual entities with live stats, so that I can study individual life histories.**

**Acceptance Criteria:**
- F key toggles follow mode
- Click entity to select: camera centers on it, white circle drawn around it, live stats pinned in inspect pane (position, energy/max, age, ID, species-specific params, lineage)
- Click bare ground: stop following
- Esc key: stop following
- Middle-mouse pan: stop following
- Auto-disables if entity dies

---

## Creator Mode (M)
Interactive world editing:
| Control | Action |
|---------|--------|
| **1-5** | Select brush: rock/plains/meadow/scrub/forest (hold while dragging to paint) |
| **Press 1-5** | Show brush name hint at top (auto-hides after 1.5s) |
| **Hold 1-5 + drag** | Paint terrain with that brush |
| **+ / -** | Brush size 1-10 (circular) |
| **G + Left click** | Spawn grazer at cell |
| **P + Left click** | Spawn predator at cell |
| **Alt + Left click** | Spawn rabbit at cell |
| **M** | Toggle creator mode (clears held brush keys) |

Terrain properties (capacity, regen rate, passable, mobility, visibility, per-species productivity) update automatically.

---

## Headless Mode & Data Export

### Run Headless
```bash
python -m simulator --headless --steps N --export output/data.json --seed 42
```

### Exported Data Format (JSON)
```json
[
  {
    "step": 1,
    "grazer_count": 197,
    "predator_count": 50,
    "total_grazer_energy": 117043.87,
    "total_predator_energy": 19948.13,
    "births_grazer": 0,
    "births_predator": 0,
    "deaths_grazer": 3,
    "deaths_predator": 0
  },
  ...
]
```

### Statistics Tracked (per timestep)
- Population counts (grazers, predators)
- Total population energy
- Cumulative births/deaths per species
- Extinction event logging

---

## Mathematical Summary

### Distance Metric (Bounded Chebyshev)
```
dist((y1,x1), (y2,x2)) = max(
    min(|y1-y2|, H - |y1-y2|),
    min(|x1-x2|, W - |x1-x2|)
)
```

### Resource Ratio
```
ratio(y, x) = resources(y, x) / capacity(y, x)  [0 if capacity=0]
```

### Neighbor Depletion Suppression
```
neighbor_avg = mean(resource_ratio of 8 neighbors)
depletion = clip(1.0 - neighbor_avg / 0.6, 0, 1)
suppression = clip(1.0 - depletion × 2.2, 0.0, 1.0)
```

### Herd Behavior Vectors
```
cohesion = mean(neighbor_positions) - my_position
alignment = mean(neighbor_velocities)
separation = Σ (my_position - neighbor_position) / distance² for close neighbors
```

### Pack Target Scoring
```
score = (chase_radius + 5 - distance_to_prey) + nearby_predators × 3
```

---

## Extensibility Points

1. **New Terrain Types**: Add to `terrain_types` in config
2. **New Species**: Extend `Entity` base class
3. **New Behaviors**: Override `move()`, `eat()`, `update_behavior()`
4. **New Sensors**: Add detection methods to entities
5. **Biome Generators**: Extend `_generate_terrain_biomes()`
6. **Analysis Tools**: Use exported JSON for population dynamics analysis
# Resource Simulator - User Stories

This document contains comprehensive user stories for all implemented features in the Resource Simulator, organized by feature area.

---

## 1. Core Simulation

### 1.1 Resource Regeneration and Diffusion

**As a researcher, I want resources to regenerate based on terrain-specific rates and local depletion, so that I can observe realistic ecosystem carrying capacity dynamics.**

**Acceptance Criteria:**
- Resources regenerate per timestep using formula: `deficit × regen_rate × 0.025 × suppression`
- Suppression factor reduces regrowth when neighboring cells are depleted (<60% capacity)
- Diffusion spreads resources to adjacent cells using 4-neighbor kernel with D=0.1
- Resource levels are clamped between 0 and terrain capacity
- Ocean and rock terrain have zero regeneration and capacity

**As a researcher, I want resource diffusion to simulate natural spread of vegetation, so that local depletion doesn't create permanent dead zones.**

**Acceptance Criteria:**
- Diffusion uses kernel `[[0,1,0],[1,-4,1],[0,1,0]] × diffusion_rate`
- Applied after regeneration each timestep
- Resources never exceed local capacity after diffusion
- Configurable `resource_diffusion_rate` (default 0.1)

### 1.2 Grazer Herd Behavior

**As an ecologist, I want grazers to exhibit herd behavior (cohesion, alignment, separation), so that I can study collective movement patterns in prey species.**

**Acceptance Criteria:**
- Each grazer detects neighbors within 8 cells (herd_radius)
- Cohesion: moves toward center of mass of nearby herd members (weight 0.4)
- Alignment: matches average movement direction of neighbors (weight 0.3)
- Separation: avoids crowding within 2 cells (weight 0.3)
- 70% chance to follow herd direction when neighbors present, otherwise gradient/random walk
- Individual `gradient_weight` varies per grazer (normal distribution around mean, clipped to [0,1])

**As a researcher, I want grazers to balance gradient-following with energy-aware exploration, so that I can observe adaptive foraging strategies.**

**Acceptance Criteria:**
- Grazers with energy >80% max get bonus for exploring depleted areas
- Grazers with energy >95% max slightly more willing to cross mountains
- Gradient weight determines probability of gradient-following vs random walk
- Vision radius (default 5) limits resource detection range

### 1.3 Predator Pack Hunting Behavior

**As a behavioral ecologist, I want predators to coordinate in pack hunting, so that I can study cooperative predation dynamics.**

**Acceptance Criteria:**
- When ≥1 predator within 10 cells, 60% chance to enter pack hunting mode
- Pack targets grazers with most nearby predators (flanking behavior)
- 40% chance per step to flank from direction of pack center
- Solo pursuit: direct chase of closest grazer within chase_radius (7 cells)
- Resource sensing: scans for depleted patches (<30% capacity) within 10 cells
- Pack wander: moves toward center of nearby predator group (≤10 cells) with 50% probability

**As a researcher, I want predators to feed from carcasses, so that I can observe scavenging behavior and energy transfer.**

**Acceptance Criteria:**
- Predators within 1 cell of carcass can consume energy
- Consumption limited to `energy_from_prey / 10` per step
- Carcass energy depletes over time (2% decay per step)
- Carcasses removed when energy < 1

### 1.4 Energy System and Metabolism

**As a systems biologist, I want entities to have realistic energy metabolism, so that I can model population viability under resource constraints.**

**Acceptance Criteria:**
- Grazers: -10 energy/step, +18 energy/resource unit consumed (×efficiency)
- Predators: -3 energy/step, +700 energy/prey killed
- Efficiency = max(0.3, resource_ratio) — 30% minimum in depleted cells
- Energy capped at species-specific `max_energy`
- Death when energy ≤ 0

### 1.5 Reproduction and Death Mechanics

**As an evolutionary biologist, I want reproduction to require minimum age and energy thresholds, so that I can observe life-history trade-offs.**

**Acceptance Criteria:**
- Grazers: reproduce at age ≥55, energy ≥650, cost = 45% of parent energy
- Predators: reproduce at age ≥65, energy ≥250, cost = 30% of parent energy
- Offspring placed within 1 cell of parent
- Offspring inherit parent's `gradient_weight` with mutation (grazers)
- Offspring start with `offspring_initial_energy` (200)
- Death tracked in statistics (births/deaths per species)

### 1.6 Extinction Detection

**As a simulation analyst, I want automatic extinction detection and reporting, so that I can identify collapse conditions without manual monitoring.**

**Acceptance Criteria:**
- Simulation ends when both grazer and predator populations reach 0
- Final statistics printed: step count, total births/deaths per species, max population reached
- Headless mode exports extinction data in JSON
- GUI displays "[ENDED]" status with restart option (R key)

---

## 2. Environment & Terrain

### 2.1 Procedural Biome Generation

**As a world builder, I want procedural biome generation with mountains, forests, plains, and ocean borders, so that I can create diverse landscapes without manual design.**

**Acceptance Criteria:**
- 200×200 grid (configurable) with 5-cell ocean border
- Mountain ridges via ridge noise: `(1 - |perlin|)²` with directional bias
- Natural passes carved at thinnest points of large ridge systems (>200 cells)
- Forests biased toward mountain proximity (within 8 cells) with organic expansion (3 iterations)
- Plains fill remaining areas
- Terrain smoothed (1 iteration majority filter)

### 2.2 Terrain Properties

**As a modeler, I want terrain types with distinct resource and mobility properties, so that landscape heterogeneity affects population dynamics.**

**Acceptance Criteria:**
| Terrain | Color | Regen | Capacity | Passable | Mobility (Prey/Pred) | Visibility (Prey/Pred) |
|---------|-------|-------|----------|----------|----------------------|------------------------|
| Ocean   | (30,60,180) | 0.0 | 0 | No | 0.0/0.0 | 0.0/0.0 |
| Rock    | (80,80,80) | 0.0 | 0 | Yes | 0.5/0.3 | 1.0/1.0 |
| Plains  | (180,200,100) | 1.5 | 30 | Yes | 1.0/1.0 | 1.0/1.0 |
| Forest  | (60,140,60) | 0.5 | 150 | Yes | 0.5/0.75 | 0.4/0.6 |

### 2.3 Resource Regeneration with Neighbor Depletion Suppression

**As an ecologist, I want resource regeneration to be suppressed by neighbor depletion, so that overgrazing creates realistic feedback loops.**

**Acceptance Criteria:**
- Neighbor depletion = average resource ratio of 8 neighbors
- Depletion factor = `clip(1.0 - neighbor_avg / 0.6, 0, 1)`
- Suppression = `clip(1.0 - depletion × 2.2, 0.0, 1.0)`
- Applied multiplicatively to base regeneration

### 2.4 Resource Diffusion

**As a spatial ecologist, I want resource diffusion between adjacent cells, so that resource gradients form naturally.**

**Acceptance Criteria:**
- Discrete Laplacian diffusion with rate D=0.1
- Wraps at boundaries (toroidal for diffusion only)
- Clamped to [0, capacity] after diffusion
- Configurable via `resource_diffusion_rate`

### 2.5 Terrain Mobility and Visibility Modifiers

**As a behavioral ecologist, I want terrain to affect movement speed and detection range, so that landscape structure influences predator-prey interactions.**

**Acceptance Criteria:**
- Mobility multipliers apply to entity move_speed per terrain type
- Visibility multipliers scale base vision/chase/sense radii
- Rock: high visibility, low mobility
- Forest: low visibility, moderate mobility
- Plains: baseline (1.0) for both
- Ocean: impassable, zero visibility

---

## 3. Visualization & Rendering

### 3.1 4×4 Per-Cell Rendering

**As a user, I want each simulation cell rendered as a 4×4 subpixel grid showing terrain, resources, and entities, so that I can see dense information at a glance.**

**Acceptance Criteria:**
- Each cell = 4×4 subpixels (8×8 pixels at cell_size=8)
- Top-Left quadrant: terrain color
- Top-Right quadrant: resource level (dark gray → bright green gradient)
- Bottom-Left quadrant: predator (orange) or carcass (red) presence
- Bottom-Right quadrant: grazer (yellow) presence
- Hovered cell highlighted with white pixel at bottom-right subpixel

### 3.2 Ocean Border Rendering

**As a user, I want ocean borders visibly distinct from inland water, so that map boundaries are clear.**

**Acceptance Criteria:**
- 5-cell wide ocean border around entire map
- Rendered with ocean color (30, 60, 180)
- Zero resources, impassable
- Visible at all zoom levels

### 3.3 Zoom/Pan with Mouse Wheel and Middle-Click Drag

**As a user, I want smooth zoom and pan controls, so that I can inspect details or view the entire map.**

**Acceptance Criteria:**
- Mouse wheel up/down: zoom in/out (1.15x per step, range 0.25x–4.0x)
- Zoom centered on mouse cursor (world position under cursor stays fixed)
- Middle mouse drag: pan view
- Ctrl+= / Ctrl+-: keyboard zoom in/out
- 0 key: reset zoom to 1.0 and pan to (0,0)

### 3.4 Layer Toggles (F1-F4)

**As a user, I want to toggle visibility of each entity layer independently, so that I can focus on specific population dynamics.**

**Acceptance Criteria:**
- F1: Toggle resources layer
- F2: Toggle grazers layer
- F3: Toggle predators layer
- F4: Toggle carcasses layer
- Changes apply immediately (dirty flag triggers re-render)
- Layer state persists across pause/resume

---

## 4. Inspection & Analysis

### 4.1 Keyboard-Driven Inspection Mode (I Key)

**As a researcher, I want to enter inspection mode with a keyboard shortcut, so that I can quickly analyze local conditions without mouse menus.**

**Acceptance Criteria:**
- Press I to toggle inspection mode
- Initial position set to current mouse position (or map center if mouse outside)
- Press I again to exit inspection mode
- Status bar shows "[INSPECT MODE]"

### 4.2 Arrow Key Navigation of Inspection Box

**As a researcher, I want to navigate the inspection region with arrow keys, so that I can systematically survey the map.**

**Acceptance Criteria:**
- ←/→/↑/↓ moves inspection center by 1 cell per press
- Bounded by map edges (respecting inspection_radius = 2)
- 5×5 grid inspection region (radius 2)
- Updates tooltip and world-view box in real-time

### 4.3 Yellow Inspection Box on World View

**As a researcher, I want a visible yellow box marking the 5×5 inspection region on the main map, so that I can see the inspected area in context.**

**Acceptance Criteria:**
- Yellow rectangle (2px) drawn around 5×5 cell region
- Scales correctly with zoom and pan
- Position matches inspection_x, inspection_y coordinates
- Only visible in inspection mode

### 4.4 Floating Tooltip with 5×5 Grid + Details

**As a researcher, I want a floating tooltip showing a 5×5 terrain/resource/entity grid plus detailed center-cell info, so that I can analyze local conditions at a glance.**

**Acceptance Criteria:**
- Tooltip panel 300×350px appears near mouse cursor
- 5×5 grid at top: each cell shows terrain color, resource %, entity counts (G/P/C)
- Center cell highlighted with yellow border
- Bottom section: center cell terrain, resources (current/capacity/%), regen rate, mobility/visibility for prey/predator
- Entity details: ID, energy, age for each entity at center
- Carcass energy shown if present
- Tooltip stays on screen (clamps to window edges)

### 4.5 Hover Highlight

**As a user, I want the cell under my mouse cursor highlighted, so that I can identify exact cell positions.**

**Acceptance Criteria:**
- Bottom-right subpixel (3,3) of hovered cell set to white (255,255,255)
- Updates in real-time with mouse motion
- Works at all zoom levels

### 4.6 Follow Mode (F Key)

**As a researcher, I want to track a specific entity with a visual indicator and live stats panel, so that I can study individual behavior over time.**

**Acceptance Criteria:**
- Press F to toggle follow mode
- Click entity to select for tracking
- Circle drawn around tracked entity (radius = max(8, cell_size×2))
- Side panel (270×180px) shows: position, energy/max, age, ID, species-specific params
- Panel updates in real-time
- Auto-disables if entity dies
- Press F again to exit follow mode

---

## 5. Configuration System

### 5.1 Keyboard-Driven Config Window (C Key → G/P/T/S/M)

**As a user, I want a keyboard-driven configuration window, so that I can adjust parameters without leaving the simulation view.**

**Acceptance Criteria:**
- Press C to open config window (right-side panel, 440px wide)
- Main menu shows 4 categories: [G]razer, [P]redator, [T]errain, [S]imulation
- Press G/P/T/S to enter category, M to return to main, ESC to close
- Window closes on click outside panel

### 5.2 Live Parameter Sliders for All Entity Types

**As a researcher, I want live sliders for all physiological and behavioral parameters, so that I can experiment with parameter effects in real-time.**

**Acceptance Criteria:**
- Grazer sliders: Energy/Step, Energy/Resource, Repro Min Energy, Repro Cost, Gradient Weight, Move Speed, Vision Radius
- Predator sliders: Energy/Step, Energy/Prey, Repro Min Energy, Repro Cost, Chase Radius, Move Speed, Resource Sense
- Terrain sliders: Resource Diffusion, Plains Regen/Capacity, Forest Regen/Capacity, Rock Regen
- Simulation sliders: Timestep, Grid Width, Grid Height
- Sliders show current value, update config object immediately on drag
- Integer sliders show whole numbers, float sliders show 2 decimals

### 5.3 Layer Toggles (F1-F4) in Config

**As a user, I want layer visibility toggles accessible from config, so that I can manage layers without memorizing hotkeys.**

**Acceptance Criteria:**
- F1-F4 keys toggle layers globally (same as main view)
- Config window reflects current layer state
- Changes apply immediately

### 5.4 Live Parameter Updates

**As an experimenter, I want parameter changes to apply immediately to the running simulation, so that I can observe effects without restarting.**

**Acceptance Criteria:**
- Slider drag → immediate `setattr` on config object
- Simulation reads config values each step
- No restart required for any parameter
- Grid size changes require reset (handled by R key)

---

## 6. Creator Mode

### 6.1 Toggle Creator Mode (M Key)

**As a scenario designer, I want to toggle creator mode with a single key, so that I can quickly switch between observation and editing.**

**Acceptance Criteria:**
- Press M to toggle creator mode
- Status bar shows "[CREATOR MODE - Brush: X Size: Y]"
- Disables inspect/follow modes when activated
- Press M again to exit

### 6.2 Brush Tools (1/2/3 for Rock/Plains/Forest)

**As a world builder, I want keyboard-selected brushes for each terrain type, so that I can paint landscapes efficiently.**

**Acceptance Criteria:**
- 1 = Rock brush (impassable, no resources)
- 2 = Plains brush (fast regen, low capacity)
- 3 = Forest brush (slow regen, high capacity)
- Current brush shown in status bar
- Brush selection persists across sessions

### 6.3 Brush Size Adjustment (+/-)

**As a world builder, I want adjustable circular brush size, so that I can paint both fine details and large areas.**

**Acceptance Criteria:**
- + / = key: increase brush size (max 10)
- - key: decrease brush size (min 1)
- Circular brush: affects cells where `dy² + dx² ≤ radius²`
- Current size shown in status bar

### 6.4 Left-Click to Paint Terrain

**As a world builder, I want to paint terrain by dragging the mouse, so that I can create custom landscapes.**

**Acceptance Criteria:**
- Left mouse drag in creator mode paints current brush terrain
- Only affects cells within brush radius
- Terrain properties (capacity, regen, passable) update automatically
- Triggers re-render (dirty flag)

### 6.5 G+Click to Spawn Grazer, P+Click to Spawn Predator

**As a scenario designer, I want to place entities directly on the map, so that I can set up specific initial conditions.**

**Acceptance Criteria:**
- Hold G + left click: spawn grazer at cell
- Hold P + left click: spawn predator at cell
- Entities spawned with default energy (500 grazer, 300 predator)
- Uses current config parameters for new entities
- Right-click also spawns (checks G/P keys)

### 6.6 Automatic Terrain Property Updates

**As a modeler, I want terrain properties to update automatically when painting, so that resource dynamics immediately reflect new terrain.**

**Acceptance Criteria:**
- Painting terrain updates: `terrain`, `resource_capacity`, `regen_rate`, `passable` arrays
- Uses `TerrainConfig` from current config
- No manual refresh needed
- Existing resources on cell persist but respect new capacity

---

## 7. Simulation Control

### 7.1 Pause/Resume (Space)

**As a user, I want to pause and resume the simulation, so that I can inspect states without time pressure.**

**Acceptance Criteria:**
- Space key toggles paused state
- Stats bar updates show current state
- No simulation steps processed while paused
- Rendering continues at display FPS

### 7.2 Reset Simulation (R Key)

**As an experimenter, I want to reset the simulation while keeping the current environment, so that I can test different initial populations on the same landscape.**

**Acceptance Criteria:**
- R key creates new Simulation with same config
- Terrain and resources preserved (new Environment not generated)
- New initial populations spawned per config
- Statistics reset to zero
- Follow/inspect/creator modes reset

### 7.3 Reload Environment (E Key)

**As an experimenter, I want to regenerate the environment while keeping current entities, so that I can test landscape changes on existing populations.**

**Acceptance Criteria:**
- E key creates new Environment (new biome generation)
- Existing grazers and predators preserved at same positions
- New resources initialized on new terrain
- Statistics preserved (births/deaths history maintained)
- Simulation step count continues

### 7.4 Zoom/Pan Controls

**As a user, I want comprehensive view navigation, so that I can examine any part of the map at any scale.**

**Acceptance Criteria:**
- Mouse wheel: zoom centered on cursor (0.25x–4.0x)
- Ctrl+= / Ctrl+-: keyboard zoom
- 0 key: reset zoom=1.0, pan=(0,0)
- Middle drag: pan view
- All controls work in all modes

### 7.5 Creator Mode (M Key) with Brushes

**As a scenario designer, I want full creator mode with terrain brushes and entity spawning, so that I can design custom experiments.**

**Acceptance Criteria:**
- M key toggles creator mode
- Brush selection: 1=rock, 2=plains, 3=forest
- Brush size: +/- keys (1-10)
- Left drag: paint terrain
- G+click: spawn grazer
- P+click: spawn predator
- Automatic terrain property updates

### 7.6 Follow Mode (F Key)

**As a researcher, I want to track individual entities with live stats, so that I can study individual life histories.**

**Acceptance Criteria:**
- F key toggles follow mode
- Click entity to select
- Circle indicator + side stats panel
- Auto-clear on entity death
- F again to exit

### 7.7 Layer Toggles (F1-F4)

**As a user, I want function-key layer toggles, so that I can quickly show/hide population layers.**

**Acceptance Criteria:**
- F1: Resources
- F2: Grazers
- F3: Predators
- F4: Carcasses
- Independent toggle state
- Immediate visual update

### 7.8 Layer Toggles: G/P/T/R Keys

**As a user, I want alternative single-key layer toggles, so that I have mnemonic shortcuts.**

**Acceptance Criteria:**
- G: Toggle grazers
- P: Toggle predators
- T: Toggle carcasses
- R: Toggle resources
- Duplicate of F1-F4 functionality
- Works in all modes

---

## 8. Data Export & Analysis

### 8.1 Headless Mode with JSON Export

**As a data scientist, I want to run simulations headless and export time-series data to JSON, so that I can analyze population dynamics programmatically.**

**Acceptance Criteria:**
- `--headless` flag runs without GUI
- `--steps N` sets max timesteps (0 = infinite)
- `--export path.json` writes population data
- `--seed N` for reproducible runs
- `--export-interval` controls write frequency (currently writes all steps)

### 8.2 Population Statistics Tracking

**As an analyst, I want per-timestep population statistics exported, so that I can plot population dynamics.**

**Acceptance Criteria:**
JSON export includes per-step:
- `step`: timestep number
- `grazer_count`, `predator_count`, `carcass_count`
- `total_grazer_energy`, `total_predator_energy`, `total_carcass_energy`
- `births_grazer`, `births_predator` (cumulative)
- `deaths_grazer`, `deaths_predator` (cumulative)

### 8.3 Extinction Detection and Reporting

**As a researcher, I want extinction events logged in exported data, so that I can identify collapse conditions in batch runs.**

**Acceptance Criteria:**
- Headless mode prints extinction summary to stdout
- Final step, total births/deaths, max populations reported
- JSON includes final step with zero populations
- GUI shows "[ENDED]" with same statistics

---

## 9. Persistence & State

### 9.1 Creator Mode State Persistence

**As a scenario designer, I want creator mode brush settings to persist, so that I don't need to reconfigure when re-entering creator mode.**

**Acceptance Criteria:**
- Brush type (1/2/3) remembered across toggle
- Brush size remembered across toggle
- State stored in Renderer instance

### 9.2 Config Changes Applied Live

**As an experimenter, I want all config changes to apply immediately without restart, so that I can iterate rapidly.**

**Acceptance Criteria:**
- Slider changes → immediate `setattr` on config objects
- Simulation reads live config each step
- Environment parameters (diffusion, terrain regen/capacity) update live
- Entity parameters (energy costs, speeds, radii) update live
- Only grid size requires reset (R key)

### 9.3 Reset Simulation (R) Keeps Environment

**As an experimenter, I want reset to preserve the current terrain and resources, so that I can test population dynamics on identical landscapes.**

**Acceptance Criteria:**
- R key: `sim = Simulation(config)` with existing `sim.env`
- New initial populations spawned
- Terrain, resources, capacities unchanged
- Statistics zeroed

### 9.4 Reload Environment (E) Keeps Entities

**As an experimenter, I want environment reload to preserve existing entities, so that I can observe how populations adapt to new landscapes.**

**Acceptance Criteria:**
- E key: creates new Environment, keeps `sim.grazers` and `sim.predators`
- Entities retain position, energy, age, ID
- New resources initialized on new terrain
- Step count and statistics history preserved
- Carcasses cleared (new environment)

---

## Summary

This document covers **50+ user stories** across **9 feature areas**, representing the complete implemented functionality of the Resource Simulator as of the current codebase. Each story follows the format:

**As a [role], I want to [action], so that [benefit]**

with specific, testable acceptance criteria derived from the actual implementation in:
- `src/simulator/simulation.py` (core loop, entities, behaviors)
- `src/simulator/environment.py` (terrain, resources, biomes)
- `src/simulator/entities.py` (entity classes)
- `src/simulator/render.py` (visualization, UI, creator mode)
- `src/simulator/config.py` (configuration system)
- `src/simulator/__main__.py` (entry points, headless mode)
- `config.yaml` (default parameters)
- `requirements-and-specifications/SPECIFICATION.md` (technical docs)
- `requirements-and-specifications/QUICKSTART.md` (user guide)
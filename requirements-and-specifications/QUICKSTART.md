# Resource Simulator - Quick Start Guide

## Installation
```bash
cd Resource-Simulator
pip install numpy scipy pyyaml pygame-ce
```

## Running

### Interactive GUI (Recommended)
```bash
python -m simulator
```
Or double-click `Resource Simulator.bat` on Desktop.

### Headless (Batch/Data Export)
```bash
python -m simulator --headless --steps 1000 --export output/sim_data.json --seed 42
```

---

## Quick Reference: Keyboard Controls

| Key | Mode | Description |
|-----|------|-------------|
| **SPACE** | All | Pause/Resume simulation |
| **ESC** | All | Quit |
| **G** | All | Toggle grazers visibility |
| **P** | All | Toggle predators visibility |
| **T** | All | Toggle carcasses visibility |
| **R** | All | Toggle resources visibility |
| **C** | All | Open Config Window (keyboard-driven) |
| **R** | All | Reset simulation (keep terrain) |
| **E** | All | Reload environment (keep entities) |
| **M** | All | Toggle **Creator Mode** |
| **I** | All | Toggle **Inspect Mode** |
| **F** | All | Toggle **Follow Mode** |
| **0** | All | Reset zoom/pan |
| **F1-F4** | All | Toggle layers (Resources, Grazers, Predators, Carcasses) |
| **Ctrl+=/-** | All | Zoom in/out |
| **Middle drag** | All | Pan view |

### Config Window (C)
Opens keyboard-driven menu with 4 categories:
| Key | Category |
|-----|----------|
| **G** | Grazer settings |
| **P** | Predator settings |
| **T** | Terrain settings |
| **S** | Simulation settings |
| **M** | Return to main menu |
| **ESC** | Close (from main) / Back to menu (from sub) |

### In Creator Mode (M)
| Key | Action |
|-----|--------|
| **1** | Rock brush |
| **2** | Plains brush |
| **3** | Meadow brush (grazer stronghold) |
| **4** | Scrub brush (rabbit stronghold) |
| **5** | Forest brush |
| **+ / -** | Brush size 1-10 |
| **Left drag** | Paint terrain |
| **G + Left click** | Spawn grazer |
| **P + Left click** | Spawn predator |

### Inspect (hover)
| Action | Result |
|--------|--------|
| Hover mouse | Cell terrain, resources and entities in the inspect pane |

### Follow Mode
| Action | Result |
|--------|--------|
| Left click animal | Camera tracks it, white circle drawn around it, its live stats pinned in the inspect pane |
| Left click bare ground | Stop following |
| **Esc** | Stop following |
| Middle-mouse pan | Stop following |

### Zoom / Pan
| Action | Control |
|--------|---------|
| Zoom In | Mouse wheel **up** (centered on cursor) / **Ctrl + =** |
| Zoom Out | Mouse wheel **down** (centered on cursor) / **Ctrl + -** |
| Reset View | **0** key |
| Pan | **Middle mouse button** drag |

---

## Visualization Guide

Each cell = **8×8 pixels** (configurable `cell_size`):

```
┌──┬──┬──┬──┐
│ T  │ T  │ T  │ R  │  T = Terrain color (ocean/rock/plains/forest)
├──┼──┼──┼──┤
│ T  │ P  │ G  │ R  │  P = Predator (orange) if present
├──┼──┼──┼──┤
│ T  │ C  │ G  │ R  │  C = Carcass (red) if present
├──┼──┼──┼──┤
│ T  │ T  │ T  │ R  │  G = Grazer (yellow) if present
└──┴──┴──┴──┘  R = Resource level (dark gray→bright green gradient)
```

**Terrain Colors:**
- 🌊 **Ocean** (30,60,180) - Impassable border (5 cells wide)
- 🟫 **Rock** (80,80,80) - Impassable, no resources
- 🟨 **Plains** (180,200,100) - Fast regen, low capacity (30)
- 🟩 **Forest** (60,140,60) - Slow regen, high capacity (150)

**Resource Gradient (Perimeter):**
- ⚫ Dark gray = Depleted (0%)
- 🟡 Yellow = Medium (50%)
- 🟢 Bright green = Full (100%)

---

## Config Window (C)

Opens keyboard-driven menu with 4 categories:

| Key | Category |
|-----|----------|
| **G** | Grazer settings |
| **P** | Predator settings |
| **T** | Terrain settings |
| **S** | Simulation settings |
| **M** | Return to main menu |
| **ESC** | Close (from main) / Back to menu (from sub) |

### Grazer Tab:
- Energy/Step (1-100) - Metabolic cost
- Energy/Resource (1-50) - Food value
- Repro Min Energy (100-1000) - Breeding threshold
- Repro Cost (0.1-0.9) - Energy given to offspring
- Gradient Weight (0-1) - Herd vs gradient bias
- Move Speed (1-5), Vision Radius (1-15)

### Predator Tab:
- Energy/Step (1-20), Energy/Prey (100-1000)
- Repro Min Energy (100-600), Repro Cost (0.1-0.9)
- Chase Radius (1-20), Move Speed (1-5)
- Resource Sense (1-20)

### Terrain Tab:
- Resource Diffusion (0-0.1)
- Plains Regen (0-2), Plains Capacity (10-200)
- Forest Regen (0-2), Forest Capacity (10-300)
- Rock Regen (0-1)

### Simulation Tab:
- Timestep ms (10-500)
- Grid Width (50-500)
- Grid Height (50-500)

*Changes apply instantly to running simulation.*

---

## Understanding the Ecology

### Population Dynamics
- **Plains** = Fast food, runs out quick → boom/bust cycles
- **Forests** = Slow steady food → sustains larger herds
- **Mountains** = Barriers creating corridors and choke points
- **Passes** = Natural hunting grounds for predators
- **Ocean Borders** = Impassable edges, no wrapping

### Herd Behavior (Grazers)
- **Cohesion**: Move toward group center
- **Alignment**: Match neighbor direction  
- **Separation**: Don't crowd (2 cell radius)
- 70% follow herd when neighbors nearby

### Pack Behavior (Predators)
- Hunt in loose groups when near each other
- Flank prey from pack center direction
- Wander together when no prey visible

### Energy Economy
```
Grazer:  -10/step  +18/resource (×efficiency)
Predator: -3/step  +700/kill
```
Efficiency drops to 30% in depleted cells → grazers must move to fresh grass.

---

## Typical Workflows

### 1. Observe Natural Dynamics
1. Run GUI: `python -m simulator`
2. Watch predator-prey oscillations
3. Note how mountains channel movement
4. Press **SPACE** to pause at interesting moments

### 2. Experiment with Parameters
1. Press **C** for Config Window
2. Increase *Grazer Gradient Weight* → more efficient grazing
3. Decrease *Predator Energy/Step* → more predators survive
4. Adjust *Forest Capacity* → changes carrying capacity

### 3. Design Custom Scenarios (Creator Mode)
1. Press **M** for Creator Mode
2. **1** + drag = paint mountain ranges
2. **3** + drag = create forest corridors
3. **G** + click = seed grazer herds
4. **P** + click = place predator packs
5. Press **M** again to exit, **SPACE** to run

### 4. Analyze Population Data
```bash
# Export data
python -m simulator --headless --steps 5000 --export output/run1.json

# Load in Python for analysis
import json, matplotlib.pyplot as plt
data = json.load(open('output/run1.json'))
steps = [d['step'] for d in data]
grazers = [d['grazer_count'] for d in data]
predators = [d['predator_count'] for d in data]
plt.plot(steps, grazers, label='Grazers')
plt.plot(steps, predators, label='Predators')
plt.legend(); plt.show()
```

### 5. Inspect Individual Behaviors
1. Press **I** (Inspect Mode)
2. Hover mouse over cells → sets initial position
3. **Arrow keys** → move inspection box 1 cell
3. Yellow box shows 5×5 inspection region in world view
4. Tooltip shows 5×5 grid + cell details

### 6. Follow Individual Entities
1. Press **F** (Follow Mode)
2. Click an entity → tracks it with live energy/age readout

---

## Tips for Stable Ecosystems

| Issue | Fix |
|-------|-----|
| Predators die out | Lower Predator Energy/Step, increase Energy/Prey |
| Grazers explode | Increase Grazer Energy/Step, lower Energy/Resource |
| Both extinct | Increase Forest Capacity, lower Diffusion |
| Oscillations too wild | Increase Repro Min Energy for both |

---

## File Structure
```
Resource-Simulator/
├── config.yaml           # All parameters
├── src/simulator/
│   ├── __main__.py       # Entry point
│   ├── simulation.py     # Core simulation loop
│   ├── environment.py    # Terrain, resources, biome gen
│   ├── entities.py       # Grazer, Predator classes
│   └── render.py         # Visualization, UI, creator mode
├── output/               # Exported JSON data
└── requirements-and-specifications/
    ├── SPECIFICATION.md  # Full technical docs
    └── QUICKSTART.md     # This file
```

---

## Common Issues

**"pygame not found"** → `pip install pygame-ce`

**Simulation too slow** → Reduce grid size in config, or increase timestep_ms

**Config window won't open** → Ensure not in headless mode

**Entities don't spawn** → Check Creator Mode is active (M), try G/P + click on passable terrain

**Extinction too fast** → Default config tuned for dynamics; adjust Repro Min Energy downward

**Zoom not working** → Use mouse wheel or Ctrl+=/- (must hover over simulation area)

**Inspection box not moving** → Ensure Inspect Mode is active (I), use arrow keys

---

## Next Steps
- Read `SPECIFICATION.md` for complete mathematical models
- Modify `config.yaml` for custom scenarios  
- Extend `entities.py` to add new species
- Use exported JSON for scientific analysis
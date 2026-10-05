# As-Built Specification — Resource Simulator

This document specifies the **current implemented behaviour** of the simulator in
enough detail to rebuild it from scratch. It is a description of what the code
*does*, not what the original `SPECIFICATION.md` / `USER_STORIES.md` asked for.
Where the two disagree, this document wins; where the implementation deviates from
its own documentation, the deviation is called out explicitly.

**Known unresolved defect:** predator populations still reach extinction. See
[§15](#15-known-defects). The simulator runs correctly and every mechanism below is
implemented, but the predator/prey equilibrium is not calibrated.

---

## 1. Scope

A 2D grid ecosystem simulation with three agent species, a procedural landscape, and
a resource field that regenerates, diffuses, and is consumed.

| Species | Role | Notes |
|---------|------|-------|
| Grazer | Prey | Continuous grazer; herds, never migrates |
| Rabbit | Prey | Stop-and-go migratory forager |
| Predator | Hunter | Pack/solo hunting, scavenging, depletion inference |

Carcasses are not agents but persistent world entities.

## 2. Module layout

```
src/simulator/
  config.py       dataclasses + YAML loader/saver
  entities.py     Entity, Prey, Grazer, Rabbit, Predator, Carcass
  environment.py  terrain generation, resource field, consumption
  simulation.py   step orchestration, behaviour, statistics
  render.py       pygame presentation
  __main__.py     CLI entry point and the run loops
  __init__.py     public re-exports (deliberately excludes render)
```

`src/` is a package layout: an editable install (`pip install -e .`) or
`PYTHONPATH=src` is required before `python -m simulator` resolves.

`pygame-ce` is the declared dependency, not `pygame`: upstream `pygame` publishes no
wheels for Python 3.13+ and its sdist fails to build because `distutils` was removed.

## 3. Coordinates, metrics, and movement

### 3.1 Grid

`y` indexes rows (0 = north), `x` indexes columns (0 = west). Both run
`0 .. map_size-1`. Entities hold float positions; all logic uses `int(y)`, `int(x)`.

### 3.2 Bounded Chebyshev distance

Used for **perception radii only** (vision, chase, sense, flock distance):

```
dy = |y1 - y2|
dx = |x1 - x2|
distance = max( min(dy, map_height - dy), min(dx, map_width - dx) )
```

This is toroidal: the cell 3 west of column 0 is at distance 3. Diagonal offsets are
not Euclidean, so `(4,4)` is distance 4, not 5.66.

**Exception — herd/separation vectors use plain, non-wrapped Chebyshev.** A toroidal
metric would report herd members as adjacent across the impassable ocean border, which
is physically meaningless. Direction vectors are likewise plain grid deltas.

### 3.3 Movement budget

There is no `int(move_speed × mobility)`. That floor of 1 made terrain speed
meaningless for slow species — a grazer with `move_speed 1` on rock (`mobility_prey 0.5`)
computed `max(1, int(0.5)) == 1` and moved exactly as fast as on plains.

Instead each entity carries `move_budget: float`. Per step:

```
move_budget += move_speed × mobility × behavioural_speed_multiplier
cells_moved  = int(move_budget)      # truncated, carry the remainder
move_budget -= cells_moved
```

The budget is per-entity and persists across steps. Slow terrain therefore costs
*time*, not a rounded penalty. Observed frames per cell:

| Terrain | `mobility_prey` | prey frames/cell | `mobility_predator` | predator frames/cell |
|---|---|---|---|---|
| plains | 1.0 | 1 | 1.0 | 1 |
| forest | 0.6 | 2 | 0.7 | 2 |
| rock | 0.2 | 5 | 0.25 | 4 |

### 3.4 Movement metabolism

Distinct from base metabolism (`energy_per_step`). Charged per cell actually advanced:

```
energy -= cells_moved × movement_energy_cost × behavioural_cost_multiplier
```

If this drives energy to zero the entity dies (`alive = False`) and the death is
counted in the same step.

## 4. Environment

### 4.1 Terrain generation

Terrain is stored as an `int8` index array into `terrain_names` (not object dtype),
so all whole-map operations are vectorised.

1. Four noise fields are generated from a seeded `numpy.random.Generator`:
   - `base_noise` — fractal Perlin, 5 octaves, persistence 0.5, scale 0.015
   - `ridge_noise` — ridged multifractal, 4 octaves, persistence 0.5, **scale 0.05**,
     with exponent **3** (`(1-|perlin|)³`) to sharpen crests into thin lines
   - `forest_noise` — fractal Perlin, 4 octaves, persistence 0.6, scale 0.01

2. **Rock ridges.** The mask is taken from `ridge_noise` alone. Critically, a
   **high-pass filter** is applied before thresholding:

   ```
   sharpened = ridge_noise - gaussian_filter(ridge_noise, sigma=3.0)
   cutoff     = quantile(sharpened, 1 - terrain_distribution.rock × 1.15)
   mask       = sharpened >= cutoff
   ```

   Thresholding the raw field directly yields one amorphous mass, because broad
   plateaus also read as high. The high-pass cut p95 ridge thickness from 20 cells
   to 6, and the largest connected blob from 1155 cells to 230.

   Then `binary_closing(iterations=1)` bridges one-cell gaps, `binary_opening` removes
   specks, and components smaller than `max(8, largest × 0.05)` are dropped.

   *Measured result:* ~5% coverage, median ridge width 2.8 cells, p95 6–7, 26–34
   branching components per map, consistent across seeds. Note this lands near 5%
   even though `terrain_distribution.rock` is 0.10 — the ×1.15 pre-smooth
   compensation and subsequent majority-filter erosion do not cancel out. Treat
   `terrain_distribution.rock` as a *pre-smoothing target*, not an achieved value.

3. **Forest / plains split.** Whatever the ridges leave is divided using
   `terrain_distribution`, not raw noise thresholds: the remaining cells are ranked
   by a forest score (`forest_noise × 0.5 + (1 - base_noise) × 0.3 +
   mountain_proximity × 0.2`, normalised) and the top
   `forest/(forest+plains)` share become forest. Forest then grows organically for 3
   iterations (`≥4` forest neighbours, or `≥3` within 6 cells of rock).

4. **Smoothing.** One iteration of an 8-neighbour majority filter.

5. **Mountain passes.** For each rock component larger than 200 cells, the thinnest
   point is found with a distance transform; if the ridge is thinner than 8 cells
   there, a 2-cell-wide corridor is carved to plains, with a 1-cell rock rim.

6. **Ocean border.** The outermost `ocean_border_width` (5) cells are set to ocean.

`terrain_distribution` is authoritative for biome ratios; `mobility_*`,
`visibility_*` and `resource_productivity` are *not* consulted during generation.

### 4.2 Resource field

Three parallel float64 arrays: `resources`, `resource_capacity`, `regen_rate`, plus
boolean `passable`, and `mobility_prey` / `mobility_predator` lookup grids. All are
derived from the current terrain by `refresh_terrain_properties()` and are re-derived
whenever terrain is painted.

`resource_ratio = resources / capacity`, or `0.0` where capacity is 0.

Initial resources are `uniform(0.3, 0.7) × capacity`.

### 4.3 Per-step field update

`Environment.step()` runs in this order, and the ordering matters:

```
1. _update_depletion_signal()   # compares against the *previous* post-regen snapshot
2. _regenerate()
3. _diffuse()
4. snapshot resources for the next step
```

`_update_depletion_signal` runs first and measures the consumption that happened
*since the last regeneration*, i.e. the whole of the previous simulation step's
grazing. Because `Simulation.step()` calls `env.step()` before any agent eats, this
yields exactly "what the grazers took" with no one-step lag and no change to the
caller.

**Regeneration**, proportional to deficit and gated by neighbour depletion:

```
deficit     = capacity - resources
base_regen  = regen_rate × deficit × regen_base_factor
depletion   = clip(1 - neighbour_avg_ratio / depletion_threshold, 0, 1)
suppression = clip(1 - depletion × depletion_suppression, min_regen_suppression, 1)
resources  += base_regen × suppression
```

`min_regen_suppression` (0.25) is a **floor**. Without it, suppression clipped to
zero once the neighbourhood average fell below ~0.27, and since regeneration is
proportional to deficit there was no floor anywhere else either — a stripped map
locked into a permanent dead state from which nothing could recover.

`neighbour_avg_ratio` averages the 8 neighbours over cells that have capacity.

**Diffusion** is a Laplacian convolution with `mode="wrap"` (toroidal), scaled by
`resource_diffusion_rate`, then clipped to `[0, capacity]`.

### 4.4 Depletion signal

A separate smoothed field used for predator resource inference:

```
loss = clip((post_regen_snapshot - resources) / capacity, 0, 1)
depletion_rate = depletion_rate × smoothing + (1 - smoothing) × loss    # smoothing 0.7
```

Regrowth is excluded by the lower clip. Because regen pushes cells back toward
capacity, a *persistently* overgrazed cell settles to `≈ 0` — so only **fresh** grazing
lights the field up. It is a change detector, not an abundance measure.

## 5. Consumption

```python
ratio      = get_resource_ratio(y, x)
efficiency = max(min_efficiency, ratio) × efficiency_scale
taken      = consume_resource(y, x, energy_from_resource × efficiency)
gained     = add_energy(taken)          # clamped to max_energy, returns the real delta
```

`consume_resource` returns `min(requested, available)`. **Intake is therefore capped by
what the cell physically holds**, which matters because `efficiency` is ratio-based:
with plains capacity at 14, a grazer requesting 18 units from a half-depleted cell
receives 7. This coupling caused a real regression (rabbit extinction) when capacity
was reduced without reducing `energy_from_resource`.

### 5.1 Terrain productivity

After intake, the post-metabolism surplus is taxed by terrain:

```
surplus = gained - energy_per_step
if surplus > 0:
    gained -= surplus × (1 - resource_productivity)
```

Plains (`1.0`) banks the entire surplus; forest (`0.333`) banks a third of it; rock
and ocean have `0.0` productivity and no capacity. The intake formula is unchanged —
productivity only affects what is *banked*, not what is extracted.

## 6. Entities

All are dataclasses. `id` comes from a module-level `itertools.count`, so IDs are
unique across all species and carcasses (the original implementation re-seeded its
counter in `__post_init__`, producing duplicates).

| Field | Notes |
|---|---|
| `x, y` | float position |
| `energy` | clamped to `[0, max_energy]` |
| `config` | live reference; `max_energy` and `move_speed` are properties that read it |
| `age` | integer; **also the step counter** |
| `alive` | cleared on death; dead entities are filtered at end of step |
| `last_dy, last_dx` | last movement, for boid alignment |
| `move_budget` | fractional movement accumulator |

Config is read live rather than cached at construction, so parameters can be changed
between steps.

`Prey` adds `gradient_weight`, `inherited_gradient_weight`, `vision_radius`,
`foraging_state`, and `mutate_gradient_weight()`.

`Predator` adds `chase_radius`, `resource_sense_radius`, `eating_state`,
`investigate_{y,x,timer}`.

`Carcass` adds `energy`, `max_energy`, `age`, `consumed_by`. `provides_energy` is
`energy > 0`.

## 7. Simulation step order

```
Simulation.step()
  env.step()                 # regeneration + diffusion + depletion signal
  _refresh_fields()          # vectorised ratio field
  _build_indices()           # spatial indices, snapshot of positions
  carcass.step()             # decay
  _update_prey(grazers)
  _update_prey(rabbits)
  _update_predators()
  _reproduce()
  _cleanup()
  _record_stats()
  append to stats_history
  extinction check
```

Because the spatial indices are built once from a start-of-step snapshot, perception
is order-independent within a step.

### 7.1 Spatial index

Neighbour queries go through a cell-bucketed index, not a linear scan. Perception was
`O(n²)` pure Python, which made a 200×200 grid unusable — 525 ms/step at 1200 agents,
reduced to 81 ms by this change (roughly linear thereafter).

## 8. Prey behaviour

Per prey, per step:

```
was_alive = alive
prey.step_energy()                 # energy -= energy_per_step; age += 1; may kill
if was_alive and not alive: count death

state = _foraging_state(prey, cfg)

if migratory:
    foraging_state = state
    if state == "grazing":
        eat(env, grazing_efficiency)      # stand still, feed hard, do not move
    else:
        eat(env, moving_efficiency)       # feed on the move
        herd = _herd_vectors(...)
        target = _forage_target(...)
        _move_prey(...)                    # may starve via movement cost
else:
    herd = _herd_vectors(...)
    target = _forage_target(...)
    _move_prey(...)
    eat(env)
```

### 8.1 Foraging state machine (migratory species)

```
if foraging_state == "grazing":
    if local_ratio <= migrate_ratio_threshold or energy_ratio >= wander_energy_threshold:
        -> migrating
    else: -> grazing
else:
    if local_ratio >= graze_ratio_threshold and energy_ratio < graze_energy_target:
        -> grazing
    else: -> migrating
```

The design constraint that makes this meaningful: **travelling must not pay for
itself.** For rabbits, travelling intake peaks at
`energy_from_resource × moving_efficiency = 6 × 0.25 = 1.5` against a metabolic cost
of 5, so a rabbit in transit always loses energy and can only make progress by
standing still somewhere worth standing. Grazing peaks at `6 × 4.0 = 24`, i.e. `+19`
net on a full cell.

Measured over 700 steps: **60% grazing / 40% migrating**, mean displacement
0.171 cells/step averaged over both states.

### 8.2 Boids

Per neighbour within `herd_radius` (plain, non-wrapped Chebyshev):

```
cohesion   += neighbour.y - my.y
alignment  += neighbour.last_dy
if gap <= separation_radius:
    separation += (neighbour - me) × (separation_radius - gap + 1)
```

Separation points **away** from the neighbour. The formula in the original
`SPECIFICATION.md` had this sign inverted relative to its own stated intent
("avoid crowding").

On 70% of steps (`herd_follow_chance`), if any neighbour exists, the weighted vector
`(cohesion×0.4 + alignment×0.3, separation×0.3)` competes with the forage gradient.
Off those steps, or with no neighbours, the individual follows its gradient.

### 8.3 Fleeing

If any living predator is within `flee_radius` (Chebyshev 5), a weighted escape
direction is built (`1/gap` per threat, pointing away). It enters candidate scoring at
weight `flee_weight` (3.0), which outweighs the gradient so a detected predator
dominates the decision.

Fleeing also changes speed and cost (see §9).

### 8.4 Forage target

Vectorised over the prey's vision disc (radius `vision_radius`). Score is
`resource_ratio`, minus `0.02 ×` Chebyshev distance to self, over passable cells only.
The "explore depleted ground" bonus for well-fed prey is **disabled for migratory
species** — they are looking for food, so prospecting is counterproductive.

### 8.5 Candidate scoring

For each of the 8 passable neighbours:

```
score  = (current_target_distance - candidate_target_distance) × gradient_weight
score += herd_vector_projection × (1 - gradient_weight) × 0.5      (if herding)
score += separation_projection                                     (if herding)
score += threat_escape_projection × flee_weight                    (if threatened)
score += (1 - candidate_mobility) × 1.5                            (if energy_ratio > 0.95)
```

Peak-condition prey tolerate rough terrain. The highest score wins; if no neighbour is
passable, the prey does not move.

## 9. Speed and movement cost multipliers

| Behaviour | Speed | Movement cost |
|---|---|---|
| Prey travelling | ×1.0 | ×1.0 |
| Prey grazing (continuous grazer) | ×0.667 | ×1.0 |
| Prey fleeing | ×1.3 | ×2.5 |
| Predator wandering / investigating / pack-wander | ×1.0 | ×1.0 |
| Predator chasing (pack hunt or solo) | ×1.2 | ×1.4 |

`grazing_move_multiplier` applies to non-migratory species, which move while feeding.
Migratory species do not move at all while grazing, so no reduction applies to them.

Predators are faster than fleeing prey in every terrain (plains 2.40 vs 1.30 cells per
step; forest 1.68 vs 0.78; rock 0.60 vs 0.26), so pursuit is always possible — but
only just in forest and rock.

## 10. Predators

```
_update_predators():
    for every predator: eating_state = "moving"        # transient, reset per step
    _feed_on_carcasses()                              # one coordinated pass

    for each predator:
        was_alive = alive
        step_energy()
        if died: count death
        if not alive: continue

        if eating_state == "feeding": continue         # holds position while feeding
        _predator_behaviour(predator)                  # may move onto prey
        if _predator_kill_prey(predator): continue
```

### 10.1 Carcass scavenging

Feeding is resolved as a **single pass before any predator moves**, so every predator
around one carcass competes for one shared store deterministically:

```
for each carcass with energy > 0:
    feeders = living predators within Chebyshev feed_radius (1)   # own cell + 8
    sort feeders by (-age, id)                                    # oldest first
    for predator in feeders:
        if carcass empty: break
        taken = carcass.take_energy(energy_from_prey / consumption_divisor)
        predator.add_energy(taken)
        predator.eating_state = "feeding"
```

Consequences, all verified by test:

- More neighbours strips a carcass proportionally faster: 13 / 7 / 4 / 2 passes for
  1 / 2 / 4 / 8 feeding predators.
- Age order decides the remainder. On a scarce carcass the oldest takes a full draw,
  the next takes the leftover, the youngest get nothing.

`eating_state` is reset at the top of every predator update, so `"feeding"` suppresses
movement for exactly one step.

### 10.2 Kill and feed are separate

A kill transfers **no energy directly**:

```
for flock in (grazers, rabbits):
    for prey at the predator's exact cell:
        prey.alive = False
        count death; increment kills_<species>
        carcasses.append(Carcass(energy = prey.energy × carcass_energy_fraction))
        predator.eating_state = "killing"
```

The energy is only recoverable by scavenging on a later step. This removed a
double-count: the original paid out an immediate transfer *and* left a carcass holding
50% of the same animal.

The kill check runs **after** movement, because a predator closing on prey during its
chase must be able to consume it the same step. Checking beforehand let a fast predator
sweep straight over its target without ever sharing a cell.

`carcass_energy_fraction` is 1.0, so a kill is worth the prey's full stored energy.
Note that this makes `energy_from_prey` a **feeding rate**, not a reward — the value of a
kill is set entirely by how much energy the prey happened to hold.

### 10.3 Behaviour selection

```
pack      = living predators within pack_radius (10), excluding self
centre    = centroid of pack

if pack and rng < pack_hunt_chance (0.6):
    target = _pack_target(predator, pack)
    if target:
        flank = centre and rng < flank_chance (0.4)
        goal  = flank goal if flanking else target cell
        move_toward(goal, chasing=True)
        return

elif _solo_target(predator) is not None:
    move_toward(target, chasing=True)
    return

if _investigate_depletion(predator): return
if pack and centre and rng < pack_wander_chance (0.5):
    move_toward(centre)
    return
_predator_wander(predator)
```

- **Vision** is `max(1, int(chase_radius × visibility_predator))`, sampled at the
  predator's cell only, and boosted by `vision_boost_on_investigate` (×1.5) while
  investigating. On plains/rock that is 7 (10 when investigating); in forest 4 (6).
- **`_solo_target`** takes the nearest prey inside vision.
- **`_pack_target`** scores each candidate as
  `(chase_radius + 5 - distance) + escorts × pack_score_bonus`, where escorts counts
  packmates within `pack_radius` of the prey — so the pack converges on the most
  contested, closest target.
- **Flanking** aims for the cell on the far side of the prey from the pack centre.
- **`can_see_through` / line of sight is not modelled.** The original implementation
  had a visibility check that returned `True` for every reachable cell (rock sees
  through rock; only impassable ocean returned `False`). There is no occlusion; vision
  is purely radial.

### 10.4 Resource sensing

When no prey is visible, if not already investigating (and while a prior waypoint is
live, the timer just decrements and the predator keeps walking to it), the environment
is scanned within `resource_sense_radius` scaled by terrain visibility:

```
evidence = where(passable and ratio <= depleted_ratio_threshold, depletion_rate, 0)
if evidence.max() < investigate_threshold: give up
else: commit a waypoint for investigate_steps and walk to it
```

This is the only consumer of the depletion signal, and it fires on *fresh* depletion
rather than bare ground.

## 11. Reproduction

Cost is a **fraction of the parent's current energy**, not of `max_energy`:

```
if age < reproduction_min_age or energy < reproduction_min_energy: no
cost = energy × reproduction_energy_cost
energy -= cost
child spawned in a random cell within Chebyshev 1 of the parent
```

Offspring inherit `inherited_gradient_weight` from the parent and then mutate it by
`N(inherited, gradient_weight_std)`, clipped to `[0,1]`. This makes gradient weight a
heritable trait under selection.

## 12. Starting conditions

Each individual spawns with:

- position: uniform random passable cell (up to 200 attempts, then map centre)
- energy: `uniform(0.5, 1.0) × max_energy`
- **age: uniform random from `0` to `reproduction_min_age` inclusive**

Randomised age avoids synchronising every individual's first reproduction and first
mortality wave, which produces artificial boom-bust cycles unrelated to the model.

## 13. Statistics and termination

`SimulationStats` is recorded every step and appended to `stats_history`:

- instantaneous: step, per-species counts and carcass count, per-species total and
  mean energy, total carcass energy, total resource
- cumulative: births and deaths per species, kills per prey species

Cumulative counters live on the simulation, not the stats record, so they survive
`reset()`'s history handling and never decrease. Every code path that kills an entity
increments the matching counter — starvation, predation, and starvation-by-movement-cost.

**Extinction** is declared when grazers, rabbits and predators are *all* empty.
Individual species going extinct does not stop the run; `extinct_step` and
`extinction_summary()` report peaks, births, deaths and kills.

## 14. Rendering

### 14.1 Layout

```
+--------------------------------+  +------------------+
|                                |  |  population      |
|                                |  |  chart           |
|   map viewport                 |  +------------------+
|   (window_width - 380)         |  |  colour legend   |
|                                |  +------------------+
|                                |  |  metrics         |
|                                |  |  key help        |
+--------------------------------+  +------------------+
```

The sidebar occupies the right `SIDEBAR_WIDTH = 380` px and is drawn as an opaque
panel. The map viewport is `window_width - 380`, so no panel can overlap the world.
Mouse input in the sidebar is rejected.

Default window 1404×768. `fit_view()` centres the camera on the map and zooms to fit;
it is called on `set_simulation` and bound to `0`.

### 14.2 Cached world layer

Terrain and resource are baked into one offscreen surface (1600×1600 at `cell_size 8`)
built from numpy, then blitted; only entity markers and UI are drawn per frame.
Rebuilds are split by cause — terrain repaints on painting / terrain reload / reset /
resize / encoding change, while resources are patched in place per step.

The buffer is a numpy array wrapped by `pygame.image.frombuffer`, which **aliases**
writable memory, so a quadrant patch is visible without rebuilding the surface.
`pygame.surfarray` is deliberately unused: on pygame-ce 2.5.x `array3d` returns a copy
and `make_surface` ignores its input.

Measured on a 200×200 map, 1024×768, dummy driver: **0.95 ms/frame** (~1050 fps) at
zoom 1.0 with grid and all layers, versus a 4.4–5.8 fps ceiling for the per-cell
`draw.rect` loop it replaced.

### 14.3 Two-tier cell encoding

Above `QUADRANT_MIN_ZOOM = 1.5`, cells use the documented 4×4 layout: terrain fills the
cell, the resource tint occupies its top-right quadrant. Below it — where a whole cell is
under 4 px and four quadrants carry no readable information — a cell is instead
**terrain hue scaled by resource brightness** across the entire cell
(`OVERVIEW_MIN_BRIGHTNESS = 0.32`). Switching modes forces a full layer rebuild.

### 14.4 Entities and legend

Carcasses draw **last**, on top of everything, with a minimum 3 px size and a light
outline. This matters: a kill deposits the carcass at the killer's own cell and that
predator stands on it while feeding, so drawing carcasses underneath meant the orange
marker of the very animal that created it hid the carcass for its entire lifetime.

The sidebar legend shows a colour swatch, label and live count for each species, the
three terrain types (read from the same config the world layer uses), and both ends of
the resource gradient.

### 14.5 Meters

FPS is an exponential moving average of `1/elapsed_ms` between `draw()` calls, not an
assumed rate. Steps/second is a 500 ms windowed count. These are independent: the
simulation advances on a `timestep_ms` accumulator, so step rate and frame rate differ
substantially.

### 14.6 Controls

| Key | Action |
|---|---|
| `Space` | pause |
| `Esc` | quit |
| `R` | reset populations, keep terrain and resources |
| `E` | regenerate terrain, keep entities and statistics |
| `M` | creator mode |
| `0` | fit view |
| `F1`–`F4` | toggle resources / grazers / predators / carcasses |
| `G` / `P` / `T` | same toggles, mnemonically |
| `F5` | toggle chart |
| wheel / middle-drag | cursor-anchored zoom / pan |
| `Ctrl` `+`/`-` | keyboard zoom |
| `1` `2` `3` | creator: rock / plains / forest brush |
| `+` / `-` | creator: brush size |
| `Ctrl`+click | creator: spawn grazer |
| `Shift`+click | creator: spawn predator |

## 15. Known defects

### 15.1 Predator extinction (unresolved)

Over 600 steps across three seeds, grazers and rabbits persist and cycle, but
**predators reach zero in every seed**, despite peaking higher (172–225) than before.

Measured cause: predators hunt successfully — 351 kills in 400 steps, and they are
faster than fleeing prey in all terrain — but sit chronically starved (mean energy 86 of
800, 276 deaths against 263 births). The margin is about **1.2 kills per offspring**.

The economics: a successful kill nets roughly +70 energy, while an unsuccessful chase
burns enough to be ruinous. Reducing `chase_cost_multiplier` 2.0 → 1.4 and
`consumption_divisor` 8 → 4 raised the peak but did not close the gap.

A significant part of the cause is a coupling introduced during rebalancing: to make
the reduced resource capacity survivable, rabbit `energy_from_resource` was cut 12 → 6.
Because a carcass is worth the prey's stored energy, that halved the value of every
kill and starved the predator population that depended on it.

Untested remedies: raise rabbit `energy_from_resource` back toward 10–12 (richer
carcasses, but partly undoes the rabbit-explosion fix); lower predator
`energy_per_step`; raise predator `max_energy`; instrument per-terrain chase success
before changing more numbers.

### 15.2 No topologically separated regions

Rock is passable, so the map interior is a single connected passable region in every
seed. Ridges slow movement rather than divide the map, and since forest is 62% of the
area, most pursuit happens in terrain where the predator is barely faster than the
prey.

### 15.3 Not implemented

- Inspect mode (`I`), follow mode (`F`), keyboard-driven config window (`C`)
- Line-of-sight / occlusion
- `timestep_ms` is honoured by the GUI accumulator but is **not** a discrete simulation
  parameter: nothing is normalised by it, so it changes how fast the same step sequence
  is emitted, not the dynamics
- Rabbits appear in `config.yaml` but in neither original specification document, so
  their behaviour follows the grazer pattern plus the migratory extension

### 15.4 Performance

`sim.step()` dominates the frame budget at roughly 90 ms for ~850 agents, which exceeds
the 100 ms `timestep_ms`. Rendering is not the bottleneck. The GUI reports ~10 fps at
those population sizes; interactive feel degrades as prey numbers grow.

## 16. Configuration reference

See `config.yaml` for the authoritative values and inline rationale. Fields are
filtered on load: keys the dataclasses do not declare are ignored, so configs stay
compatible across schema changes.

New behaviour is driven by: `migratory`, `moving_efficiency`, `grazing_efficiency`,
`graze_ratio_threshold`, `migrate_ratio_threshold`, `graze_energy_target`,
`wander_energy_threshold`, `movement_energy_cost`, `flee_speed_multiplier`,
`flee_cost_multiplier`, `grazing_move_multiplier`, `chase_speed_multiplier`,
`chase_cost_multiplier`, `resource_productivity`, `feed_radius`,
`min_regen_suppression`, `carcass_energy_fraction`, `consumption_divisor`.

## 17. CLI

```bash
python -m simulator                          # GUI
python -m simulator --headless --steps 1000 \
    --export output/run.json --seed 42       # headless, reproducible
```

Options: `--config`, `--headless`, `--steps`, `--export`, `--seed`. `--steps` overrides
`simulation.max_steps`. Environment generation and agent decisions share one seeded
generator, so a given `--seed` replays identically.

## 18. Tests

```bash
python -m pytest tests/ -q
```

`tests/test_simulation.py` holds 18 regression tests, each tied to a defect found
while reconciling the code with the documents: prey that consumed without depleting the
grid; prey unable to out-earn metabolism; predators sweeping past prey mid-chase; a
kill that paid out energy twice; regeneration that could halt permanently; cumulative
statistics that were never populated; carcass feeding order and shared-store depletion;
bounded distance wraparound; spatial index equivalence to brute force; and the
species-specific energy invariants (continuous grazers must out-earn metabolism on bare
ground; migratory species must not be able to fund travel).

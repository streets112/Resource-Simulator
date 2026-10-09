import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    convolve as nd_convolve,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from simulator.config import EnvironmentConfig

_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


class Environment:
    def __init__(self, config: EnvironmentConfig, width: int, height: int, seed: int | None = None):
        self.config = config
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        self.ocean_border = config.ocean_border_width

        # Terrain is stored as indices into self.terrain_names so that whole-map
        # operations (smoothing, expansion, masking) stay vectorised.
        self.terrain_names = list(config.terrain_types.keys())
        self._terrain_index = {name: i for i, name in enumerate(self.terrain_names)}
        self.terrain = np.zeros((height, width), dtype=np.int8)

        self.resources = np.zeros((height, width), dtype=np.float64)
        self.resource_capacity = np.zeros((height, width), dtype=np.float64)
        self.regen_rate = np.zeros((height, width), dtype=np.float64)
        self.passable = np.zeros((height, width), dtype=bool)
        self.mobility_prey = np.zeros((height, width), dtype=np.float64)
        self.mobility_predator = np.zeros((height, width), dtype=np.float64)

        # Fractional resource loss since the previous regeneration pass, smoothed.
        # Fresh grazing lights this up; a steadily depleted cell settles to ~0.
        self.depletion_rate = np.zeros((height, width), dtype=np.float64)

        # Number of prey feeding in each cell this step. Regeneration is suppressed
        # where a cell is being actively worked, so a herd camping on ground can
        # strip it; a lone animal passing through does not freeze the ground.
        self.occupied = np.zeros((height, width), dtype=np.int32)

        # Scent field for predator tracking. Deposited by prey, diffuses with wind,
        # decays over time. Higher = fresher/more concentrated scent.
        self.scent = np.zeros((height, width), dtype=np.float64)

        self._generate_terrain_biomes()
        self._snapshot_resources()
        self._initialize_resources()

    # ------------------------------------------------------------------ terrain

    def _generate_terrain_biomes(self) -> None:
        names = self.terrain_names

        base_noise = self._fractal(octaves=5, persistence=0.5, scale=0.015)
        ridge_noise = self._ridge(octaves=4, persistence=0.5, scale=0.05)
        forest_noise = self._fractal(octaves=4, persistence=0.6, scale=0.01)
        meadow_noise = self._fractal(octaves=4, persistence=0.55, scale=0.02)
        scrub_noise = self._fractal(octaves=4, persistence=0.5, scale=0.035)

        mountain_mask = self._build_ridges(ridge_noise)

        # Biomes are allocated sequentially: each takes its share of the
        # lowland, scored by its own noise field, and whatever is left
        # becomes plains. Forests expand organically into adjacent ground,
        # so their mask is finalised before the next biome claims cells.
        distribution = self.config.terrain_distribution
        remaining = ~mountain_mask
        remaining_cells = int(remaining.sum())

        mountain_distance = distance_transform_edt(~mountain_mask)
        mountain_proximity = 1.0 - np.clip(mountain_distance / 8.0, 0.0, 1.0)

        biome_plan = tuple(
            (name, noise, expands)
            for name, noise, expands in (
                ("forest", forest_noise, True),
                ("meadow", meadow_noise, False),
                ("scrub", scrub_noise, False),
            )
            if name in self._terrain_index
        )
        weights = {name: float(distribution.get(name, 0.0)) for name, _, _ in biome_plan}
        plains_weight = float(distribution.get("plains", 0.0))
        weight_sum = sum(weights.values()) + plains_weight

        masks = {name: np.zeros_like(remaining) for name, _, _ in biome_plan}
        if remaining_cells > 0 and weight_sum > 0:
            available = remaining.copy()
            available_cells = remaining_cells
            for name, noise, expands in biome_plan:
                wanted = int(round(remaining_cells * weights[name] / weight_sum))
                wanted = min(wanted, available_cells)
                if weights[name] <= 0 or wanted <= 0:
                    continue
                score = _normalize(
                    noise * 0.5 + (1.0 - base_noise) * 0.3 + mountain_proximity * 0.2
                )
                cutoff = np.quantile(score[available], 1.0 - wanted / available_cells)
                mask = available & (score >= cutoff)
                if expands:
                    mask = self._expand_forests(mask, ~mountain_mask)
                masks[name] = mask
                available &= ~mask
                available_cells = int(available.sum())
        elif remaining_cells > 0:
            # Degenerate distribution: no weights at all, everything forest.
            if "forest" in self._terrain_index:
                masks["forest"] = remaining.copy()

        def _paint(mask, name):
            if name in self._terrain_index:
                self.terrain[mask] = self._terrain_index[name]

        _paint(mountain_mask, "rock")
        for name, _, _ in biome_plan:
            _paint(masks[name], name)
        plains_mask = remaining & ~mountain_mask
        for name, _, _ in biome_plan:
            plains_mask &= ~masks[name]
        _paint(plains_mask, "plains")

        self.terrain = self._smooth_terrain(self.terrain, names, iterations=1)
        self._carve_mountain_passes(mountain_mask)
        self._apply_ocean_border()
        self.refresh_terrain_properties()

    def _build_ridges(self, score: np.ndarray) -> np.ndarray:
        """Long connected mountain ridges that partition the map into valleys.

        The coverage threshold is derived from terrain_distribution rather than
        being hardcoded, and a morphological closing bridges the small gaps that
        would otherwise leave a ridge as a chain of disconnected specks. Only the
        substantial components survive, so what remains are a few long spines
        rather than scattered rubble.
        """
        # Smooth with a majority filter erodes thin ridges and lets the majority
        # class expand, so aim slightly high and let the filter settle the rest.
        target_rock = float(self.config.terrain_distribution.get("rock", 0.06)) * 1.15
        target_rock = min(max(target_rock, 0.0), 0.9)

        if target_rock <= 0:
            return np.zeros_like(score, dtype=bool)
        if target_rock >= 1.0:
            return np.ones_like(score, dtype=bool)

        # Subtract a blurred copy so only sharp crests survive the threshold.
        # The raw ridge field has broad plateaus that also read as high, and
        # thresholding it directly yields one amorphous mass rather than ridges:
        # this cut p95 thickness from 20 cells to 6 and the largest blob from
        # 1155 cells to 230.
        sharpened = score - gaussian_filter(score, sigma=3.0)

        cutoff = np.quantile(sharpened, 1.0 - target_rock)
        mask = sharpened >= cutoff

        # Ridge thickness is governed by the threshold, not by morphology:
        # erode-then-dilate is an opening, which only strips thin protrusions and
        # leaves a wide ridge just as wide. Closing reconnects the one-cell gaps
        # that keep a crest from reading as dashes.
        structure = np.ones((3, 3), dtype=bool)
        mask = binary_closing(mask, structure=structure, iterations=1)
        mask = binary_opening(mask, structure=structure, iterations=1)

        # Keep only substantial systems: a ridge nobody can route around is
        # noise, a ridge spanning the map is what channels movement.
        labelled, count = label(mask, structure=np.ones((3, 3), dtype=int))
        if count == 0:
            return mask
        sizes = np.bincount(labelled.ravel())
        largest = int(sizes[1:].max()) if sizes.size > 1 else 0
        if largest == 0:
            return np.zeros_like(mask)
        keep = np.zeros(sizes.shape, dtype=bool)
        keep[1:] = sizes[1:] >= max(8, int(largest * 0.05))
        return keep[labelled]

    def _fractal(self, octaves: int, persistence: float, scale: float) -> np.ndarray:
        total = np.zeros((self.height, self.width), dtype=np.float64)
        amplitude = 1.0
        frequency = scale
        norm = 0.0
        for _ in range(octaves):
            total += self._perlin(frequency) * amplitude
            norm += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / norm

    def _ridge(self, octaves: int, persistence: float, scale: float) -> np.ndarray:
        total = np.zeros((self.height, self.width), dtype=np.float64)
        amplitude = 1.0
        frequency = scale
        norm = 0.0
        for _ in range(octaves):
            # A higher exponent sharpens the crest into a thin line instead of a
            # broad plateau, which is what produces striations rather than slabs.
            ridge = 1.0 - np.abs(self._perlin(frequency))
            total += (ridge**3) * amplitude
            norm += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / norm

    def _directional_bias(self) -> np.ndarray:
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        x = np.linspace(0.0, 1.0, self.width)
        y = np.linspace(0.0, 1.0, self.height)
        grid_x, grid_y = np.meshgrid(x, y)
        bias = (grid_x * np.cos(angle) + grid_y * np.sin(angle)) * 0.5 + 0.5
        bias = bias + np.sin(grid_x * 20.0 + self.rng.uniform(0, 2 * np.pi)) * 0.1
        bias = bias + np.cos(grid_y * 15.0 + self.rng.uniform(0, 2 * np.pi)) * 0.1
        return bias

    def _perlin(self, scale: float) -> np.ndarray:
        h, w = self.height, self.width
        grid_x = int(w * scale) + 1
        grid_y = int(h * scale) + 1

        gradients = self.rng.uniform(-1.0, 1.0, (grid_y, grid_x, 2))
        gradients /= np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-8

        x = np.linspace(0.0, w * scale, w)
        y = np.linspace(0.0, h * scale, h)
        mesh_x, mesh_y = np.meshgrid(x, y)

        x0 = np.floor(mesh_x).astype(int)
        y0 = np.floor(mesh_y).astype(int)
        x1 = (x0 + 1) % grid_x
        y1 = (y0 + 1) % grid_y

        sx = mesh_x - x0
        sy = mesh_y - y0
        u = sx * sx * (3.0 - 2.0 * sx)
        v = sy * sy * (3.0 - 2.0 * sy)

        def dot(ix, iy):
            gx = gradients[iy, ix, 0]
            gy = gradients[iy, ix, 1]
            return gx * (mesh_x - ix / scale) + gy * (mesh_y - iy / scale)

        n00, n10 = dot(x0, y0), dot(x1, y0)
        n01, n11 = dot(x0, y1), dot(x1, y1)
        ix0 = n00 + u * (n10 - n00)
        ix1 = n01 + u * (n11 - n01)
        return ix0 + v * (ix1 - ix0)

    def _thin_ridges(self, mask: np.ndarray) -> np.ndarray:
        neighbours = nd_convolve(mask.astype(np.float64), _NEIGHBOUR_KERNEL, mode="nearest")
        return mask & (neighbours >= 2)

    def _drop_small_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        labelled, count = label(mask, structure=np.ones((3, 3)))
        if count == 0:
            return mask
        sizes = np.bincount(labelled.ravel())
        keep = np.zeros(sizes.shape, dtype=bool)
        keep[1:] = sizes[1:] >= min_size
        return keep[labelled]

    def _expand_forests(self, forest_mask: np.ndarray, available: np.ndarray) -> np.ndarray:
        """Organic growth of forest into adjacent open ground near mountains."""
        mountain_distance = distance_transform_edt(~available)
        for _ in range(3):
            neighbours = nd_convolve(
                forest_mask.astype(np.float64), _NEIGHBOUR_KERNEL, mode="nearest"
            )
            grow = (neighbours >= 4) | ((neighbours >= 3) & (mountain_distance < 6))
            forest_mask = forest_mask | (grow & available)
        return forest_mask

    def _smooth_terrain(self, terrain: np.ndarray, names: list[str], iterations: int) -> np.ndarray:
        for _ in range(iterations):
            counts = np.stack(
                [
                    nd_convolve(
                        (terrain == idx).astype(np.float64), _NEIGHBOUR_KERNEL, mode="nearest"
                    )
                    for idx in range(len(names))
                ]
            )
            terrain = counts.argmax(axis=0).astype(np.int8)
        return terrain

    def _carve_mountain_passes(self, mountain_mask: np.ndarray) -> None:
        labelled, count = label(mountain_mask, structure=np.ones((3, 3)))
        for component in range(1, count + 1):
            region = labelled == component
            if region.sum() <= 200:
                continue
            dist = distance_transform_edt(region)
            y, x = np.unravel_index(int(np.argmin(dist)), dist.shape)
            if dist[y, x] < 8:
                self._carve_pass_at(int(y), int(x), width=2)

    def _carve_pass_at(self, cy: int, cx: int, width: int) -> None:
        h, w = self.height, self.width
        outer_width = width + 2
        reach = np.arange(-outer_width, outer_width + 1)
        grid_y, grid_x = np.meshgrid(reach, reach, indexing="ij")
        squared = grid_y**2 + grid_x**2
        inner = squared <= width**2
        outer = (squared <= outer_width**2) & ~inner

        plains = self._terrain_index["plains"]
        rock = self._terrain_index["rock"]

        ys, xs = cy + grid_y, cx + grid_x
        keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        self.terrain[ys[keep], xs[keep]] = plains

        ys, xs = cy + grid_y, cx + grid_x
        keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w) & outer
        replace = keep & (self.terrain[ys.clip(0, h - 1), xs.clip(0, w - 1)] == rock)
        self.terrain[ys[replace], xs[replace]] = plains

    def _perlin_1d(self, length: int, scale: float, seed_offset: int = 0) -> np.ndarray:
        """Generate 1D Perlin-like noise using simple value noise with smoothing."""
        # Generate base noise using hash
        noise = np.zeros(length)
        for i in range(length):
            # Simple hash-based noise
            val = self.rng.uniform(-1.0, 1.0)
            noise[i] = val
        # Smooth with moving average
        smoothed = np.zeros(length)
        for i in range(length):
            if i == 0:
                smoothed[i] = noise[i]
            elif i == length - 1:
                smoothed[i] = (noise[i-1] + noise[i]) * 0.5
            else:
                smoothed[i] = (noise[i-1] + noise[i] + noise[i+1]) / 3.0
        return smoothed

    def _apply_ocean_border(self) -> None:
        border = self.ocean_border
        if border <= 0:
            return
        ocean = self._terrain_index["ocean"]
        
        # Generate wavy border widths for each side
        variation = max(2, border)  # Larger variation for more organic look
        
        # Generate noise for each side using 1D perlin noise with different seeds
        # Top border: wave along x
        top_noise = self._perlin_1d(self.width, 0.03, 0) * variation + border
        # Bottom border: wave along x (different seed)
        bottom_noise = self._perlin_1d(self.width, 0.03, 1000) * variation + border
        # Left border: wave along y
        left_noise = self._perlin_1d(self.height, 0.03, 2000) * variation + border
        # Right border: wave along y (different seed)
        right_noise = self._perlin_1d(self.height, 0.05, 3000) * variation + border
        
        top_noise = np.clip(top_noise, border, border * 3).astype(int)
        bottom_noise = np.clip(bottom_noise, border, border * 3).astype(int)
        left_noise = np.clip(left_noise, border, border * 3).astype(int)
        right_noise = np.clip(right_noise, border, border * 3).astype(int)
        
        # Top border
        for x in range(self.width):
            w = int(top_noise[x])
            self.terrain[:w, x] = self._terrain_index["ocean"]
        
        # Bottom border
        for x in range(self.width):
            w = int(bottom_noise[x])
            self.terrain[-w:, x] = self._terrain_index["ocean"]
        
        # Left border
        for y in range(self.height):
            w = int(left_noise[y])
            self.terrain[y, :w] = self._terrain_index["ocean"]
        
        # Right border
        for y in range(self.height):
            w = int(right_noise[y])
            self.terrain[y, -w:] = self._terrain_index["ocean"]

    def refresh_terrain_properties(self) -> None:
        """Re-derive per-cell capacity/regen/passability from the current terrain and config."""
        self.resource_capacity.fill(0.0)
        self.regen_rate.fill(0.0)
        self.passable.fill(False)
        self.mobility_prey.fill(0.0)
        self.mobility_predator.fill(0.0)
        for name, terrain_cfg in self.config.terrain_types.items():
            mask = self.terrain == self._terrain_index[name]
            if not mask.any():
                continue
            self.resource_capacity[mask] = terrain_cfg.resource_capacity
            self.regen_rate[mask] = terrain_cfg.resource_regen_rate
            self.passable[mask] = terrain_cfg.passable
            self.mobility_prey[mask] = terrain_cfg.mobility_prey
            self.mobility_predator[mask] = terrain_cfg.mobility_predator
        np.clip(self.resources, 0.0, self.resource_capacity, out=self.resources)

    def set_terrain(self, y: int, x: int, name: str) -> None:
        """Creator-mode painting: update terrain and its derived properties in place."""
        if not self.in_bounds(y, x) or name not in self._terrain_index:
            return
        self.terrain[y, x] = self._terrain_index[name]
        terrain_cfg = self.config.terrain_types[name]
        self.resource_capacity[y, x] = terrain_cfg.resource_capacity
        self.regen_rate[y, x] = terrain_cfg.resource_regen_rate
        self.passable[y, x] = terrain_cfg.passable
        self.resources[y, x] = min(self.resources[y, x], terrain_cfg.resource_capacity)

    # ---------------------------------------------------------------- resources

    def _initialize_resources(self) -> None:
        self.resources = self.rng.uniform(0.3, 0.7) * self.resource_capacity
        self._snapshot_resources()

    def _snapshot_resources(self) -> None:
        self._post_regen = self.resources.copy()

    def step(self, occupied: np.ndarray | None = None, wind_dir: float = 0.0, wind_intensity: float = 0.0) -> None:
        if occupied is not None and occupied.shape == self.occupied.shape:
            self.occupied = occupied
        self._update_depletion_signal()
        self._regenerate()
        self._diffuse()
        self._update_scent(wind_dir, wind_intensity)
        self._snapshot_resources()

    def _add_scent(self, y: int, x: int, amount: float) -> None:
        """Add scent at a location. Called when prey moves through a cell."""
        if 0 <= y < self.height and 0 <= x < self.width:
            self.scent[y, x] = min(1.0, self.scent[y, x] + amount)

    def _update_scent(self, wind_dir: float, wind_intensity: float, dt: float = 1.0) -> None:
        """Update scent field: decay + advection (wind) + diffusion.
        
        Scent flows downwind from prey positions. Predators use gradient descent
        to follow the strongest scent concentration. At 20 cells distance,
        scent spreads to ~6 cells width with ~1/6 intensity.
        """
        # Decay: scent fades over time
        self.scent *= 0.99  # 1% decay per step
        
        # Advection: wind pushes scent downwind
        if wind_intensity > 0.01:
            wind_dx = float(np.cos(wind_dir)) * wind_intensity
            wind_dy = float(np.sin(wind_dir)) * wind_intensity
            
            # Use upwind scheme for advection (more stable)
            advected = np.zeros_like(self.scent)
            shift_y = int(round(wind_dy * 3))
            shift_x = int(round(wind_dx * 3))
            
            if shift_y >= 0:
                advected[shift_y:, :] = self.scent[:-shift_y if shift_y else None, :]
            else:
                advected[:shift_y] = self.scent[-shift_y:, :]
            
            if shift_x >= 0:
                advected[:, shift_x:] = advected[:, :-shift_x if shift_x else None]
            else:
                advected[:, :shift_x] = advected[:, -shift_x:]
            
            # Blend advected scent with original (partial advection)
            self.scent = 0.7 * self.scent + 0.3 * advected
        
        # Diffusion: spread scent to neighbors (Gaussian-like spread)
        # This creates the 20 cells = 6 cells width, 1/6 intensity effect
        laplacian = nd_convolve(self.scent, _LAPLACIAN_KERNEL * 0.05, mode="wrap")
        self.scent += laplacian
        
        np.clip(self.scent, 0.0, 1.0, out=self.scent)

    def _regenerate(self) -> None:
        deficit = self.resource_capacity - self.resources
        base_regen = self.regen_rate * deficit * self.config.regen_base_factor
        # Suppression is floored rather than allowed to reach zero: a stripped
        # neighbourhood slows regrowth to a trickle instead of halting it, so the
        # world can always recover instead of locking into a permanent dead state.
        suppression = np.clip(
            1.0 - self._neighbour_depletion() * self.config.depletion_suppression,
            self.config.min_regen_suppression,
            1.0,
        )
        if self.config.regen_blocked_by_grazing:
            # Ground being worked does not regrow, but only once enough prey are
            # on it to count as camping. A lone grazer passing through still
            # lets the cell recover; a herd settling in strips it for good.
            worked = self.occupied >= max(1, int(self.config.regen_block_threshold))
            suppression = np.where(worked, 0.0, suppression)
        self.resources += base_regen * suppression
        np.clip(self.resources, 0.0, self.resource_capacity, out=self.resources)

    def _neighbour_depletion(self) -> np.ndarray:
        cap_mask = self.resource_capacity > 0
        ratio = np.zeros_like(self.resources)
        np.divide(self.resources, self.resource_capacity, out=ratio, where=cap_mask)

        neighbour_sum = nd_convolve(ratio, _NEIGHBOUR_KERNEL, mode="wrap")
        neighbour_count = nd_convolve(
            cap_mask.astype(np.float64), _NEIGHBOUR_KERNEL, mode="wrap"
        )
        neighbour_avg = np.where(
            neighbour_count > 0, neighbour_sum / np.maximum(neighbour_count, 1e-8), 1.0
        )
        return np.clip(1.0 - neighbour_avg / self.config.depletion_threshold, 0.0, 1.0)

    def _diffuse(self) -> None:
        rate = self.config.resource_diffusion_rate
        if rate <= 0:
            return
        
        # Diffuse only within the same terrain type to prevent
        # resources from leaking across terrain boundaries (e.g., rock -> plains)
        for terrain_idx in range(len(self.terrain_names)):
            mask = (self.terrain == terrain_idx)
            if not mask.any():
                continue
            
            # Extract resources for this terrain type
            masked_resources = np.where(mask, self.resources, 0.0)
            
            # Apply diffusion only within this terrain type
            laplacian = nd_convolve(masked_resources, _LAPLACIAN_KERNEL * rate, mode="wrap")
            
            # Only apply diffusion where the terrain matches
            self.resources += np.where(mask, laplacian, 0.0)
        
        np.clip(self.resources, 0.0, self.resource_capacity, out=self.resources)

    def _update_depletion_signal(self) -> None:
        """Fractional loss accumulated since the previous regeneration pass.

        Regrowth is excluded (clipped at 0). Smoothing gives the signal a short
        memory so a cell that is merely *already* stripped does not keep
        attracting predators -- only fresh grazing does.
        """
        denominator = np.where(self.resource_capacity > 0, self.resource_capacity, 1.0)
        loss = np.clip((self._post_regen - self.resources) / denominator, 0.0, 1.0)
        smoothing = float(self.config.depletion_signal_smoothing)
        self.depletion_rate *= smoothing
        self.depletion_rate += (1.0 - smoothing) * loss

    def consume_resource(self, y: int, x: int, amount: float) -> float:
        if not self.in_bounds(y, x):
            return 0.0
        taken = min(amount, float(self.resources[y, x]))
        if taken <= 0:
            return 0.0
        self.resources[y, x] -= taken
        return taken

    # ----------------------------------------------------------------- queries

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def clamp(self, y: int, x: int) -> tuple[int, int]:
        return (
            int(min(max(y, 0), self.height - 1)),
            int(min(max(x, 0), self.width - 1)),
        )

    def get_resource_at(self, y: int, x: int) -> float:
        y, x = self.clamp(y, x)
        return float(self.resources[y, x])

    def get_resource_ratio(self, y: int, x: int) -> float:
        y, x = self.clamp(y, x)
        capacity = self.resource_capacity[y, x]
        return float(self.resources[y, x] / capacity) if capacity > 0 else 0.0

    def get_terrain_at(self, y: int, x: int) -> str:
        y, x = self.clamp(y, x)
        return self.terrain_names[int(self.terrain[y, x])]

    def get_terrain_color(self, y: int, x: int) -> tuple[int, int, int]:
        return self.config.terrain_types[self.get_terrain_at(y, x)].color

    def is_passable(self, y: int, x: int) -> bool:
        y, x = self.clamp(y, x)
        return bool(self.passable[y, x])

    def mobility(self, y: int, x: int, predator: bool) -> float:
        terrain = self.config.terrain_types[self.get_terrain_at(y, x)]
        return terrain.mobility_predator if predator else terrain.mobility_prey

    def productivity(self, y: int, x: int, species: str = "") -> float:
        """How much of a prey's post-metabolism surplus this terrain lets it bank.

        Species-specific overrides let a terrain favour one prey over
        another, which is what gives each species its own niche.
        """
        terrain = self.config.terrain_types[self.get_terrain_at(y, x)]
        value = {
            "grazer": terrain.grazer_productivity,
            "rabbit": terrain.rabbit_productivity,
        }.get(species)
        if value is None:
            value = terrain.resource_productivity
        return value

    def visibility(self, y: int, x: int, predator: bool) -> float:
        terrain = self.config.terrain_types[self.get_terrain_at(y, x)]
        return terrain.visibility_predator if predator else terrain.visibility_prey

    def effective_vision(self, y: int, x: int, base_radius: int, predator: bool) -> int:
        return max(1, int(base_radius * self.visibility(y, x, predator)))

    def total_resource(self) -> float:
        return float(self.resources.sum())

    def as_lists(self) -> tuple[list, list]:
        return (
            [[self.terrain_names[int(i)] for i in row] for row in self.terrain],
            self.resources.tolist(),
        )


def _normalize(array: np.ndarray) -> np.ndarray:
    lo = array.min()
    hi = array.max()
    if hi - lo < 1e-12:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)
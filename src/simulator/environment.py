import numpy as np
from simulator.config import EnvironmentConfig, TerrainConfig


class Environment:
    def __init__(self, config: EnvironmentConfig, width: int, height: int, seed: int | None = None):
        self.config = config
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        self.terrain = np.empty((height, width), dtype=object)
        self.resources = np.zeros((height, width), dtype=np.float32)
        self.resource_capacity = np.zeros((height, width), dtype=np.float32)
        self.regen_rate = np.zeros((height, width), dtype=np.float32)
        self.passable = np.zeros((height, width), dtype=bool)

        self._generate_terrain_biomes()
        self._initialize_resources()

    def _generate_terrain_biomes(self) -> None:
        terrain_names = list(self.config.terrain_types.keys())
        self.ocean_border = self.config.ocean_border_width

        base_noise = self._generate_fractal_noise(octaves=5, persistence=0.5, scale=0.015)
        ridge_noise = self._generate_ridge_noise(octaves=4, persistence=0.5, scale=0.02)
        forest_noise = self._generate_fractal_noise(octaves=4, persistence=0.6, scale=0.01)
        detail_noise = self._generate_fractal_noise(octaves=3, persistence=0.5, scale=0.05)

        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
        ridge_noise = (ridge_noise - ridge_noise.min()) / (ridge_noise.max() - ridge_noise.min())
        forest_noise = (forest_noise - forest_noise.min()) / (forest_noise.max() - forest_noise.min())
        detail_noise = (detail_noise - detail_noise.min()) / (detail_noise.max() - detail_noise.min())

        mountain_mask = self._create_mountain_ridges(ridge_noise, detail_noise)
        forest_mask = self._create_forest_patches(forest_noise, base_noise, mountain_mask)
        plains_mask = ~(mountain_mask | forest_mask)

        self.terrain[mountain_mask] = 'rock'
        self.terrain[forest_mask] = 'forest'
        self.terrain[plains_mask] = 'plains'

        self.terrain = self._smooth_terrain(self.terrain, terrain_names, iterations=1)

        self._carve_mountain_passes(mountain_mask)

        # Add ocean border around the map
        self._add_ocean_border()

        for name, tcfg in self.config.terrain_types.items():
            mask = self.terrain == name
            self.resource_capacity[mask] = tcfg.resource_capacity
            self.regen_rate[mask] = tcfg.resource_regen_rate
            self.passable[mask] = tcfg.passable

    def _generate_fractal_noise(self, octaves: int = 4, persistence: float = 0.5, scale: float = 0.02) -> np.ndarray:
        h, w = self.height, self.width
        noise = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0

        for _ in range(octaves):
            perlin = self._perlin_noise_2d(h, w, frequency)
            noise += perlin * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= 2

        return noise / max_amplitude

    def _generate_ridge_noise(self, octaves: int = 4, persistence: float = 0.5, scale: float = 0.02) -> np.ndarray:
        h, w = self.height, self.width
        noise = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0

        for _ in range(octaves):
            perlin = self._perlin_noise_2d(h, w, frequency)
            ridges = 1.0 - np.abs(perlin)
            ridges = ridges * ridges
            noise += ridges * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= 2

        return noise / max_amplitude

    def _create_mountain_ridges(self, ridge_noise: np.ndarray, detail_noise: np.ndarray) -> np.ndarray:
        h, w = self.height, self.width

        directional_bias = self._generate_directional_bias()

        mountain_score = ridge_noise * 0.7 + detail_noise * 0.2 + directional_bias * 0.1

        mountain_score = (mountain_score - mountain_score.min()) / (mountain_score.max() - mountain_score.min())

        mountain_threshold = 0.38
        mountain_mask = mountain_score > mountain_threshold

        mountain_mask = self._thin_ridges(mountain_score, mountain_mask)

        mountain_mask = self._connect_ridge_segments(mountain_mask)

        return mountain_mask

    def _generate_directional_bias(self) -> np.ndarray:
        h, w = self.height, self.width
        angle = self.rng.uniform(0, 2 * np.pi)
        dir_x = np.cos(angle)
        dir_y = np.sin(angle)

        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)

        bias = (X * dir_x + Y * dir_y) * 0.5 + 0.5

        wave_x = np.sin(X * 20 + self.rng.uniform(0, 2*np.pi)) * 0.1
        wave_y = np.cos(Y * 15 + self.rng.uniform(0, 2*np.pi)) * 0.1

        return bias + wave_x + wave_y

    def _thin_ridges(self, mountain_score: np.ndarray, mountain_mask: np.ndarray) -> np.ndarray:
        h, w = mountain_mask.shape
        thinned = mountain_mask.copy()

        for y in range(h):
            for x in range(w):
                if mountain_mask[y, x]:
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny = (y + dy) % h
                            nx = (x + dx) % w
                            if mountain_mask[ny, nx]:
                                neighbors += 1
                    if neighbors < 2:
                        thinned[y, x] = False

        return thinned

    def _connect_ridge_segments(self, mountain_mask: np.ndarray) -> np.ndarray:
        h, w = mountain_mask.shape
        connected = mountain_mask.copy()

        from scipy.ndimage import label
        labeled, num_features = label(mountain_mask, structure=[[1,1,1],[1,1,1],[1,1,1]])

        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() < 5:
                connected[component] = False

        return connected

    def _create_forest_patches(self, forest_noise: np.ndarray, base_noise: np.ndarray, mountain_mask: np.ndarray) -> np.ndarray:
        h, w = self.height, self.width

        mountain_distance = self._compute_mountain_distance(mountain_mask)
        mountain_proximity = 1.0 - np.clip(mountain_distance / 8.0, 0, 1)

        forest_score = forest_noise * 0.5 + (1 - base_noise) * 0.3 + mountain_proximity * 0.2

        forest_score = (forest_score - forest_score.min()) / (forest_score.max() - forest_score.min())

        forest_threshold = 0.55
        forest_mask = (forest_score > forest_threshold) & ~mountain_mask

        forest_mask = self._expand_forests_natural(forest_mask, mountain_mask, iterations=3)

        return forest_mask

    def _compute_mountain_distance(self, mountain_mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mountain_mask)

    def _expand_forests_natural(self, forest_mask: np.ndarray, mountain_mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        h, w = forest_mask.shape
        expanded = forest_mask.copy()

        mountain_distance = self._compute_mountain_distance(mountain_mask)

        for _ in range(iterations):
            new_expanded = expanded.copy()
            for y in range(h):
                for x in range(w):
                    if not expanded[y, x] and not mountain_mask[y, x]:
                        forest_neighbors = 0
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny = (y + dy) % h
                                nx = (x + dx) % w
                                if expanded[ny, nx]:
                                    forest_neighbors += 1
                        dist = mountain_distance[y, x]
                        if forest_neighbors >= 4 or (forest_neighbors >= 3 and dist < 6):
                            new_expanded[y, x] = True
            expanded = new_expanded

        return expanded

    def _add_ocean_border(self) -> None:
        """Add ocean border around the map."""
        border = self.ocean_border
        
        # Top border
        self.terrain[:border, :] = 'ocean'
        # Bottom border
        self.terrain[-border:, :] = 'ocean'
        # Left border
        self.terrain[:, :border] = 'ocean'
        # Right border
        self.terrain[:, -border:] = 'ocean'

    def _carve_mountain_passes(self, mountain_mask: np.ndarray) -> None:
        h, w = self.height, self.width

        from scipy.ndimage import label, distance_transform_edt
        labeled, num_features = label(mountain_mask, structure=[[1,1,1],[1,1,1],[1,1,1]])

        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() > 200:
                dist = distance_transform_edt(component)
                thinnest = np.unravel_index(np.argmin(dist), dist.shape)
                y, x = thinnest

                if dist[y, x] < 8:
                    self._carve_pass_at(y, x, mountain_mask, width=2)

    def _carve_pass_at(self, cy: int, cx: int, mountain_mask: np.ndarray, width: int = 2) -> None:
        h, w = self.height, self.width

        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                if dy*dy + dx*dx <= width*width:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        self.terrain[ny, nx] = 'plains'

        extended_width = width + 2
        for dy in range(-extended_width, extended_width + 1):
            for dx in range(-extended_width, extended_width + 1):
                if width*width < dy*dy + dx*dx <= extended_width*extended_width:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if self.terrain[ny, nx] == 'rock':
                            self.terrain[ny, nx] = 'plains'

    def _perlin_noise_2d(self, h: int, w: int, scale: float) -> np.ndarray:
        grid_x = int(w * scale) + 1
        grid_y = int(h * scale) + 1

        gradients = self.rng.uniform(-1, 1, (grid_y, grid_x, 2))
        norms = np.linalg.norm(gradients, axis=2, keepdims=True)
        gradients = gradients / (norms + 1e-8)

        x = np.linspace(0, w * scale, w)
        y = np.linspace(0, h * scale, h)
        X, Y = np.meshgrid(x, y)

        x0 = np.floor(X).astype(int)
        y0 = np.floor(Y).astype(int)
        x1 = (x0 + 1) % grid_x
        y1 = (y0 + 1) % grid_y

        sx = X - x0
        sy = Y - y0

        u = sx * sx * (3 - 2 * sx)
        v = sy * sy * (3 - 2 * sy)

        def dot_grad(ix, iy):
            gx, gy = gradients[iy, ix, 0], gradients[iy, ix, 1]
            dx = X - ix / scale
            dy = Y - iy / scale
            return gx * dx + gy * dy

        n00 = dot_grad(x0, y0)
        n10 = dot_grad(x1, y0)
        n01 = dot_grad(x0, y1)
        n11 = dot_grad(x1, y1)

        ix0 = n00 + u * (n10 - n00)
        ix1 = n01 + u * (n11 - n01)

        return ix0 + v * (ix1 - ix0)

    def _smooth_terrain(self, terrain: np.ndarray, terrain_names: list, iterations: int = 2) -> np.ndarray:
        h, w = terrain.shape
        for _ in range(iterations):
            new_terrain = terrain.copy()
            for y in range(h):
                for x in range(w):
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny = (y + dy) % h
                            nx = (x + dx) % w
                            if terrain[ny, nx] is not None:
                                neighbors.append(terrain[ny, nx])
                    if neighbors:
                        unique, counts = np.unique(neighbors, return_counts=True)
                        new_terrain[y, x] = unique[np.argmax(counts)]
            terrain = new_terrain
        return terrain

    def _initialize_resources(self) -> None:
        self.resources = self.rng.uniform(0.3, 0.7) * self.resource_capacity

    def step(self) -> None:
        self._regenerate()
        self._diffuse()

    def _regenerate(self) -> None:
        neighbor_depletion = self._compute_neighbor_depletion()
        deficit = self.resource_capacity - self.resources
        base_regen = self.regen_rate * deficit * 0.025
        suppression = np.clip(1.0 - neighbor_depletion * 2.2, 0.0, 1.0)
        regen = base_regen * suppression
        self.resources += regen
        np.clip(self.resources, 0, self.resource_capacity, out=self.resources)

    def _compute_neighbor_depletion(self) -> np.ndarray:
        ratio = np.zeros_like(self.resources)
        cap_mask = self.resource_capacity > 0
        ratio[cap_mask] = self.resources[cap_mask] / self.resource_capacity[cap_mask]

        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0
        from scipy.signal import convolve2d
        neighbor_sum = convolve2d(ratio, kernel, mode="same", boundary="wrap")
        neighbor_count = convolve2d(cap_mask.astype(np.float32), kernel, mode="same", boundary="wrap")
        neighbor_avg = np.where(neighbor_count > 0, neighbor_sum / (neighbor_count + 1e-8), 1.0)

        depletion = np.clip(1.0 - neighbor_avg / 0.6, 0, 1)
        return depletion

    def _diffuse(self) -> None:
        if self.config.resource_diffusion_rate <= 0:
            return
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) * self.config.resource_diffusion_rate
        from scipy.signal import convolve2d
        diff = convolve2d(self.resources, kernel, mode="same", boundary="wrap")
        self.resources += diff
        np.clip(self.resources, 0, self.resource_capacity, out=self.resources)

    def consume_resource(self, y: int, x: int, amount: float) -> float:
        available = self.resources[y, x]
        taken = min(amount, available)
        self.resources[y, x] -= taken
        return taken

    def get_resource_at(self, y: int, x: int) -> float:
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        return self.resources[y, x]

    def get_terrain_at(self, y: int, x: int) -> str:
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        return self.terrain[y, x]

    def is_passable(self, y: int, x: int) -> bool:
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        return self.passable[y, x]

    def get_terrain_color(self, y: int, x: int) -> tuple[int, int, int]:
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        t = self.terrain[y, x]
        return self.config.terrain_types[t].color

    def get_resource_ratio(self, y: int, x: int) -> float:
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        cap = self.resource_capacity[y, x]
        return self.resources[y, x] / cap if cap > 0 else 0.0

    def get_mobility_prey(self, terrain: str) -> float:
        return self.config.terrain_types[terrain].mobility_prey

    def get_mobility_predator(self, terrain: str) -> float:
        return self.config.terrain_types[terrain].mobility_predator

    def get_visibility_prey(self, terrain: str) -> float:
        return self.config.terrain_types[terrain].visibility_prey

    def get_visibility_predator(self, terrain: str) -> float:
        return self.config.terrain_types[terrain].visibility_predator

    def get_effective_vision_prey(self, y: int, x: int, base_radius: int) -> int:
        terrain = self.get_terrain_at(y, x)
        visibility = self.get_visibility_prey(terrain)
        return max(1, int(base_radius * visibility))

    def get_effective_vision_predator(self, y: int, x: int, base_radius: int) -> int:
        terrain = self.get_terrain_at(y, x)
        visibility = self.get_visibility_predator(terrain)
        return max(1, int(base_radius * visibility))

    def can_see_through(self, from_y: int, from_x: int, to_y: int, to_x: int, is_predator: bool) -> bool:
        from_terrain = self.get_terrain_at(from_y, from_x)
        to_terrain = self.get_terrain_at(to_y, to_x)
        
        if from_terrain == 'rock':
            return True
        
        if is_predator:
            return self.get_visibility_predator(from_terrain) > 0.5
        else:
            return self.get_visibility_prey(from_terrain) > 0.5
import math

import numpy as np

try:  # pragma: no cover - import guard for headless environments
    import pygame

    PYGAME_AVAILABLE = True
except Exception:  # pragma: no cover - any SDL/import failure must stay non-fatal
    pygame = None  # type: ignore[assignment]
    PYGAME_AVAILABLE = False

from simulator.config import Config, VisualizationConfig
from simulator.simulation import Simulation


def _clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else (high if value > high else value)


def _probe_frombuffer_aliasing() -> bool:
    """Does ``pygame.image.frombuffer`` share memory with a writable numpy array?

    It does on pygame-ce 2.5.x, which lets the cached world layer be patched in
    place. On a build that copies instead we fall back to an explicit bytes copy.
    """
    if not PYGAME_AVAILABLE:
        return False
    probe = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        surface = pygame.image.frombuffer(probe, (4, 4), "RGB")
    except (pygame.error, ValueError, TypeError):  # pragma: no cover - driver dependent
        return False
    probe[1, 2] = (11, 22, 33)
    return surface.get_at((2, 1))[:3] == (11, 22, 33)


class PopulationChart:
    """Rolling population history for all three species."""

    def __init__(self, x: int, y: int, width: int, height: int, max_history: int = 500):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        # Built lazily so the module stays constructible without pygame.
        self.rect = None
        if PYGAME_AVAILABLE:
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.max_history = max_history
        self.grazer_history: list[int] = []
        self.rabbit_history: list[int] = []
        self.predator_history: list[int] = []
        self.visible = True
        self.show_grazers = True
        self.show_rabbits = True
        self.show_predators = True
        self._title_surface = None
        self._peak_surface = None
        self._peak_value: int | None = None

    def set_bounds(self, x: int, y: int, width: int, height: int) -> None:
        self.x, self.y = int(x), int(y)
        self.width, self.height = max(1, int(width)), max(1, int(height))
        if PYGAME_AVAILABLE:
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, grazers: int, rabbits: int, predators: int, paused: bool = False) -> None:
        if paused:
            return
        self.grazer_history.append(grazers)
        self.rabbit_history.append(rabbits)
        self.predator_history.append(predators)
        for history in (self.grazer_history, self.rabbit_history, self.predator_history):
            if len(history) > self.max_history:
                del history[: len(history) - self.max_history]

    def max_population(self) -> int:
        values = [0]
        if self.show_grazers:
            values.append(max(self.grazer_history, default=0))
        if self.show_rabbits:
            values.append(max(self.rabbit_history, default=0))
        if self.show_predators:
            values.append(max(self.predator_history, default=0))
        return max(values) or 1

    def draw(self, screen, font, small_font, colors) -> None:
        if not self.visible or screen is None or font is None:
            return
        rect = self.rect if self.rect is not None else pygame.Rect(
            self.x, self.y, self.width, self.height
        )
        if rect.width <= 0 or rect.height <= 0:
            return
        pygame.draw.rect(screen, (18, 18, 28), rect)
        pygame.draw.rect(screen, (70, 70, 90), rect, 1)

        # The title is static, so rendering it every frame is pure waste.
        if self._title_surface is None:
            self._title_surface = font.render("Populations", True, (220, 220, 230))
        screen.blit(self._title_surface, (rect.x + 8, rect.y + 4))

        plot = pygame.Rect(rect.x + 34, rect.y + 26, rect.width - 44, rect.height - 60)
        if plot.width <= 0 or plot.height <= 0:
            return
        pygame.draw.rect(screen, (10, 10, 16), plot)

        peak = self.max_population()
        series = (
            (self.grazer_history, colors.get("grazer", (255, 255, 0)), self.show_grazers),
            (self.rabbit_history, colors.get("rabbit", (255, 0, 255)), self.show_rabbits),
            (self.predator_history, colors.get("predator", (255, 100, 0)), self.show_predators),
        )
        span = max((len(h) for h, _, _ in series), default=0)
        if span < 2:
            return

        for history, color, show in series:
            if not show or len(history) < 2:
                continue
            points = []
            for i, value in enumerate(history):
                px = plot.x + int((i / (span - 1)) * (plot.width - 1))
                py = plot.bottom - int((value / peak) * (plot.height - 1))
                points.append((px, py))
            pygame.draw.lines(screen, color, False, points, 1)

        if small_font is not None:
            if self._peak_value != peak:
                self._peak_value = peak
                self._peak_surface = small_font.render(str(peak), True, (170, 170, 190))
            if self._peak_surface is not None:
                screen.blit(self._peak_surface, (rect.x + 4, plot.y - 4))


class MainRenderer:
    """World view with per-cell subpixel overlays, entity layers, zoom/pan and creator mode.

    The terrain+resource field is baked once into an offscreen surface at native
    resolution (``cell_size * map_width`` x ``cell_size * map_height``) and blitted
    per frame, so a 200x200 map costs one blit instead of ~80k ``draw.rect`` calls.
    The baked layer is rebuilt only when the world actually changes.
    """

    MIN_ZOOM = 0.25
    MAX_ZOOM = 4.0
    ZOOM_STEP = 1.15
    BRUSHES = ("rock", "plains", "forest")

    # Zoomed-in full-map scaling is cached until the zoom changes. Beyond this
    # pixel budget the scaled copy would be huge, so only the visible window is
    # resampled per frame instead.
    SCALE_CACHE_PIXEL_BUDGET = 16_777_216

    # Below this many on-screen pixels per cell a grid is unreadable noise, and
    # drawing one line per cell costs far more than it communicates.
    GRID_MIN_CELL_PX = 6.0
    GRID_MIN_ZOOM = 0.75

    FPS_SMOOTHING = 0.1
    RATE_WINDOW_MS = 500

    def __init__(self, config: Config):
        self.config = config
        self.vis = config.visualization
        self.sim: Simulation | None = None
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        self.zoom = 1.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.dirty = True
        # Forces a rebuild of the cached terrain+resource surface.
        self.dirty_world = True

        self.show_resources = True
        self.show_grazers = True
        self.show_predators = True
        self.show_carcasses = True
        self.show_grid = self.vis.show_grid

        self.creator_mode = False
        self.paused = False
        self.brush_index = 0
        self.brush_size = 3
        self.hover: tuple[int, int] | None = None
        self._panning = False
        self._last_mouse = (0, 0)

        self.chart = PopulationChart(10, 10, 400, 260)
        self.fps = self.vis.fps or 60

        # Actual drawable area; kept in sync with the window on resize.
        self.view_w = max(1, int(self.vis.window_width))
        self.view_h = max(1, int(self.vis.window_height))

        # --- cached terrain+resource layer -----------------------------------
        self._base_pixels: np.ndarray | None = None
        self._base_surface = None
        self._base_size: tuple[int, int] | None = None
        self._base_env = None
        self._base_shape: tuple[int, int] | None = None
        self._base_cell = 0
        self._base_step: int | None = None
        self._base_resources = True
        self._cell_colors: np.ndarray | None = None
        self._frombuffer_aliases = False

        # --- cached scaled layer --------------------------------------------
        self._scaled_surface = None
        self._scaled_size: tuple[int, int] | None = None
        self._scale_key: tuple | None = None

        # --- rate meters ------------------------------------------------------
        self.fps_ema = 0.0
        self.step_rate = 0.0
        self._last_frame_ticks: int | None = None
        self._rate_last_step: int | None = None
        self._rate_steps = 0
        self._rate_ms = 0.0

    # -------------------------------------------------------------- lifecycle

    def set_simulation(self, sim: Simulation) -> None:
        self.sim = sim
        self.dirty_world = True
        self._invalidate_scale_cache()
        if self.screen is None:
            self.initialize()

    def initialize(self) -> None:
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is unavailable; install pygame-ce")
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.vis.window_width, self.vis.window_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Resource Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 12)
        self._frombuffer_aliases = _probe_frombuffer_aliasing()
        self._sync_viewport()
        self.dirty_world = True
        self._invalidate_scale_cache()

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def tick(self) -> int:
        """Cap the frame rate and return elapsed milliseconds."""
        return int(self.clock.tick(self.fps)) if self.clock else 0

    # ----------------------------------------------------------------- camera

    def _sync_viewport(self) -> None:
        size = None
        if self.screen is not None:
            try:
                size = self.screen.get_size()
            except pygame.error:  # pragma: no cover - display went away
                size = None
        if size is None:
            size = (int(self.vis.window_width), int(self.vis.window_height))
        self.view_w = max(1, int(size[0]))
        self.view_h = max(1, int(size[1]))
        self.vis.window_width = self.view_w
        self.vis.window_height = self.view_h
        self._place_chart()

    def _place_chart(self) -> None:
        margin = 10
        width = _clamp_int(400, margin, max(margin, self.view_w - 2 * margin))
        height = _clamp_int(260, margin, max(margin, self.view_h - 2 * margin))
        x = _clamp_int(self.view_w - width - margin, margin, max(margin, self.view_w - margin))
        self.chart.set_bounds(x, margin, width, height)

    def _cell_pixels(self) -> float:
        """Screen pixels spanned by one world cell at the current zoom."""
        return max(1e-6, self.vis.cell_size * self.zoom)

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        # World coordinates are cell indices, so one cell must span
        # cell_size * zoom pixels; without that factor the whole map collapses
        # into a cell_size-scaled corner of the window.
        scale = self._cell_pixels()
        cx = self.view_w * 0.5
        cy = self.view_h * 0.5
        return (
            int((x - self.camera_x) * scale + cx),
            int((y - self.camera_y) * scale + cy),
        )

    def _screen_to_world(self, x: int, y: int) -> tuple[float, float]:
        scale = self._cell_pixels()
        cx = self.view_w * 0.5
        cy = self.view_h * 0.5
        return (
            (x - cx) / scale + self.camera_x,
            (y - cy) / scale + self.camera_y,
        )

    def _zoom_at(self, screen_x: int, screen_y: int, factor: float) -> None:
        before = self._screen_to_world(screen_x, screen_y)
        self.zoom = float(np.clip(self.zoom * factor, self.MIN_ZOOM, self.MAX_ZOOM))
        after = self._screen_to_world(screen_x, screen_y)
        self.camera_x += before[0] - after[0]
        self.camera_y += before[1] - after[1]

    # ----------------------------------------------------------------- events

    def handle_events(self, sim: Simulation) -> str | None:
        """Returns None to continue, or a control string such as 'quit'."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event)
            elif event.type == pygame.KEYDOWN:
                if self._handle_key(event, sim):
                    return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_left_click(event, sim)
                elif event.button == 2:
                    self._panning = True
                    self._last_mouse = event.pos
                elif event.button == 4:
                    self._zoom_at(*event.pos, self.ZOOM_STEP)
                    self.dirty = True
                elif event.button == 5:
                    self._zoom_at(*event.pos, 1.0 / self.ZOOM_STEP)
                    self.dirty = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                self._panning = False
            elif event.type == pygame.MOUSEMOTION:
                self.hover = self._cell_at(*event.pos)
                if self._panning:
                    scale = self._cell_pixels()
                    dx = (event.pos[0] - self._last_mouse[0]) / scale
                    dy = (event.pos[1] - self._last_mouse[1]) / scale
                    self.camera_x -= dx
                    self.camera_y -= dy
                    self._last_mouse = event.pos
                if self.creator_mode and self.hover is not None:
                    self._paint_at(*self.hover)
                self.dirty = True
        return None

    def _handle_resize(self, event) -> None:
        # Honour whatever SDL reports; SDL itself refuses sizes the OS cannot do.
        width = max(1, int(event.size[0]))
        height = max(1, int(event.size[1]))
        try:
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        except pygame.error:  # pragma: no cover - keep the existing surface
            pass
        self._sync_viewport()
        # The window changed, so anything derived from it must be re-derived.
        self.dirty_world = True
        self.dirty = True

    def _handle_key(self, event, sim: Simulation) -> bool:
        key = event.key
        mods = pygame.key.get_mods()
        ctrl = bool(mods & pygame.KMOD_CTRL)
        plus_keys = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
        minus_keys = (pygame.K_MINUS, pygame.K_KP_MINUS)
        # K_PLUS and K_EQUALS share a keycode, so `in` collapses the pair.
        wants_plus = key in plus_keys
        wants_minus = key in minus_keys

        if key == pygame.K_ESCAPE:
            return True
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r and not ctrl:
            sim.reset()
            self.dirty_world = True
        elif key == pygame.K_e:
            sim.reload_environment()
            self.dirty_world = True
        elif key == pygame.K_F1:
            self.show_resources = not self.show_resources
            self.dirty_world = True
        elif key == pygame.K_F2:
            self.show_grazers = not self.show_grazers
        elif key == pygame.K_F3:
            self.show_predators = not self.show_predators
        elif key == pygame.K_F4:
            self.show_carcasses = not self.show_carcasses
        elif key == pygame.K_g:
            self.show_grazers = not self.show_grazers
        elif key == pygame.K_p:
            self.show_predators = not self.show_predators
        elif key == pygame.K_t:
            self.show_carcasses = not self.show_carcasses
        elif key == pygame.K_0:
            self.zoom, self.camera_x, self.camera_y = 1.0, 0.0, 0.0
        elif wants_plus:
            # ctrl disambiguates keyboard zoom from the creator brush size.
            if ctrl:
                self._zoom_at(*pygame.mouse.get_pos(), self.ZOOM_STEP)
            elif self.creator_mode:
                self.brush_size = min(10, self.brush_size + 1)
        elif wants_minus:
            if ctrl:
                self._zoom_at(*pygame.mouse.get_pos(), 1.0 / self.ZOOM_STEP)
            elif self.creator_mode:
                self.brush_size = max(1, self.brush_size - 1)
        elif key == pygame.K_F5:
            self.chart.visible = not self.chart.visible
        elif key == pygame.K_m:
            self.creator_mode = not self.creator_mode
        elif self.creator_mode and key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.brush_index = key - pygame.K_1
        self.dirty = True
        return False

    def _handle_left_click(self, event, sim: Simulation) -> None:
        if not self.creator_mode:
            return
        cell = self._cell_at(*event.pos)
        if cell is None:
            return
        y, x = cell
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_SHIFT:
            sim.spawn_predator_at(y, x)
        elif mods & pygame.KMOD_CTRL:
            sim.spawn_grazer_at(y, x)

    def _paint_at(self, y: int, x: int) -> None:
        if self.sim is None or not (0 <= y < self.sim.map_height and 0 <= x < self.sim.map_width):
            return
        terrain = self.BRUSHES[self.brush_index]
        radius = self.brush_size
        height = self.sim.map_height
        width = self.sim.map_width
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if not 0 <= yy < height:
                continue
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx > radius * radius:
                    continue
                xx = x + dx
                if 0 <= xx < width:
                    self.sim.env.set_terrain(yy, xx, terrain)
        self.sim.env.refresh_terrain_properties()
        self.dirty_world = True

    def _cell_at(self, screen_x: int, screen_y: int) -> tuple[int, int] | None:
        if self.sim is None:
            return None
        wx, wy = self._screen_to_world(screen_x, screen_y)
        y, x = int(wy), int(wx)
        if 0 <= y < self.sim.map_height and 0 <= x < self.sim.map_width:
            return y, x
        return None

    # ------------------------------------------------------- cached world layer

    def _invalidate_scale_cache(self) -> None:
        self._scale_key = None
        self._scaled_surface = None
        self._scaled_size = None

    def _world_layer_state(self, sim: Simulation) -> tuple[bool, bool]:
        """Return (terrain_changed, resources_changed) for the cached layer.

        Terrain only moves when the world is explicitly invalidated (painting,
        E reload, R reset, resize, environment swap). The resource field, by
        contrast, moves on every simulation step, so it is tracked separately
        and patched in place without touching the rest of the surface.
        """
        cell = max(1, int(self.vis.cell_size))
        shape = (sim.map_height, sim.map_width)
        if (
            self.dirty_world
            or self._base_surface is None
            or self._base_env is not sim.env
            or self._base_cell != cell
            or self._base_shape != shape
            or self._base_resources != self.show_resources
        ):
            return True, True
        return False, self._base_step != sim.current_step

    def _terrain_lut(self, sim: Simulation) -> np.ndarray:
        env = sim.env
        names = list(getattr(env, "terrain_names", []) or [])
        count = max(1, len(names))
        lut = np.zeros((count, 3), dtype=np.uint8)
        terrain_types = getattr(env.config, "terrain_types", {}) or {}
        for i, name in enumerate(names):
            cfg = terrain_types.get(name)
            color = getattr(cfg, "color", None) if cfg is not None else None
            if color is None:
                color = self.vis.colors.get(name)
            if color is None:
                continue
            lut[i, 0] = int(color[0])
            lut[i, 1] = int(color[1])
            lut[i, 2] = int(color[2])
        return lut

    def _terrain_colors(self, sim: Simulation) -> np.ndarray:
        """(H, W, 3) uint8 terrain colour per cell."""
        lut = self._terrain_lut(sim)
        indices = np.clip(sim.env.terrain.astype(np.intp), 0, lut.shape[0] - 1)
        return lut[indices]

    def _resource_shade(self, sim: Simulation, colors: np.ndarray) -> np.ndarray | None:
        """(H, W, 3) uint8 resource tint per cell, or None where capacity is zero."""
        env = sim.env
        capacity = env.resource_capacity
        active = capacity > 0
        if not active.any():
            return None
        vis_colors = self.vis.colors
        low = np.asarray(vis_colors.get("resource_low", (30, 30, 30)), dtype=np.float32)
        high = np.asarray(vis_colors.get("resource_high", (50, 255, 50)), dtype=np.float32)
        ratio = np.zeros(env.resources.shape, dtype=np.float32)
        np.divide(env.resources, capacity, out=ratio, where=active)
        np.clip(ratio, 0.0, 1.0, out=ratio)
        shade = colors.astype(np.float32)
        mix = ratio[:, :, None] * (high - low) + low
        np.copyto(shade, mix, where=active[:, :, None])
        np.clip(shade, 0, 255, out=shade)
        return shade.astype(np.uint8)

    def _ensure_base_storage(self, pixel_width: int, pixel_height: int) -> None:
        """(Re)allocate the offscreen layer buffer and the Surface that wraps it.

        ``pygame.image.frombuffer`` aliases a writable numpy array instead of
        copying it, so the surface and ``self._base_pixels`` are the same memory
        and a quadrant patch is visible without rebuilding the surface.
        ``pygame.surfarray`` is deliberately unused: on pygame-ce 2.5.x
        ``array3d`` hands back a copy and ``make_surface`` ignores its input.
        """
        if self._base_pixels is not None and self._base_size == (pixel_width, pixel_height):
            return
        self._base_pixels = np.zeros((pixel_width, pixel_height, 3), dtype=np.uint8)
        self._base_size = (pixel_width, pixel_height)
        self._base_surface = self._surface_from_pixels(self._base_pixels)
        self._invalidate_scale_cache()

    def _cell_view(self, map_width: int, map_height: int) -> np.ndarray | None:
        pixels = self._base_pixels
        cell = self._base_cell
        if pixels is None or cell <= 0:
            return None
        if pixels.shape[0] != map_width * cell or pixels.shape[1] != map_height * cell:
            return None
        return pixels.reshape(map_width, cell, map_height, cell, 3)

    def _paint_world(self, colors: np.ndarray, shade, full: bool) -> None:
        """Write terrain (and optionally the resource quadrant) into the layer.

        A cell is ``cell_size`` pixels square: terrain fills all of it and the
        resource tint occupies only its top-right quadrant, the same sub-cell
        slot the old per-rect renderer drew. Terrain is painted as two
        half-plane writes rather than one 5-D broadcast, which is roughly 2x
        faster for a 1600x1600 layer.
        """
        map_height, map_width = colors.shape[0], colors.shape[1]
        view = self._cell_view(map_width, map_height)
        if view is None:
            return
        half = max(1, self._base_cell // 2)
        # Surfaces are x-major, so transpose the small (H, W, 3) cell arrays
        # rather than the multi-megabyte pixel buffer.
        if full:
            colors_xy = colors.transpose(1, 0, 2)[:, None, :, None, :]
            view[:, :, :, :half, :] = colors_xy
            view[:, :, :, half:, :] = colors_xy
        if shade is not None:
            view[:, half:, :, :half, :] = shade.transpose(1, 0, 2)[:, None, :, None, :]

    def _surface_from_pixels(self, pixels: np.ndarray):
        height, width = pixels.shape[0], pixels.shape[1]
        size = (width, height)
        if self._frombuffer_aliases:
            try:
                return pygame.image.frombuffer(pixels, size, "RGB")
            except (pygame.error, ValueError, TypeError):  # pragma: no cover
                self._frombuffer_aliases = False
        return pygame.image.frombuffer(pixels.tobytes(), size, "RGB")

    def _rebuild_base(self, sim: Simulation) -> None:
        env = sim.env
        cell = max(1, int(self.vis.cell_size))
        map_height, map_width = env.terrain.shape
        # Bookkeeping must be in place before painting: _cell_view sizes itself
        # from _base_cell / _base_size.
        self._base_env = env
        self._base_shape = (map_height, map_width)
        self._base_cell = cell
        self._base_step = sim.current_step
        self._base_resources = self.show_resources
        self._ensure_base_storage(map_width * cell, map_height * cell)

        colors = self._terrain_colors(sim)
        self._cell_colors = colors
        shade = self._resource_shade(sim, colors) if self.show_resources else None
        self._paint_world(colors, shade, full=True)

        self.dirty_world = False
        self._invalidate_scale_cache()

    def _update_resource_layer(self, sim: Simulation) -> None:
        colors = self._cell_colors
        if colors is None or not self.show_resources:
            return
        shade = self._resource_shade(sim, colors)
        if shade is not None:
            self._paint_world(colors, shade, full=False)
        self._base_step = sim.current_step
        if not self._frombuffer_aliases:  # pragma: no cover - defensive
            self._base_surface = self._surface_from_pixels(self._base_pixels)
            self._invalidate_scale_cache()

    def _ensure_world_layer(self, sim: Simulation) -> None:
        terrain_changed, resources_changed = self._world_layer_state(sim)
        if terrain_changed:
            self._rebuild_base(sim)
        elif resources_changed:
            self._update_resource_layer(sim)
        if self._base_surface is None:
            return

        if abs(self.zoom - 1.0) < 1e-9:
            self._scaled_surface = self._base_surface
            self._scaled_size = self._base_size
            self._scale_key = None
            return

        key = (self.zoom, self._base_size)
        if self._scale_key == key:
            return

        base_w, base_h = self._base_size
        target_w = max(1, int(round(base_w * self.zoom)))
        target_h = max(1, int(round(base_h * self.zoom)))
        if target_w * target_h <= self.SCALE_CACHE_PIXEL_BUDGET:
            self._scaled_surface = pygame.transform.smoothscale(
                self._base_surface, (target_w, target_h)
            )
            self._scaled_size = (target_w, target_h)
        else:
            # Too large to keep resident: resample only the visible window each
            # frame instead of holding a ~150 MB surface in memory.
            self._scaled_surface = None
            self._scaled_size = None
        self._scale_key = key

    # ----------------------------------------------------------------- render

    def draw(self, sim: Simulation, paused: bool = False) -> None:
        if self.screen is None:
            self.dirty = False
            return
        self._update_rate_meters(sim)

        self.screen.fill((10, 10, 20))
        self._draw_world(sim)
        # get_statistics() copies every agent list, so call it once per frame.
        stats = sim.get_statistics()
        self.chart.update(stats["grazers"], stats["rabbits"], stats["predators"], paused)
        self.chart.draw(self.screen, self.font, self.small_font, self.vis.colors)
        self._draw_hud(sim, paused, stats)
        if self.show_grid:
            self._draw_grid(sim)
        pygame.display.flip()
        self.dirty = False

    def _update_rate_meters(self, sim: Simulation) -> None:
        """EMA frame rate plus a windowed simulation step rate, both from real time."""
        now = pygame.time.get_ticks()
        previous = self._last_frame_ticks
        self._last_frame_ticks = now
        if previous is None:
            return
        elapsed_ms = now - previous
        if elapsed_ms <= 0:
            return

        instantaneous = 1000.0 / elapsed_ms
        if self.fps_ema <= 0.0:
            self.fps_ema = instantaneous
        else:
            self.fps_ema += self.FPS_SMOOTHING * (instantaneous - self.fps_ema)

        step = int(getattr(sim, "current_step", 0) or 0)
        if self._rate_last_step is None:
            self._rate_last_step = step
        delta = step - self._rate_last_step
        self._rate_last_step = step
        if delta > 0:  # a reset (R) rewinds the counter; ignore that spike
            self._rate_steps += delta
        self._rate_ms += elapsed_ms
        if self._rate_ms >= self.RATE_WINDOW_MS:
            self.step_rate = self._rate_steps * 1000.0 / self._rate_ms
            self._rate_steps = 0
            self._rate_ms = 0.0

    def _draw_world(self, sim: Simulation) -> None:
        self._ensure_world_layer(sim)
        self._blit_world_layer()
        self._draw_entity_markers(sim)

    def _blit_world_layer(self) -> None:
        """Blit the cached terrain+resource layer, cropped to the viewport.

        Base-surface pixel ``u`` maps to screen ``u * zoom + origin`` (the
        surface is already in ``cell_size`` units per cell), which is what makes
        the single blit agree with ``_world_to_screen`` for entity markers.
        """
        if self.screen is None or self._base_size is None or self._scaled_size is None:
            return
        cell = max(1, self._base_cell)
        zoom = self.zoom
        origin_x = self.view_w * 0.5 - self.camera_x * cell * zoom
        origin_y = self.view_h * 0.5 - self.camera_y * cell * zoom

        width, height = self._scaled_size
        u0 = _clamp_int(int(math.ceil(-origin_x)), 0, width)
        u1 = _clamp_int(int(math.ceil(self.view_w - origin_x)), 0, width)
        v0 = _clamp_int(int(math.ceil(-origin_y)), 0, height)
        v1 = _clamp_int(int(math.ceil(self.view_h - origin_y)), 0, height)
        if u1 <= u0 or v1 <= v0:
            return

        dest_x = int(round(origin_x)) + u0
        dest_y = int(round(origin_y)) + v0

        if self._scaled_surface is not None:
            source = pygame.Rect(u0, v0, u1 - u0, v1 - v0)
            self.screen.blit(self._scaled_surface, (dest_x, dest_y), source)
            return

        if self._base_surface is None:  # pragma: no cover - defensive
            return
        base_w, base_h = self._base_size
        bu0 = _clamp_int(int(u0 / zoom), 0, base_w)
        bu1 = _clamp_int(int(math.ceil(u1 / zoom)), 0, base_w)
        bv0 = _clamp_int(int(v0 / zoom), 0, base_h)
        bv1 = _clamp_int(int(math.ceil(v1 / zoom)), 0, base_h)
        if bu1 <= bu0 or bv1 <= bv0:
            return
        source_rect = pygame.Rect(bu0, bv0, bu1 - bu0, bv1 - bv0)
        dest_w = max(1, int(round(source_rect.width * zoom)))
        dest_h = max(1, int(round(source_rect.height * zoom)))
        crop = self._base_surface.subsurface(source_rect)
        resized = pygame.transform.smoothscale(crop, (dest_w, dest_h))
        self.screen.blit(resized, (dest_x, dest_y))

    def _visible_bounds(self, sim: Simulation):
        scale = self._cell_pixels()
        half_w = self.view_w / (2.0 * scale)
        half_h = self.view_h / (2.0 * scale)
        y0 = int(max(0, self.camera_y - half_h))
        y1 = int(min(sim.map_height, self.camera_y + half_h + 1))
        x0 = int(max(0, self.camera_x - half_w))
        x1 = int(min(sim.map_width, self.camera_x + half_w + 1))
        return y0, max(y0 + 1, y1), x0, max(x0 + 1, x1)

    def _draw_entity_markers(self, sim: Simulation) -> None:
        screen = self.screen
        if screen is None:
            return
        colors = self.vis.colors
        scale = self._cell_pixels()
        cam_x, cam_y = self.camera_x, self.camera_y
        centre_x = self.view_w * 0.5
        centre_y = self.view_h * 0.5
        view_w, view_h = self.view_w, self.view_h

        cell_px = max(1, int(round(scale)))
        half = max(1, cell_px // 2)
        draw_rect = pygame.draw.rect

        def blit(entity, color) -> None:
            sx = int((entity.x - cam_x) * scale + centre_x)
            if sx < -half or sx > view_w:
                return
            sy = int((entity.y - cam_y) * scale + centre_y)
            if sy < -half or sy > view_h:
                return
            draw_rect(screen, color, (sx, sy, half, half))

        if self.show_carcasses:
            carcass_color = colors.get("carcass", (255, 0, 0))
            for carcass in sim.carcasses:
                blit(carcass, carcass_color)
        if self.show_predators:
            predator_color = colors.get("predator", (255, 100, 0))
            for predator in sim.predators:
                blit(predator, predator_color)
        if self.show_grazers:
            grazer_color = colors.get("grazer", (255, 255, 0))
            rabbit_color = colors.get("rabbit", (255, 0, 255))
            for grazer in sim.grazers:
                blit(grazer, grazer_color)
            for rabbit in sim.rabbits:
                blit(rabbit, rabbit_color)

        hover = self.hover
        if hover is not None:
            hover_y, hover_x = hover
            if 0 <= hover_y < sim.map_height and 0 <= hover_x < sim.map_width:
                sx = int((hover_x - cam_x) * scale + centre_x)
                sy = int((hover_y - cam_y) * scale + centre_y)
                draw_rect(screen, (255, 255, 255), (sx + half, sy + half, half, half))

    def _draw_grid(self, sim: Simulation) -> None:
        screen = self.screen
        if screen is None:
            return
        zoom = self.zoom
        if zoom < self.GRID_MIN_ZOOM:
            return
        scale = self._cell_pixels()
        if scale < self.GRID_MIN_CELL_PX:
            return

        y0, y1, x0, x1 = self._visible_bounds(sim)
        if y1 <= y0 or x1 <= x0:
            return

        # Only span the part of the map that is actually on screen.
        centre_x = self.view_w * 0.5
        centre_y = self.view_h * 0.5
        span_top = _clamp_int(int((y0 - self.camera_y) * scale + centre_y), 0, self.view_h)
        span_bottom = _clamp_int(int((y1 - self.camera_y) * scale + centre_y), 0, self.view_h)
        span_left = _clamp_int(int((x0 - self.camera_x) * scale + centre_x), 0, self.view_w)
        span_right = _clamp_int(int((x1 - self.camera_x) * scale + centre_x), 0, self.view_w)
        if span_bottom <= span_top or span_right <= span_left:
            return

        color = (40, 40, 55)
        draw_line = pygame.draw.line
        for x in range(x0, x1):
            sx = int((x - self.camera_x) * scale + centre_x)
            if -1 <= sx <= self.view_w:
                draw_line(screen, color, (sx, span_top), (sx, span_bottom))
        for y in range(y0, y1):
            sy = int((y - self.camera_y) * scale + centre_y)
            if -1 <= sy <= self.view_h:
                draw_line(screen, color, (span_left, sy), (span_right, sy))

    # --------------------------------------------------------------------- hud

    def _hud_lines(self, sim: Simulation, paused: bool, stats: dict | None = None) -> list[str]:
        if stats is None:
            stats = sim.get_statistics()
        flags = ""
        if paused:
            flags += "  [PAUSED]"
        if sim.ended:
            flags += "  [ENDED]"
        return [
            f"step {sim.current_step}{flags}",
            f"grazers {stats['grazers']}   rabbits {stats['rabbits']}"
            f"   predators {stats['predators']}   carcasses {stats['carcasses']}",
            f"resource {stats['total_resource']:.0f}",
            f"fps {int(round(self.fps_ema))}   steps/s {self.step_rate:.1f}"
            f"   timestep {self.config.simulation.timestep_ms}ms"
            f"  x{self.config.simulation.simulation_speed:.2f}"
            f"  zoom {self.zoom:.2f}",
            "[CREATOR] "
            f"brush={self.BRUSHES[self.brush_index]} size={self.brush_size}"
            "  [1/2/3] brush  [+/-] size  [ctrl+click] grazer  [shift+click] predator"
            if self.creator_mode
            else "space pause | r reset | e new terrain | m creator | f1-f4 layers",
        ]

    def _draw_hud(self, sim: Simulation, paused: bool, stats: dict | None = None) -> None:
        if not self.vis.show_stats or self.screen is None or self.font is None:
            return
        lines = self._hud_lines(sim, paused, stats)
        if not lines:
            return

        line_height = 18
        padding = 8
        margin = 4
        # default=0 covers an empty line list, which used to raise ValueError.
        text_width = max((self.font.size(text)[0] for text in lines), default=0)
        # Never let the panel overflow the window.
        panel_width = min(text_width + padding * 2, max(0, self.view_w - 2 * margin))
        panel_height = min(line_height * len(lines) + 10, max(0, self.view_h - 2 * margin))
        if panel_width <= 0 or panel_height <= 0:
            return

        panel = pygame.Rect(
            _clamp_int(margin, 0, max(0, self.view_w - panel_width)),
            _clamp_int(self.view_h - panel_height - margin, 0, max(0, self.view_h - panel_height)),
            panel_width,
            panel_height,
        )

        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, panel.topleft)
        for i, text in enumerate(lines):
            y = panel.y + 6 + i * line_height
            if y + line_height > panel.bottom:
                break
            surface = self.font.render(text, True, (225, 225, 235))
            x = panel.x + padding
            visible = min(surface.get_width(), panel.right - x)
            if visible > 0:
                self.screen.blit(surface, (x, y), pygame.Rect(0, 0, visible, surface.get_height()))
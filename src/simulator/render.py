import numpy as np
from simulator.config import Config, VisualizationConfig
from simulator.simulation import Simulation
from simulator.environment import Environment
from simulator.entities import Grazer, Predator, Rabbit

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


class PopulationChart:
    """Live population chart showing historical populations of all species."""
    
    def __init__(self, x: int, y: int, width: int, height: int, max_history: int = 500):
        self.rect = pygame.Rect(x, y, width, height)
        self.max_history = max_history
        self.grazer_history: list[int] = []
        self.rabbit_history: list[int] = []
        self.predator_history: list[int] = []
        self.visible = True
        self.show_grazers = True
        self.show_rabbits = True
        self.show_predators = True
        
    def update(self, grazer_count: int, rabbit_count: int, predator_count: int, paused: bool = False) -> None:
        """Add current population counts to history."""
        if not paused:
            self.grazer_history.append(grazer_count)
            self.rabbit_history.append(rabbit_count)
            self.predator_history.append(predator_count)

            # Keep history within max length
            if len(self.grazer_history) > self.max_history:
                self.grazer_history = self.grazer_history[-self.max_history:]
                self.rabbit_history = self.rabbit_history[-self.max_history:]
                self.predator_history = self.predator_history[-self.max_history:]

    def clear(self) -> None:
        """Clear all history."""
        self.grazer_history.clear()
        self.rabbit_history.clear()
        self.predator_history.clear()

    def toggle_visibility(self) -> None:
        self.visible = not self.visible

    def toggle_grazers(self) -> None:
        self.show_grazers = not self.show_grazers

    def toggle_rabbits(self) -> None:
        self.show_rabbits = not self.show_rabbits

    def toggle_predators(self) -> None:
        self.show_predators = not self.show_predators

    def get_max_population(self) -> int:
        """Get the maximum population across all species and history for auto-scaling."""
        max_pop = 0
        for history in [self.grazer_history, self.rabbit_history, self.predator_history]:
            if history:
                max_pop = max(max_pop, max(history))
        return max_pop

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        if not self.visible:
            return
        
        # Draw background
        pygame.draw.rect(screen, (20, 20, 30), self.rect)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 2)
        
        # Title
        title = font.render("Population Chart (F5 to toggle, G/R/P to toggle lines)", True, (255, 255, 100))
        screen.blit(title, (self.rect.x + 10, self.rect.y + 5))
        
        # Legend
        legend_y = self.rect.y + 30
        if self.show_grazers:
            pygame.draw.line(screen, (255, 255, 0), (self.rect.x + 15, legend_y + 5), (self.rect.x + 35, legend_y + 5), 3)
            text = font.render("Grazers (Deer)", True, (255, 255, 0))
            screen.blit(text, (self.rect.x + 40, legend_y))
        if self.show_rabbits:
            pygame.draw.line(screen, (255, 0, 255), (self.rect.x + 15, legend_y + 25), (self.rect.x + 35, legend_y + 25), 3)
            text = font.render("Rabbits", True, (255, 0, 255))
            screen.blit(text, (self.rect.x + 40, legend_y + 20))
        if self.show_predators:
            pygame.draw.line(screen, (255, 100, 0), (self.rect.x + 15, legend_y + 45), (self.rect.x + 35, legend_y + 45), 3)
            text = font.render("Predators", True, (255, 100, 0))
            screen.blit(text, (self.rect.x + 40, legend_y + 40))
        
        if not any([self.grazer_history, self.rabbit_history, self.predator_history]):
            return
        
        # Calculate chart area
        chart_x = self.rect.x + 10
        chart_y = self.rect.y + 80
        chart_width = self.rect.width - 20
        chart_height = self.rect.height - 100
        
        # Draw grid lines
        max_pop = self.get_max_population()
        if max_pop == 0:
            max_pop = 1
        
        # Horizontal grid lines (population levels)
        for i in range(5):
            y = chart_y + chart_height - int((i / 4) * chart_height)
            pygame.draw.line(screen, (50, 50, 60), (chart_x, y), (chart_x + chart_width, y), 1)
            # Population labels
            pop_label = int(max_pop * (i / 4))
            label = font.render(str(pop_label), True, (150, 150, 150))
            screen.blit(label, (chart_x - 40, y - 8))
        
        # Vertical grid lines (time steps)
        for i in range(6):
            x = chart_x + int((i / 5) * chart_width)
            pygame.draw.line(screen, (50, 50, 60), (x, chart_y), (x, chart_y + chart_height), 1)
            # Time labels
            if self.grazer_history:
                step = len(self.grazer_history) * (i / 5)
                label = font.render(f"{int(step):.0f}", True, (150, 150, 150))
                screen.blit(label, (x - 10, chart_y + chart_height + 5))
        
        # Draw population lines
        # Grazers (yellow)
        if self.show_grazers and len(self.grazer_history) > 1:
            points = []
            for i, count in enumerate(self.grazer_history):
                x = chart_x + int((i / (len(self.grazer_history) - 1)) * chart_width)
                y = chart_y + chart_height - int((count / max_pop) * chart_height)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(screen, (255, 255, 0), False, points, 2)
        
        # Rabbits (magenta)
        if self.show_rabbits and len(self.rabbit_history) > 1:
            points = []
            for i, count in enumerate(self.rabbit_history):
                x = chart_x + int((i / (len(self.rabbit_history) - 1)) * chart_width)
                y = chart_y + chart_height - int((count / max_pop) * chart_height)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(screen, (255, 0, 255), False, points, 2)
        
        # Predators (orange/red)
        if self.show_predators and len(self.predator_history) > 1:
            points = []
            for i, count in enumerate(self.predator_history):
                x = chart_x + int((i / (len(self.predator_history) - 1)) * chart_width)
                y = chart_y + chart_height - int((count / max_pop) * chart_height)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(screen, (255, 100, 0), False, points, 2)
        
        # Current population values
        info_y = self.rect.y + self.rect.height - 60
        if self.show_grazers and self.grazer_history:
            g = self.grazer_history[-1]
            text = small_font.render(f"Grazers: {g}", True, (255, 255, 0))
            screen.blit(text, (self.rect.x + 10, info_y))
        if self.show_rabbits and self.rabbit_history:
            r = self.rabbit_history[-1]
            text = small_font.render(f"Rabbits: {r}", True, (255, 0, 255))
            screen.blit(text, (self.rect.x + 150, info_y))
        if self.show_predators and self.predator_history:
            p = self.predator_history[-1]
            text = small_font.render(f"Predators: {p}", True, (255, 100, 0))
            screen.blit(text, (self.rect.x + 300, info_y))


class Slider:
    """A simple slider widget for adjusting values."""
    
    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float, initial_val: float, label: str = ""):
        self.rect = pygame.Rect(x, y, width, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = 10
        
    def get_handle_pos(self) -> int:
        """Get the x position of the slider handle based on current value."""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)
    
    def set_value_from_pos(self, x: int) -> None:
        """Set value from mouse x position."""
        ratio = max(0, min(1, (x - self.rect.x) / self.rect.width))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events. Returns True if value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            handle_x = self.get_handle_pos()
            handle_rect = pygame.Rect(handle_x - self.handle_radius, self.rect.centery - self.handle_radius, 
                                       self.handle_radius * 2, self.handle_radius * 2)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_value_from_pos(event.pos[0])
            return True
        return False
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the slider."""
        # Track
        pygame.draw.rect(screen, (60, 60, 80), self.rect, border_radius=5)
        # Fill
        fill_width = self.get_handle_pos() - self.rect.x
        if fill_width > 0:
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
            pygame.draw.rect(screen, (100, 150, 255), fill_rect, border_radius=5)
        # Handle
        handle_x = self.get_handle_pos()
        pygame.draw.circle(screen, (200, 200, 220), (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(screen, (100, 100, 120), (handle_x, self.rect.centery), self.handle_radius, 2)
        # Label and value
        label_text = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))


class ConfigWindow:
    """Configuration window for simulation parameters."""
    
    def __init__(self, x: int, y: int, width: int, height: int, config: Config):
        self.rect = pygame.Rect(x, y, width, height)
        self.config = config
        self.visible = False
        self.font = pygame.font.SysFont('consolas', 14)
        self.small_font = pygame.font.SysFont('consolas', 12)
        self.sliders = []
        self.checkboxes = []
        self._create_controls()
        
    def _create_controls(self) -> None:
        """Create slider and checkbox controls for config parameters."""
        y_offset = 30
        slider_width = self.rect.width - 40
        
        # Simulation speed slider
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            0.1, 5.0, self.config.simulation.simulation_speed, "Speed"
        ))
        y_offset += 50
        
        # Grazers
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            0, 200, self.config.grazer.initial_count, "Initial Grazers"
        ))
        y_offset += 50
        
        # Rabbits
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            0, 200, self.config.rabbit.initial_count, "Initial Rabbits"
        ))
        y_offset += 50
        
        # Predators
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            0, 100, self.config.predator.initial_count, "Initial Predators"
        ))
        y_offset += 50
        
        # Grass regrowth rate
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            0.001, 0.1, self.config.environment.grass_regrowth_rate, "Grass Regrowth"
        ))
        y_offset += 50
        
        # Grazer energy
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            10, 300, self.config.grazer.max_energy, "Grazer Energy"
        ))
        y_offset += 50
        
        # Rabbit energy
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            10, 300, self.config.rabbit.max_energy, "Rabbit Energy"
        ))
        y_offset += 50
        
        # Predator energy
        self.sliders.append(Slider(
            self.rect.x + 20, self.rect.y + y_offset, slider_width,
            10, 500, self.config.predator.max_energy, "Predator Energy"
        ))
        y_offset += 50
        
    def toggle_visibility(self) -> None:
        self.visible = not self.visible
        
    def handle_event(self, event: pygame.event.Event) -> None:
        if not self.visible:
            return
        for slider in self.sliders:
            slider.handle_event(event)
            
    def update_config(self) -> None:
        """Apply slider values to config."""
        self.config.simulation.simulation_speed = self.sliders[0].value
        self.config.grazer.initial_count = int(self.sliders[1].value)
        self.config.rabbit.initial_count = int(self.sliders[2].value)
        self.config.predator.initial_count = int(self.sliders[3].value)
        self.config.environment.grass_regrowth_rate = self.sliders[4].value
        self.config.grazer.max_energy = self.sliders[5].value
        self.config.rabbit.max_energy = self.sliders[6].value
        self.config.predator.max_energy = self.sliders[7].value
        
    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return
            
        # Background
        pygame.draw.rect(screen, (30, 30, 40), self.rect)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 2)
        
        # Title
        title = self.font.render("Configuration (C to close)", True, (255, 255, 100))
        screen.blit(title, (self.rect.x + 10, self.rect.y + 5))
        
        # Draw sliders
        for slider in self.sliders:
            slider.draw(screen, self.small_font)


class Renderer:
    """Base renderer class for the simulation."""
    
    def __init__(self, config: Config, vis_config: VisualizationConfig):
        self.config = config
        self.vis_config = vis_config
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.running = False
        self.sim = None
        self.config_obj = None
        
    def set_simulation(self, sim: Simulation, config: Config) -> None:
        """Set the simulation and config to render."""
        self.sim = sim
        self.config_obj = config
        if self.screen is None:
            self.initialize()
        
    def initialize(self) -> None:
        """Initialize pygame and create the display."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not available")
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.vis_config.window_width, self.vis_config.window_height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Resource Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.running = True
        
    def initialize(self) -> None:
        """Initialize pygame and create the display."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is not available. Install with: pip install pygame")
            
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.vis_config.window_width, self.vis_config.window_height)
        )
        pygame.display.set_caption("Resource Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('consolas', 16)
        self.small_font = pygame.font.SysFont('consolas', 12)
        
    def handle_events(self, simulation: Simulation) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    simulation.reset()
        return True
        
    def draw(self, simulation: Simulation) -> None:
        """Draw the simulation state. Override in subclasses."""
        raise NotImplementedError
        
    def run(self, simulation: Simulation) -> None:
        """Main render loop."""
        self.running = True
        self.initialize()
        
        while self.running:
            self.running = self.handle_events(simulation)
            if not self.running:
                break
                
            if not paused:
                simulation.step()
                
            self.draw(simulation)
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()


class MainRenderer(Renderer):
    """Main renderer with full visualization including entities, charts, and UI."""
    
    def __init__(self, config: Config, vis_config: VisualizationConfig):
        super().__init__(config, vis_config)
        self.population_chart = None
        self.config_window = None
        self.show_chart = True
        self.show_grid = True
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
    def initialize(self) -> None:
        super().initialize()
        
        # Create population chart
        chart_width = 400
        chart_height = 300
        self.population_chart = PopulationChart(
            self.vis_config.window_width - chart_width - 10,
            10,
            chart_width,
            chart_height
        )
        
        # Create config window
        self.config_window = ConfigWindow(
            10, 10, 350, 500, self.config
        )
        
    def handle_events(self, simulation: Simulation) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    simulation.reset()
                elif event.key == pygame.K_c:
                    self.config_window.toggle_visibility()
                elif event.key == pygame.K_F5:
                    self.population_chart.toggle_visibility()
                elif event.key == pygame.K_g:
                    self.population_chart.toggle_grazers()
                elif event.key == pygame.K_r:
                    self.population_chart.toggle_rabbits()
                elif event.key == pygame.K_p:
                    self.population_chart.toggle_predators()
                elif event.key == pygame.K_F1:
                    self.show_grid = not self.show_grid
                    
            # Handle config window events
            if self.config_window:
                self.config_window.handle_event(event)
                
        return True
        
    def _world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.camera_x) * self.zoom + self.vis_config.window_width / 2)
        screen_y = int((y - self.camera_y) * self.zoom + self.vis_config.window_height / 2)
        return screen_x, screen_y
        
    def _draw_environment(self, simulation: Simulation) -> None:
        """Draw the environment (grass)."""
        env = simulation.env
        if not self.show_grid:
            return
            
        cell_size = max(1, int(self.vis_config.cell_size * self.zoom))
        for y in range(env.height):
            for x in range(env.width):
                grass = env.resources[y, x]
                if grass > 0:
                    screen_x, screen_y = self._world_to_screen(x * self.vis_config.cell_size, y * self.vis_config.cell_size)
                    # Color based on grass amount
                    green_val = int(50 + (grass / env.resource_capacity[y, x]) * 200)
                    color = (0, green_val, 0)
                    rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
                    pygame.draw.rect(self.screen, color, rect)
                    
    def _draw_entities(self, simulation: Simulation) -> None:
        """Draw all entities."""
        for entity_list in [simulation.grazers, simulation.rabbits, simulation.predators]:
            for entity in entity_list:
                if not entity.alive:
                    continue
                    
                screen_x, screen_y = self._world_to_screen(entity.x, entity.y)
                radius = max(1, int(entity.size * self.zoom))
                
                if isinstance(entity, Grazer):
                    color = (255, 255, 0)  # Yellow
                elif isinstance(entity, Rabbit):
                    color = (255, 0, 255)  # Magenta
                elif isinstance(entity, Predator):
                    color = (255, 100, 0)  # Orange/Red
                else:
                    color = (255, 255, 255)
                    
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            # Draw energy bar above entity
            if self.zoom > 0.5:
                energy_ratio = entity.energy / entity.max_energy
                bar_width = radius * 2
                bar_height = 3
                bar_x = screen_x - radius
                bar_y = screen_y - radius - 6
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
                
    def _draw_ui(self, simulation: Simulation, paused: bool = False) -> None:
        """Draw UI elements (stats, controls)."""
        # Stats panel
        stats = simulation.get_statistics()
        panel_x = 10
        panel_y = 10
        panel_width = 250
        panel_height = 200
        
        # Semi-transparent background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (100, 100, 120), (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("Simulation Stats", True, (255, 255, 100))
        self.screen.blit(title, (panel_x + 10, panel_y + 5))
        
        # Stats lines
        lines = [
            f"Step: {simulation.current_step}",
            f"Paused: {'Yes' if paused else 'No'}",
            f"Speed: {self.config.simulation.simulation_speed:.1f}x",
            f"Grazers: {stats.get('grazers', 0)}",
            f"Rabbits: {stats.get('rabbits', 0)}",
            f"Predators: {stats.get('predators', 0)}",
            f"Total Grass: {stats.get('total_grass', 0):.0f}",
            f"Entities: {len(simulation.grazers) + len(simulation.rabbits) + len(simulation.predators)}",
        ]
        
        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (panel_x + 10, panel_y + 30 + i * 20))
            
        # Controls help
        help_lines = [
            "SPACE - Pause/Resume",
            "R - Reset",
            "C - Config Window",
            "F5 - Toggle Chart",
            "G/R/P - Toggle Chart Lines",
            "F1 - Toggle Grid",
            "ESC - Quit"
        ]
        
        help_y = panel_y + panel_height + 10
        for i, line in enumerate(help_lines):
            text = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(text, (panel_x + 10, help_y + i * 18))
            
    def draw(self, simulation: Simulation, paused: bool = False) -> None:
        """Draw the complete simulation."""
        # Clear screen
        self.screen.fill((10, 10, 20))
        
        # Draw environment
        self._draw_environment(simulation)
        
        # Draw entities
        self._draw_entities(simulation)
        
        # Draw UI
        self._draw_ui(simulation, paused)
        
        # Update and draw population chart
        stats = simulation.get_statistics()
        if self.population_chart:
            self.population_chart.update(
                stats.get('grazers', 0),
                stats.get('rabbits', 0),
                stats.get('predators', 0),
                paused
            )
            self.population_chart.draw(self.screen, self.font, self.small_font)
            
        # Draw config window
        if self.config_window:
            self.config_window.draw(self.screen)
            # Update config from sliders
            self.config_window.update_config()
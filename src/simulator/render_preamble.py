import numpy as np
from simulator.config import Config, VisualizationConfig
from simulator.simulation import Simulation
from simulator.environment import Environment
from simulator.entities import Grazer, Predator

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial, label, getter, setter, step=1):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.getter = getter
        self.setter = setter
        self.step = step
        self.dragging = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (60, 60, 60), self.rect)
        pygame.draw.rect(screen, (120, 120, 120), self.rect, 1)

        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(ratio * self.rect.width)
        handle_rect = pygame.Rect(handle_x - 4, self.rect.y - 2, 8, self.rect.height + 4)
        pygame.draw.rect(screen, (200, 200, 100), handle_rect)

        if isinstance(self.step, int) or (isinstance(self.step, float) and self.step >= 1):
            text = font.render(f"{self.label}: {self.value:.0f}", True, (220, 220, 220))
        else:
            text = font.render(f"{self.label}: {self.value:.2f}", True, (220, 220, 220))
        screen.blit(text, (self.rect.x, self.rect.y - 22))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_from_mouse(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_from_mouse(event.pos)

    def update_from_mouse(self, pos):
        ratio = (pos[0] - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        if self.step > 0:
            self.value = round(self.value / self.step) * self.step
            self.value = max(self.min_val, min(self.max_val, self.value))
        if self.setter:
            self.setter(self.value)


class ConfigWindow:
    def __init__(self, config, sim, screen_width, screen_height):
        self.config = config
        self.sim = sim
        self.font = pygame.font.Font(None, 16)
        self.title_font = pygame.font.Font(None, 20)
        self.section_font = pygame.font.Font(None, 18)
        self.sliders = []
        self.sections = {}
        self.visible = True
        self.rect = pygame.Rect(screen_width - 420, 10, 410, screen_height - 20)
        self.section_states = {
            'grazer': True,
            'predator': True,
            'terrain': True,
            'simulation': True,
        }
        self._create_sections()

    def _create_sections(self):
        y_offset = 35
        slider_width = 370
        x_start = self.rect.x + 20

        self.sections = {
            'grazer': {
                'label': 'GRAZER',
                'color': (100, 200, 100),
                'expanded': True,
                'sliders': [],
                'items': [
                    ("Energy/Step", self.config.grazer, 'energy_per_step', 1, 100, 1),
                    ("Energy/Resource", self.config.grazer, 'energy_from_resource', 1, 50, 1),
                    ("Repro Min Energy", self.config.grazer, 'reproduction_min_energy', 100, 1000, 10),
                    ("Repro Cost", self.config.grazer, 'reproduction_energy_cost', 0.1, 0.9, 0.05),
                    ("Gradient Weight", self.config.grazer, 'gradient_weight', 0.0, 1.0, 0.05),
                    ("Move Speed", self.config.grazer, 'move_speed', 1, 5, 1),
                    ("Vision Radius", self.config.grazer, 'vision_radius', 1, 15, 1),
                ]
            },
            'predator': {
                'label': 'PREDATOR',
                'color': (255, 165, 0),
                'expanded': True,
                'sliders': [],
                'items': [
                    ("Energy/Step", self.config.predator, 'energy_per_step', 1, 20, 1),
                    ("Energy/Prey", self.config.predator, 'energy_from_prey', 100, 1000, 50),
                    ("Repro Min Energy", self.config.predator, 'reproduction_min_energy', 100, 600, 10),
                    ("Repro Cost", self.config.predator, 'reproduction_energy_cost', 0.1, 0.9, 0.05),
                    ("Chase Radius", self.config.predator, 'chase_radius', 1, 20, 1),
                    ("Move Speed", self.config.predator, 'move_speed', 1, 5, 1),
                    ("Resource Sense", self.config.predator, 'resource_sense_radius', 1, 20, 1),
                ]
            },
            'terrain': {
                'label': 'TERRAIN',
                'color': (180, 180, 100),
                'expanded': True,
                'sliders': [],
                'items': [
                    ("Resource Diffusion", self.config.environment, 'resource_diffusion_rate', 0.0, 0.1, 0.01),
                    ("Plains Regen", self.config.environment.terrain_types['plains'], 'resource_regen_rate', 0.0, 2.0, 0.1),
                    ("Plains Capacity", self.config.environment.terrain_types['plains'], 'resource_capacity', 10, 200, 5),
                    ("Forest Regen", self.config.environment.terrain_types['forest'], 'resource_regen_rate', 0.0, 2.0, 0.1),
                    ("Forest Capacity", self.config.environment.terrain_types['forest'], 'resource_capacity', 10, 300, 5),
                    ("Rock Regen", self.config.environment.terrain_types['rock'], 'resource_regen_rate', 0.0, 1.0, 0.1),
                ]
            },
            'simulation': {
                'label': 'SIMULATION',
                'color': (100, 150, 255),
                'expanded': True,
                'sliders': [],
                'items': [
                    ("Timestep (ms)", self.config.simulation, 'timestep_ms', 10, 500, 10),
                    ("Grid Width", self.config.simulation, 'width', 50, 500, 10),
                    ("Grid Height", self.config.simulation, 'height', 50, 500, 10),
                ]
            },
        }

        for section_key, section in self.sections.items():
            section['rect'] = pygame.Rect(x_start, y_offset, slider_width + 20, 28)
            y_offset += 30

            for item in section['items']:
                label, obj, attr, min_v, max_v, step = item
                initial = getattr(obj, attr)
                slider = Slider(
                    x_start + 15, y_offset, slider_width, 18, min_v, max_v, initial, label,
                    lambda o=obj, a=attr: getattr(o, a),
                    lambda v, o=obj, a=attr, s=step: setattr(o, a, round(v / s) * s)
                )
                section['sliders'].append(slider)
                self.sliders.append(slider)
                y_offset += 24
            y_offset += 8

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.visible = False
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos):
                self.visible = False
                return False
            for section_key, section in self.sections.items():
                if section['rect'].collidepoint(event.pos):
                    section['expanded'] = not section['expanded']
                    self._rebuild_layout()
                    return True
            for slider in self.sliders:
                slider.handle_event(event)
        elif event.type == pygame.MOUSEMOTION:
            for slider in self.sliders:
                slider.handle_event(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            for slider in self.sliders:
                slider.handle_event(event)
        return True

    def _rebuild_layout(self):
        y_offset = 35
        x_start = self.rect.x + 20
        slider_width = 370

        for section_key, section in self.sections.items():
            section['rect'] = pygame.Rect(x_start, y_offset, slider_width + 20, 28)
            y_offset += 30

            if section['expanded']:
                for slider in section['sliders']:
                    slider.rect.x = x_start + 15
                    slider.rect.y = y_offset
                    y_offset += 24
            y_offset += 8

    def draw(self, screen):
        if not self.visible:
            return False

        pygame.draw.rect(screen, (30, 30, 40), self.rect)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 2)

        title = self.title_font.render("Configuration Panel (ESC/click outside to close)", True, (255, 255, 100))
        screen.blit(title, (self.rect.x + 10, self.rect.y + 8))

        for section_key, section in self.sections.items():
            sec_rect = section['rect']
            color = section['color']
            label = section['label']
            expanded = section['expanded']

            pygame.draw.rect(screen, color, sec_rect)
            pygame.draw.rect(screen, (200, 200, 200), sec_rect, 1)

            arrow = "▼" if expanded else "▶"
            header = self.section_font.render(f"{arrow} {label}", True, (255, 255, 255))
            screen.blit(header, (sec_rect.x + 10, sec_rect.y + 4))

            if expanded:
                for slider in section['sliders']:
                    slider.draw(screen, self.font)

        return True

    def close(self):
        self.visible = False



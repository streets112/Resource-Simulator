import sys
sys.path.insert(0, 'src')
from simulator.config import load_config
from simulator.simulation import Simulation
from simulator.render import MainRenderer

config = load_config('config.yaml')
sim = Simulation(config)
renderer = MainRenderer(config.visualization, config.simulation.width, config.simulation.height)
renderer.set_simulation(sim, config)

import pygame
running = True
paused = False
frame = 0
while running and frame < 150:
    frame += 1
    result = renderer.handle_events()
    if result is False:
        running = False
    elif result == 'pause':
        paused = not paused
    elif result == 'reset':
        sim = Simulation(config)
        renderer.set_simulation(sim, config)
    elif result == 'reload_env':
        pass
    if not paused:
        sim.step()
        renderer.dirty = True
    renderer.draw(sim)
    renderer.update_config_window()
    renderer.tick()
    if frame == 10:
        print('Frame 10 - testing C key (open config)')
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c)
        pygame.event.post(event)
    if frame == 20:
        print('Frame 20 - checking section rects')
        for k, v in renderer.config_panel.sections.items():
            r = v['rect']
            print('  {}: rect=({},{},{},{}), expanded={}, col_x={}'.format(k, r.x, r.y, r.w, r.h, v['expanded'], v.get('_col_x')))
    if frame == 30:
        print('Frame 30 - clicking on GRAZER header')
        rect = renderer.config_panel.sections['grazer']['rect']
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(rect.centerx, rect.centery), button=1)
        pygame.event.post(event)
        event = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(rect.centerx, rect.centery), button=1)
        pygame.event.post(event)
    if frame == 40:
        print('Frame 40 - checking GRAZER expanded:', renderer.config_panel.sections['grazer']['expanded'])
        for k, v in renderer.config_panel.sections.items():
            print('  {}: expanded={}'.format(k, v['expanded']))
    if frame == 50:
        print('Frame 50 - clicking on GRAZER header again to collapse')
        rect = renderer.config_panel.sections['grazer']['rect']
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(rect.centerx, rect.centery), button=1)
        pygame.event.post(event)
        event = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(rect.centerx, rect.centery), button=1)
        pygame.event.post(event)
    if frame == 60:
        print('Frame 60 - checking GRAZER expanded:', renderer.config_panel.sections['grazer']['expanded'])
        for k, v in renderer.config_panel.sections.items():
            print('  {}: expanded={}'.format(k, v['expanded']))
print('Done successfully')
renderer.close()
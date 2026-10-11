"""Carcass system for carcass decay and management."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.simulation import Simulation
    from simulator.entities import Carcass
    from simulator.config import CarcassConfig


class CarcassSystem:
    """Handles carcass decay and management."""
    
    def __init__(self, sim) -> None:
        self.sim = sim
    
    def step_carcasses(self) -> None:
        """Step all carcasses and remove expired ones."""
        alive_carcasses = []
        for carcass in self.sim.carcasses:
            carcass.step()
            if carcass.provides_energy:
                # Carcass still has energy, keep it
                pass
            # The Carcass.step() method handles decay internally
            # Carcasses that no longer provide energy will be filtered out by Simulation._cleanup()
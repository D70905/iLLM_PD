# -*- coding: utf-8 -*-
"""
specs.protocol — Abstract DesignProtocol interface
====================================================

A DesignProtocol bundles three things:

  1. What the spec needs from the user (required_user_inputs)
  2. What the spec needs from the FEA (required_fea_outputs)
  3. How the spec evaluates a design (allowable_values + evaluate)
  4. How the spec contributes to RL reward (reward_components)

Concrete protocols (e.g. JTG_D50_2017, MEPDG_Simplified) implement this
interface. The RL environment and HARA harness consume the abstract
interface, so they don't need to know which spec is in use.

This is the core abstraction that enables iLLM-PD to operate
"regulation-agnostically": adding a new spec means adding a new
DesignProtocol subclass, with no changes to RL/harness code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class DesignInputs:
    """
    User-provided design context (spec-agnostic core + spec-specific extras).

    Attributes:
        pavement_type:  'semi_rigid' | 'full_asphalt' | 'flexible'
        road_class:     'expressway' | 'highway_1' | 'highway_2' |
                        'highway_3' | 'highway_4'
        traffic_level:  'light' | 'medium' | 'heavy' | 'extra_heavy'
        thickness:      [h1, h2, h3] in meters (surface, base, subbase)
        modulus:        [E1, E2, E3] in MPa
        poisson:        [nu1, nu2, nu3]
        E_subgrade:     subgrade modulus, MPa
        nu_subgrade:    subgrade Poisson ratio
        design_life:    design life in years (default 15)
        extras:         spec-specific extra parameters
                        (e.g. mean_annual_temp for MEPDG, frost_zone for JTG)
    """
    pavement_type: str
    road_class: str
    traffic_level: str
    thickness: List[float]
    modulus: List[float]
    poisson: List[float]
    E_subgrade: float
    nu_subgrade: float
    design_life: int = 15
    extras: Dict = field(default_factory=dict)


@dataclass
class DesignEvaluation:
    """
    Result of evaluating one design against one spec.

    Attributes:
        feasible:           bool — passes ALL checks defined by the spec
        margins:            dict — {indicator_name: capacity/demand},
                            >=1.0 means pass. Smaller = closer to failing.
        responses:          dict — raw FEA outputs used (for audit).
        allowable_values:   dict — capacity values used (for audit).
        critical_indicator: str  — name of the worst-margin indicator.
        spec_name:          str  — protocol that produced this evaluation.
        details:            dict — spec-specific extras (predicted distress,
                            structural number, etc.) for downstream use.
    """
    feasible: bool
    margins: Dict[str, float]
    responses: Dict[str, float]
    allowable_values: Dict[str, float]
    critical_indicator: str
    spec_name: str
    details: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Abstract DesignProtocol
# ─────────────────────────────────────────────────────────────────

class DesignProtocol(ABC):
    """
    Abstract base class for pavement design protocols.

    Each concrete protocol implements:
        - name           : human-readable name (e.g. "JTG D50-2017")
        - citation       : formal citation for paper / audit log
        - required_fea_outputs() : list of FEA result keys this spec needs
        - allowable_values()     : capacity values per indicator
        - evaluate()             : compute DesignEvaluation
        - reward_components()    : convert evaluation to RL reward
    """

    name: str = "AbstractProtocol"
    citation: str = ""

    @abstractmethod
    def required_fea_outputs(self) -> List[str]:
        """
        Return the list of FEA output field names this protocol needs.

        These keys MUST be present in the dict passed to evaluate(...).

        Example keys (defined in fea/abaqus_script.py "responses"):
            'epsilon_a_microstrain'  — AC bottom radial tensile strain
            'sigma_t_MPa'            — Semi-rigid base bottom tensile stress
            'epsilon_z_microstrain'  — Subgrade top vertical compressive strain
        """
        raise NotImplementedError

    @abstractmethod
    def allowable_values(self, inputs: DesignInputs) -> Dict[str, float]:
        """
        Compute spec-defined allowable values from the design context.

        Returns a dict, e.g.:
            {
                'epsilon_a_allowable_microstrain': 250.0,
                'sigma_t_allowable_MPa': 0.45,
                'epsilon_z_allowable_microstrain': 350.0,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        inputs: DesignInputs,
        fea_outputs: Dict[str, float],
    ) -> DesignEvaluation:
        """
        Evaluate one design configuration against this spec.

        Args:
            inputs:      DesignInputs
            fea_outputs: dict with keys matching self.required_fea_outputs()

        Returns:
            DesignEvaluation
        """
        raise NotImplementedError

    @abstractmethod
    def reward_components(
        self,
        evaluation: DesignEvaluation,
    ) -> Dict[str, float]:
        """
        Convert a DesignEvaluation to reward components for RL.

        Returns:
            dict like {'performance': 0.8, 'feasibility': 1.0}
            Caller (rl/reward.py) combines with weights into scalar reward.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "<{}: {}>".format(self.__class__.__name__, self.name)


# ─────────────────────────────────────────────────────────────────
# Helpers for protocol implementations
# ─────────────────────────────────────────────────────────────────

def margin_to_score(margin: float, ideal_low: float = 1.0,
                    ideal_high: float = 1.5, hard_floor: float = 0.5) -> float:
    """
    Convert a capacity/demand margin into a [0,1] score.

    A piecewise scoring function used by RL reward:
        margin < hard_floor  -> 0.0           (severely infeasible)
        hard_floor..ideal_low-> linear ramp to 0.5
        ideal_low..ideal_high-> 1.0           (sweet spot: safe but not over-designed)
        ideal_high+         -> exponentially decaying (over-design penalty)

    This is a common shape used in HARA-compatible RL rewards; it rewards
    being "just safe enough" rather than blindly maximizing safety
    (which would lead to absurdly thick pavements).
    """
    if margin < hard_floor:
        return 0.0
    elif margin < ideal_low:
        return 0.5 * (margin - hard_floor) / (ideal_low - hard_floor)
    elif margin <= ideal_high:
        return 1.0
    else:
        # over-design penalty: smooth decay
        return max(0.0, 1.0 - 0.3 * (margin - ideal_high))

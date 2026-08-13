# -*- coding: utf-8 -*-
"""
specs — Pavement design protocol implementations
==================================================

Exposes a unified ``DesignProtocol`` interface and concrete implementations
for each supported design code:

    JTG_D50_2017       — JTG D50-2017 公路沥青路面设计规范 (China, current)
    MEPDG_Simplified   — ME-PDG (NCHRP 1-37A), simplified single-temperature
                          implementation. Documented simplifications inside.

Use the router for user-facing CLI:

    from specs import get_protocol
    protocol = get_protocol('JTG_D50_2017')   # or 'MEPDG'

Or for interactive selection:

    from specs.router import interactive_select
    protocol = interactive_select()
"""
from specs.protocol import DesignProtocol, DesignInputs, DesignEvaluation
from specs.jtg_d50 import JTG_D50_2017
from specs.mepdg import MEPDG_Simplified
from specs.router import get_protocol, interactive_select, list_protocols

__all__ = [
    'DesignProtocol',
    'DesignInputs',
    'DesignEvaluation',
    'JTG_D50_2017',
    'MEPDG_Simplified',
    'get_protocol',
    'interactive_select',
    'list_protocols',
]

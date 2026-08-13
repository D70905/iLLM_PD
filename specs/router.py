# -*- coding: utf-8 -*-
"""
specs.router — Protocol selection (programmatic + interactive)
================================================================

Provides:
    get_protocol(name)      — programmatic lookup
    interactive_select()    — CLI prompt for user
    list_protocols()        — enumerate available protocols

Usage:
    # Programmatic
    from specs import get_protocol
    protocol = get_protocol('JTG_D50_2017')

    # CLI / agentic
    from specs import interactive_select
    protocol = interactive_select()
"""
from typing import List

from specs.protocol import DesignProtocol
from specs.jtg_d50 import JTG_D50_2017
from specs.mepdg import MEPDG_Simplified


# Canonical registry. Keys are normalized (case-insensitive, alphanumeric).
_REGISTRY = {
    # JTG D50-2017 aliases
    'jtg':            JTG_D50_2017,
    'jtg_d50':        JTG_D50_2017,
    'jtg_d50_2017':   JTG_D50_2017,
    'china':          JTG_D50_2017,
    'chinese':        JTG_D50_2017,
    # ME-PDG aliases
    'mepdg':          MEPDG_Simplified,
    'me_pdg':         MEPDG_Simplified,
    'me-pdg':         MEPDG_Simplified,
    'nchrp':          MEPDG_Simplified,
    'aashtoware':     MEPDG_Simplified,
    'us':             MEPDG_Simplified,
    'usa':            MEPDG_Simplified,
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(' ', '_')


def get_protocol(name: str, **kwargs) -> DesignProtocol:
    """
    Get protocol by name (case-insensitive, multiple aliases supported).

    Args:
        name: protocol name. Recognized:
            - 'JTG_D50_2017', 'jtg', 'china'  -> JTG_D50_2017
            - 'MEPDG', 'me-pdg', 'us'         -> MEPDG_Simplified
        **kwargs: passed to protocol constructor

    Raises:
        ValueError if name is not recognized.
    """
    key = _normalize(name)
    if key not in _REGISTRY:
        raise ValueError(
            "Unknown protocol '{}'. Available: {}".format(
                name, ', '.join(sorted({c.name for c in _REGISTRY.values()}))
            )
        )
    return _REGISTRY[key](**kwargs)


def list_protocols() -> List[dict]:
    """
    Return metadata for each unique protocol.
    """
    seen = set()
    out = []
    for cls in _REGISTRY.values():
        if cls in seen:
            continue
        seen.add(cls)
        instance = cls()
        out.append({
            'name':      instance.name,
            'class':     cls.__name__,
            'citation':  instance.citation,
        })
    return out


def interactive_select() -> DesignProtocol:
    """
    Prompt the user to pick a protocol.

    Returns the selected DesignProtocol instance.
    """
    print('=' * 70)
    print('iLLM-PD: Auditable LLM-Harness for Regulated Pavement Design')
    print('=' * 70)
    print()
    print('Which pavement design code do you want to apply?')
    print()
    print('  [1] JTG D50-2017   (China, current — issued 2017-09-01)')
    print('        Full 3-indicator check: AC fatigue, semi-rigid fatigue,')
    print('        subgrade vertical strain')
    print()
    print('  [2] ME-PDG         (NCHRP 1-37A, simplified)')
    print('        US/AASHTOWare-equivalent; ★ simplified implementation:')
    print('        single-temperature, single-axle, bottom-up fatigue +')
    print('        total rutting only.')
    print()
    print('  [3] Cross-spec     Run both [1] and [2], compare results')
    print()
    while True:
        choice = input('Your selection [1/2/3]: ').strip()
        if choice == '1':
            print()
            print('Selected: JTG D50-2017')
            print()
            return JTG_D50_2017()
        elif choice == '2':
            print()
            print('Selected: ME-PDG (simplified)')
            print('★ Note: Full ME-PDG requires climate data and axle '
                  'spectrum; this is a research-grade simplified version.')
            print()
            return MEPDG_Simplified()
        elif choice == '3':
            print()
            print('Cross-spec comparison mode. Returning both protocols.')
            print()
            return [JTG_D50_2017(), MEPDG_Simplified()]
        else:
            print('Invalid selection. Please enter 1, 2, or 3.')

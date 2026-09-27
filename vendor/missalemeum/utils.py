"""Extrait de `api/utils.py` (Missale Meum) : fonctions utiles au calendrier."""
import re
from typing import List, Pattern, Union

from vendor.missalemeum.constants.common import CUSTOM_PREFACES


def match_all(observances: Union[str, 'Observance', List[Union[str, 'Observance']]],  # noqa: F821
                patterns: Union[List[str], str, List[Pattern], Pattern]):
    matches = []
    if not isinstance(observances, (list, tuple)):
        observances = [observances]
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]
    for observance in observances:
        observance_id = observance if isinstance(observance, str) else observance.id
        for pattern in patterns:
            if re.match(pattern, observance_id):
                matches.append(observance)
    return matches


def match_first(observances: Union[str, 'Observance', List[Union[str, 'Observance']]],  # noqa: F821
                patterns: Union[List[str], str, List[Pattern], Pattern]):
    if matches := match_all(observances, patterns):
        return matches[0]


def get_custom_preface(celebration: 'Observance', tempora: 'Observance' = None) -> Union[str, None]:  # noqa: F821
    for pattern, preface_name in CUSTOM_PREFACES:
        if (re.match(pattern, celebration.id)) or (tempora and celebration.rank > 1 and re.match(pattern, tempora.id)):
            return preface_name
    return None

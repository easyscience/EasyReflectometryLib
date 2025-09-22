__author__ = 'github.com/arm61'

import re
from typing import Tuple

ATOM_REGEX = '([A-Z][a-z]*)(\\d*)'
OPENERS = '({['
CLOSERS = ')}]'


def _dictify(tuples: Tuple[Tuple[str, str]]) -> dict:
    """
    :param tuples: tuples of tuples with atom and occurance.
    :return: Dict of atoms and occurance.
    """
    res = dict()
    for atom, n in tuples:
        try:
            res[atom] += int(n or 1)
        except KeyError:
            res[atom] = int(n or 1)
    return res


def _fuse(mol1: dict, mol2: dict, w: int = 1) -> dict:
    """
    :param mol1: First dict to fuse
    :param mol2: Second dict to fuse
    :param w: Weight for dicts
    :return: Fused dictionaries
    """
    return {atom: (mol1.get(atom, 0) + mol2.get(atom, 0)) * w for atom in set(mol1) | set(mol2)}


def _parse(formula: str) -> Tuple[dict, int]:
    """
    :param formula: Chemical formula as a string
    :return: Tuple containing; formula as a dictwith occurences
        of each atom and an iterator.
    """
    token_list = []
    molecule_dict = {}
    i = 0

    while i < len(formula):
        # Using a classic loop allow for manipulating the cursor
        token = formula[i]

        if token in CLOSERS:
            # Check for an index for this part
            m = re.match('\\d+', formula[i + 1 :])
            if m:
                weight = int(m.group(0))
                i += len(m.group(0))
            else:
                weight = 1

            submol = _dictify(re.findall(ATOM_REGEX, ''.join(token_list)))
            return _fuse(molecule_dict, submol, weight), i

        if token in OPENERS:
            submol, letter = _parse(formula[i + 1 :])
            molecule_dict = _fuse(molecule_dict, submol)
            # skip the already read submol
            i += letter + 1
        else:
            token_list.append(token)

        i += 1

    # Fuse in all that's left at base level
    return _fuse(molecule_dict, _dictify(re.findall(ATOM_REGEX, ''.join(token_list)))), i


def parse_formula(formula: str) -> dict:
    """
    :param formula: Chemical formula as a string
    :return: Formula as a dict with occurences of each atom.
    """
    return _parse(formula)[0]

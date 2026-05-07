# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/arm61'

import periodictable as pt

from easyreflectometry.special.parsing import parse_formula


def weighted_average(a: float, b: float, p: float) -> float:
    """Determine the weighted average for a and b, where p is the weight.

    Parameters
    ----------
    a : float
        First value.
    b : float
        Second value.
    p : float
        Weight.

    Returns
    -------
    float
        Weighted average.
    """
    return a * (1 - p) + b * p


def neutron_scattering_length(formula: str) -> complex:
    """Determine the neutron scattering length for a chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula.

    Returns
    -------
    complex
        Real and imaginary descriptors for the scattering length in angstrom.
    """
    formula_as_dict = parse_formula(formula)
    scattering_length = 0 + 0j
    for key, value in formula_as_dict.items():
        scattering_length += pt.elements.symbol(key).neutron.b_c * value
        if pt.elements.symbol(key).neutron.b_c_i:
            inc = pt.elements.symbol(key).neutron.b_c_i
        else:
            inc = 0
        scattering_length += inc * 1j * value
    return scattering_length * 1e-5


def molecular_weight(formula: str) -> float:
    """Determine the molecular weight for a chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula.

    Returns
    -------
    float
        Molecular weight of the material in kilograms.
    """
    formula_as_dict = parse_formula(formula)
    mw = 0
    for key, value in formula_as_dict.items():
        mw += pt.elements.symbol(key).mass * value
    return mw


def area_per_molecule_to_scattering_length_density(
    scattering_length: float,
    thickness: float,
    area_per_molecule: float,
) -> float:
    """Find the scattering length density for a given area per molecule.

    Parameters
    ----------
    scattering_length : float
        Scattering length of component, in angstrom.
    thickness : float
        Thickness of component, in angstrom.
    area_per_molecule : float
        Area per molecule, in angstrom^2.

    Returns
    -------
    float
        Scattering length density of layer in e-6 1/angstrom^2.
    """
    return scattering_length / (thickness * area_per_molecule) * 1e6


def density_to_sld(scattering_length: float, molecular_weight: float, density: float) -> float:
    """Find the scattering length density from the mass density of a material.

    Parameters
    ----------
    scattering_length : float
        Scattering length of component, in angstrom.
    molecular_weight : float
        Molecular weight of component, in u.
    density : float
        Mass density of the component, in gram centimeter^-3.

    Returns
    -------
    float
        Scattering length density of layer in e-6 1/angstrom^2.
    """
    # 0.602214076 is avogadros constant times 1e-24
    return 0.602214076e6 * density * scattering_length / molecular_weight

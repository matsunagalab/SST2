#!/usr/bin/env python3
# coding: utf-8

"""
Build a sub-System (a subset of particles) by copying an existing
openmm.System instead of re-parametrising a sub-topology with a
ForceField.

REST2 needs solute-only and solvent-only Systems to evaluate the
solute-solvent PME energy by subtraction.  Building them from the parent
System has three advantages over ``ForceField.createSystem`` on a
sub-topology:

* it works for Systems that have no ForceField object (Amber prmtop,
  serialized ``system.xml``);
* the nonbonded settings (cutoff, Ewald tolerance, PME alpha and grid,
  dispersion correction, switching) are copied verbatim, so the
  subtraction is exact;
* no residue-template matching is involved, so a solute that starts or
  ends inside a chain needs no ``ignoreExternalBonds`` and cannot pick
  up bonds that the sub-topology does not have.

Only interaction terms whose particles are all in the subset are kept.
"""

import logging

import numpy as np
import openmm
import openmm.app as app
from openmm import unit

logger = logging.getLogger(__name__)


# Forces that never belong to a sub-System used for energy evaluation.
_SKIPPED_FORCES = (
    openmm.MonteCarloBarostat,
    openmm.MonteCarloAnisotropicBarostat,
    openmm.MonteCarloMembraneBarostat,
    openmm.AndersenThermostat,
)
try:  # OpenMM >= 8.1
    _SKIPPED_FORCES += (openmm.MonteCarloFlexibleBarostat,)
except AttributeError:  # pragma: no cover
    pass


def _copy_tabulated(func):
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(func))


def _copy_global_parameters(src, dst):
    for i in range(src.getNumGlobalParameters()):
        dst.addGlobalParameter(
            src.getGlobalParameterName(i), src.getGlobalParameterDefaultValue(i)
        )


def _copy_energy_parameter_derivatives(src, dst):
    for i in range(src.getNumEnergyParameterDerivatives()):
        dst.addEnergyParameterDerivative(src.getEnergyParameterDerivativeName(i))


def _copy_virtual_site(site, remap):
    """Return a copy of ``site`` with remapped particle indices, or None if
    one of its reference particles is not in the subset."""
    n = site.getNumParticles()
    parts = [site.getParticle(i) for i in range(n)]
    if any(p not in remap for p in parts):
        return None
    new = [remap[p] for p in parts]
    if isinstance(site, openmm.TwoParticleAverageSite):
        return openmm.TwoParticleAverageSite(
            new[0], new[1], site.getWeight(0), site.getWeight(1)
        )
    if isinstance(site, openmm.ThreeParticleAverageSite):
        return openmm.ThreeParticleAverageSite(
            new[0], new[1], new[2],
            site.getWeight(0), site.getWeight(1), site.getWeight(2),
        )
    if isinstance(site, openmm.OutOfPlaneSite):
        return openmm.OutOfPlaneSite(
            new[0], new[1], new[2],
            site.getWeight12(), site.getWeight13(), site.getWeightCross(),
        )
    if isinstance(site, openmm.LocalCoordinatesSite):
        return openmm.LocalCoordinatesSite(
            new,
            site.getOriginWeights(),
            site.getXWeights(),
            site.getYWeights(),
            site.getLocalPosition(),
        )
    raise NotImplementedError(f"virtual site type {type(site).__name__}")


def _all_in(remap, *particles):
    return all(p in remap for p in particles)


def _copy_nonbonded(src, remap, keep):
    dst = openmm.NonbondedForce()
    dst.setNonbondedMethod(src.getNonbondedMethod())
    dst.setCutoffDistance(src.getCutoffDistance())
    dst.setEwaldErrorTolerance(src.getEwaldErrorTolerance())
    dst.setPMEParameters(*src.getPMEParameters())
    dst.setLJPMEParameters(*src.getLJPMEParameters())
    dst.setUseSwitchingFunction(src.getUseSwitchingFunction())
    dst.setSwitchingDistance(src.getSwitchingDistance())
    dst.setUseDispersionCorrection(src.getUseDispersionCorrection())
    dst.setExceptionsUsePeriodicBoundaryConditions(
        src.getExceptionsUsePeriodicBoundaryConditions()
    )
    dst.setIncludeDirectSpace(src.getIncludeDirectSpace())
    dst.setReciprocalSpaceForceGroup(src.getReciprocalSpaceForceGroup())
    dst.setReactionFieldDielectric(src.getReactionFieldDielectric())
    _copy_global_parameters(src, dst)
    for i in keep:
        dst.addParticle(*src.getParticleParameters(i))
    exception_map = {}
    for i in range(src.getNumExceptions()):
        p1, p2, q, sig, eps = src.getExceptionParameters(i)
        if _all_in(remap, p1, p2):
            exception_map[i] = dst.addException(remap[p1], remap[p2], q, sig, eps)
    for i in range(src.getNumParticleParameterOffsets()):
        name, p, q, sig, eps = src.getParticleParameterOffset(i)
        if p in remap:
            dst.addParticleParameterOffset(name, remap[p], q, sig, eps)
    for i in range(src.getNumExceptionParameterOffsets()):
        name, e, q, sig, eps = src.getExceptionParameterOffset(i)
        if e in exception_map:
            dst.addExceptionParameterOffset(name, exception_map[e], q, sig, eps)
    return dst


def _copy_custom_nonbonded(src, remap, keep):
    dst = openmm.CustomNonbondedForce(src.getEnergyFunction())
    dst.setNonbondedMethod(src.getNonbondedMethod())
    dst.setCutoffDistance(src.getCutoffDistance())
    dst.setUseSwitchingFunction(src.getUseSwitchingFunction())
    dst.setSwitchingDistance(src.getSwitchingDistance())
    dst.setUseLongRangeCorrection(src.getUseLongRangeCorrection())
    for i in range(src.getNumPerParticleParameters()):
        dst.addPerParticleParameter(src.getPerParticleParameterName(i))
    _copy_global_parameters(src, dst)
    for i in range(src.getNumTabulatedFunctions()):
        dst.addTabulatedFunction(
            src.getTabulatedFunctionName(i),
            _copy_tabulated(src.getTabulatedFunction(i)),
        )
    for i in range(src.getNumComputedValues()):
        dst.addComputedValue(*src.getComputedValueParameters(i))
    _copy_energy_parameter_derivatives(src, dst)
    for i in keep:
        dst.addParticle(src.getParticleParameters(i))
    for i in range(src.getNumExclusions()):
        p1, p2 = src.getExclusionParticles(i)
        if _all_in(remap, p1, p2):
            dst.addExclusion(remap[p1], remap[p2])
    for i in range(src.getNumInteractionGroups()):
        g1, g2 = src.getInteractionGroupParameters(i)
        g1 = [remap[p] for p in g1 if p in remap]
        g2 = [remap[p] for p in g2 if p in remap]
        if g1 and g2:
            dst.addInteractionGroup(g1, g2)
    return dst


def _copy_custom_bond(src, remap):
    dst = openmm.CustomBondForce(src.getEnergyFunction())
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumPerBondParameters()):
        dst.addPerBondParameter(src.getPerBondParameterName(i))
    _copy_global_parameters(src, dst)
    _copy_energy_parameter_derivatives(src, dst)
    for i in range(src.getNumBonds()):
        p1, p2, params = src.getBondParameters(i)
        if _all_in(remap, p1, p2):
            dst.addBond(remap[p1], remap[p2], params)
    return dst


def _copy_custom_angle(src, remap):
    dst = openmm.CustomAngleForce(src.getEnergyFunction())
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumPerAngleParameters()):
        dst.addPerAngleParameter(src.getPerAngleParameterName(i))
    _copy_global_parameters(src, dst)
    _copy_energy_parameter_derivatives(src, dst)
    for i in range(src.getNumAngles()):
        p1, p2, p3, params = src.getAngleParameters(i)
        if _all_in(remap, p1, p2, p3):
            dst.addAngle(remap[p1], remap[p2], remap[p3], params)
    return dst


def _copy_custom_torsion(src, remap):
    dst = openmm.CustomTorsionForce(src.getEnergyFunction())
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumPerTorsionParameters()):
        dst.addPerTorsionParameter(src.getPerTorsionParameterName(i))
    _copy_global_parameters(src, dst)
    _copy_energy_parameter_derivatives(src, dst)
    for i in range(src.getNumTorsions()):
        p1, p2, p3, p4, params = src.getTorsionParameters(i)
        if _all_in(remap, p1, p2, p3, p4):
            dst.addTorsion(remap[p1], remap[p2], remap[p3], remap[p4], params)
    return dst


def _copy_harmonic_bond(src, remap):
    dst = openmm.HarmonicBondForce()
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumBonds()):
        p1, p2, r0, k = src.getBondParameters(i)
        if _all_in(remap, p1, p2):
            dst.addBond(remap[p1], remap[p2], r0, k)
    return dst


def _copy_harmonic_angle(src, remap):
    dst = openmm.HarmonicAngleForce()
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumAngles()):
        p1, p2, p3, a0, k = src.getAngleParameters(i)
        if _all_in(remap, p1, p2, p3):
            dst.addAngle(remap[p1], remap[p2], remap[p3], a0, k)
    return dst


def _copy_periodic_torsion(src, remap):
    dst = openmm.PeriodicTorsionForce()
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumTorsions()):
        p1, p2, p3, p4, n, phase, k = src.getTorsionParameters(i)
        if _all_in(remap, p1, p2, p3, p4):
            dst.addTorsion(remap[p1], remap[p2], remap[p3], remap[p4], n, phase, k)
    return dst


def _copy_rb_torsion(src, remap):
    dst = openmm.RBTorsionForce()
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumTorsions()):
        p1, p2, p3, p4, *c = src.getTorsionParameters(i)
        if _all_in(remap, p1, p2, p3, p4):
            dst.addTorsion(remap[p1], remap[p2], remap[p3], remap[p4], *c)
    return dst


def _copy_cmap(src, remap):
    dst = openmm.CMAPTorsionForce()
    dst.setUsesPeriodicBoundaryConditions(src.usesPeriodicBoundaryConditions())
    for i in range(src.getNumMaps()):
        size, energy = src.getMapParameters(i)
        dst.addMap(size, energy)
    for i in range(src.getNumTorsions()):
        m, *p = src.getTorsionParameters(i)
        if _all_in(remap, *p):
            dst.addTorsion(m, *[remap[x] for x in p])
    return dst


def _copy_cmmotion(src, remap):
    return openmm.CMMotionRemover(src.getFrequency())


_COPIERS = {
    openmm.HarmonicBondForce: _copy_harmonic_bond,
    openmm.HarmonicAngleForce: _copy_harmonic_angle,
    openmm.PeriodicTorsionForce: _copy_periodic_torsion,
    openmm.RBTorsionForce: _copy_rb_torsion,
    openmm.CMAPTorsionForce: _copy_cmap,
    openmm.CustomBondForce: _copy_custom_bond,
    openmm.CustomAngleForce: _copy_custom_angle,
    openmm.CustomTorsionForce: _copy_custom_torsion,
    openmm.CMMotionRemover: _copy_cmmotion,
}


def build_subsystem(system, keep, *, strict=True):
    """Return a new System containing only the particles in ``keep``.

    Parameters
    ----------
    system : openmm.System
        Parent System.  It is not modified.
    keep : iterable of int
        Particle indices to keep (any order; the sub-System orders them
        increasingly).
    strict : bool
        If True, raise on a force type that cannot be copied; if False,
        skip it with a warning.

    Returns
    -------
    sub : openmm.System
        Sub-System.  Force names are preserved.
    index_map : dict
        Parent particle index -> sub-System particle index.
    """
    keep = sorted(set(int(i) for i in keep))
    n = system.getNumParticles()
    if not keep or keep[0] < 0 or keep[-1] >= n:
        raise ValueError("keep must be a non-empty subset of the particle indices")
    remap = {old: new for new, old in enumerate(keep)}

    sub = openmm.System()
    sub.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    for old in keep:
        sub.addParticle(system.getParticleMass(old))

    dropped_vsites = 0
    for old in keep:
        if system.isVirtualSite(old):
            site = _copy_virtual_site(system.getVirtualSite(old), remap)
            if site is None:
                dropped_vsites += 1
                continue
            sub.setVirtualSite(remap[old], site)
    if dropped_vsites:
        raise ValueError(
            f"{dropped_vsites} virtual sites in the subset depend on particles "
            f"outside it; extend the subset to whole molecules"
        )

    for i in range(system.getNumConstraints()):
        p1, p2, d = system.getConstraintParameters(i)
        if _all_in(remap, p1, p2):
            sub.addConstraint(remap[p1], remap[p2], d)

    for force in system.getForces():
        if isinstance(force, _SKIPPED_FORCES):
            continue
        if isinstance(force, openmm.NonbondedForce):
            new = _copy_nonbonded(force, remap, keep)
        elif isinstance(force, openmm.CustomNonbondedForce):
            new = _copy_custom_nonbonded(force, remap, keep)
        else:
            copier = None
            for cls, fn in _COPIERS.items():
                if isinstance(force, cls):
                    copier = fn
                    break
            if copier is None:
                msg = f"cannot copy force type {type(force).__name__} into a sub-System"
                if strict:
                    raise NotImplementedError(msg)
                logger.warning(msg + "; skipped")
                continue
            new = copier(force, remap)
        new.setName(force.getName())
        new.setForceGroup(force.getForceGroup())
        sub.addForce(new)

    return sub, remap


def subset_topology_positions(topology, positions, keep):
    """Sub-topology and positions for ``keep`` (bonds restricted to the
    subset, box vectors preserved), via Modeller.delete."""
    keep = set(int(i) for i in keep)
    atoms = list(topology.atoms())
    modeller = app.Modeller(topology, positions)
    modeller.delete([a for a in atoms if a.index not in keep])
    return modeller.topology, modeller.positions

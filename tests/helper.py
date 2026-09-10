from pathlib import Path

import numpy as np
from pyobjcryst import loadCrystal
from scipy.optimize import least_squares, leastsq

from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    Profile,
)
from diffpy.srfit.pdf import PDFGenerator, PDFParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser


def run_ni_example():
    structure_path = str(Path(__file__).parent / "data" / "Ni.cif")
    profile_path = str(Path(__file__).parent / "data" / "Ni.gr")
    initial_pv_dict = {
        "s0": 0.4,
        "qdamp": 0.04,
        "qbroad": 0.02,
        "G1_a": 3.52,
        "G1_delta2": 2,
        "G1_Uiso_0": 0.005,
    }
    variables_to_refine = [
        "G1_a",
        "s0",
        "G1_Uiso_0",
        "G1_delta2",
        "qdamp",
        "qbroad",
    ]
    PDF_RMIN = 1.5
    # PDF_RMAX = 50
    PDF_RMAX = 20  # reduced for testing purposes
    PDF_RSTEP = 0.01
    QMAX = 25
    QMIN = 0.1
    SCALE_I = 0.4
    CUBICLAT_I = 3.52
    UISO_I = 0.005
    DELTA2_I = 2
    QDAMP_I = 0.04
    QBROAD_I = 0.02
    RUN_PARALLEL = True

    p_cif = getParser("cif")
    stru1 = p_cif.parseFile(structure_path)
    sg = p_cif.spacegroup.short_name
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(profile_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)
    generator_crystal1.delta2.value = DELTA2_I
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)
    if RUN_PARALLEL:
        try:
            import multiprocessing
            from multiprocessing import Pool

            import psutil

            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_crystal1.parallel(ncpu=ncpu, mapfunc=pool.map)
        except ImportError:
            print(
                "\nYou don't appear to have the necessary packages for "
                "parallelization"
            )
    contribution.setProfile(profile, xname="r")
    contribution.setEquation("s0*G1")
    recipe = FitRecipe()
    recipe.addContribution(contribution)
    recipe.crystal.G1.qdamp.value = QDAMP_I
    recipe.crystal.G1.qbroad.value = QBROAD_I
    recipe.crystal.G1.setQmax(QMAX)
    recipe.crystal.G1.setQmin(QMIN)
    recipe.addVar(contribution.s0, SCALE_I, name="s0")
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase, sg)
    for par in spacegroupparams.latpars:
        recipe.addVar(par, value=CUBICLAT_I, fixed=False, name="G1_a")
    for par in spacegroupparams.adppars:
        recipe.addVar(par, value=UISO_I, fixed=False, name="G1_Uiso_0")
    recipe.addVar(generator_crystal1.delta2, name="G1_delta2")
    recipe.addVar(
        generator_crystal1.qdamp,
        fixed=False,
        name="qdamp",
        value=QDAMP_I,
    )
    recipe.addVar(
        generator_crystal1.qbroad,
        fixed=False,
        name="qbroad",
        value=QBROAD_I,
    )
    recipe.fithooks[0].verbose = 0
    for init_name, init_value in initial_pv_dict.items():
        if init_name in recipe._parameters:
            recipe._parameters[init_name].value = init_value
    recipe.fix("all")
    for var_name in variables_to_refine:
        recipe.free(var_name)
        least_squares(
            recipe.residual,
            recipe.values,
            x_scale="jac",
        )
    diffpy_pv_dict = {}
    for pname, parameter in recipe._parameters.items():
        diffpy_pv_dict[pname] = parameter.value
    return diffpy_pv_dict


def run_multi_contribution_example():
    ciffile_ni = Path(__file__).parent / "data/ni.cif"
    ciffile_si = Path(__file__).parent / "data/si.cif"
    xdata_ni = Path(__file__).parent / "data/ni-q27r60-xray.gr"
    ndata_ni = Path(__file__).parent / "data/ni-q27r100-neutron.gr"
    xdata_si = Path(__file__).parent / "data/si-q27r60-xray.gr"
    xdata_sini = Path(__file__).parent / "data/si90ni10-q27r60-xray.gr"

    def makeProfile(datafile):
        profile = Profile()
        parser = PDFParser()
        parser.parse_file(datafile)
        profile.load_parsed_data(parser)
        profile.set_calculation_range(xmax=20)
        return profile

    def makeContribution(name, generator, profile):
        contribution = FitContribution(name)
        contribution.add_profile_generator(generator)
        contribution.set_profile(profile, xname="r")
        return contribution

    xprofile_ni = makeProfile(xdata_ni)
    xprofile_si = makeProfile(xdata_si)
    nprofile_ni = makeProfile(ndata_ni)
    xprofile_sini = makeProfile(xdata_sini)
    xgenerator_ni = PDFGenerator("xG_ni")
    stru = loadCrystal(ciffile_ni)
    xgenerator_ni.setStructure(stru)
    phase_ni = xgenerator_ni.phase
    xgenerator_si = PDFGenerator("xG_si")
    stru = loadCrystal(ciffile_si)
    xgenerator_si.setStructure(stru)
    phase_si = xgenerator_si.phase
    ngenerator_ni = PDFGenerator("nG_ni")
    ngenerator_ni.setPhase(phase_ni)
    xgenerator_sini_ni = PDFGenerator("xG_sini_ni")
    xgenerator_sini_ni.setPhase(phase_ni)
    xgenerator_sini_si = PDFGenerator("xG_sini_si")
    xgenerator_sini_si.setPhase(phase_si)
    xcontribution_ni = makeContribution("xnickel", xgenerator_ni, xprofile_ni)
    xcontribution_si = makeContribution("xsilicon", xgenerator_si, xprofile_si)
    ncontribution_ni = makeContribution("nnickel", ngenerator_ni, nprofile_ni)
    xcontribution_sini = makeContribution(
        "xsini", xgenerator_sini_ni, xprofile_sini
    )
    xcontribution_sini.add_profile_generator(xgenerator_sini_si)
    xcontribution_sini.set_equation("scale * (xG_sini_ni +  xG_sini_si)")
    xcontribution_ni.set_residual_equation("resv")
    xcontribution_si.set_residual_equation("resv")
    ncontribution_ni.set_residual_equation("resv")
    xcontribution_sini.set_residual_equation("resv")
    recipe = FitRecipe()
    recipe.add_contribution(xcontribution_ni)
    recipe.add_contribution(xcontribution_si)
    recipe.add_contribution(ncontribution_ni)
    recipe.add_contribution(xcontribution_sini)
    for par in phase_ni.sgpars:
        recipe.add_variable(par, name=par.name + "_ni")
    delta2_ni = recipe.create_new_variable("delta2_ni", 2.5)
    recipe.add_constraint(xgenerator_ni.delta2, delta2_ni)
    recipe.add_constraint(ngenerator_ni.delta2, delta2_ni)
    recipe.add_constraint(xgenerator_sini_ni.delta2, delta2_ni)
    for par in phase_si.sgpars:
        recipe.add_variable(par, name=par.name + "_si")
    delta2_si = recipe.create_new_variable("delta2_si", 2.5)
    recipe.add_constraint(xgenerator_si.delta2, delta2_si)
    recipe.add_constraint(xgenerator_sini_si.delta2, delta2_si)
    recipe.add_variable(xgenerator_ni.scale, name="xscale_ni")
    recipe.add_variable(xgenerator_si.scale, name="xscale_si")
    recipe.add_variable(ngenerator_ni.scale, name="nscale_ni")
    recipe.add_variable(xcontribution_sini.scale, 1.0, "xscale_sini")
    recipe.create_new_variable("pscale_sini_ni", 0.8)
    recipe.add_constraint(xgenerator_sini_ni.scale, "pscale_sini_ni")
    recipe.add_constraint(xgenerator_sini_si.scale, "1 - pscale_sini_ni")
    xgenerator_ni.qdamp.value = 0.055
    xgenerator_si.qdamp.value = 0.051
    ngenerator_ni.qdamp.value = 0.030
    xgenerator_sini_ni.qdamp.value = 0.052
    xgenerator_sini_si.qdamp.value = 0.052
    leastsq(recipe.residual, recipe.get_values())
    diffpy_pv_dict = {
        name: par.value for name, par in recipe._parameters.items()
    }
    return diffpy_pv_dict


if __name__ == "__main__":
    # diffpy_pv_dict = run_ni_example()
    diffpy_pv_dict = run_multi_contribution_example()
    print(diffpy_pv_dict)

import numpy as np
from diffpy.srfit.fitbase import FitContribution, FitRecipe, Profile
from diffpy.srfit.pdf import PDFGenerator, PDFParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser


def make_cmi_recipe(cif_path, dat_path, variable_values={}):
    """Creates and returns a diffpy.cmi Fit Recipe object.

    Parameters
    ----------
    cif_path :  str
        The full path to the structure CIF file to load.
    dat_path :  str
        The full path to the PDF data to be fit.
    variable_values : dict,
        The dictionary of variable values to initialize the
        FitRecipe.

    Returns
    ----------
    recipe : FitRecipe
        The created FitRecipe.
    """
    PDF_RMIN = variable_values.get("xmin", 1.5)
    PDF_RMAX = variable_values.get("xmax", 50)
    PDF_RSTEP = variable_values.get("dx", 0.01)
    QMAX = variable_values.get("qmax", 25)
    QMIN = variable_values.get("qmin", 0.1)
    SCALE_I = variable_values.get("s0", 0.4)
    CUBICLAT_I = variable_values.get("G1_a", 3.52)
    UISO_I = variable_values.get("G1_Uiso_0", 0.005)
    DELTA2_I = variable_values.get("G1_delta2", 2)
    QDAMP_I = variable_values.get("qdamp", 0.04)
    QBROAD_I = variable_values.get("qbroad", 0.02)
    RUN_PARALLEL = True

    p_cif = getParser("cif")
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
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
    return recipe

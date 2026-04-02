import warnings
from pathlib import Path

import numpy

from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    FitResults,
    Profile,
)
from diffpy.srfit.pdf import PDFGenerator, PDFParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser


class PDFAdapter:
    """Adapter to expose PDF fitting interface. Designed to provide a
    simplified PDF fitting interface for human users and AI agents.

    Attributes
    ----------
    recipe : FitRecipe
        The FitRecipe object managing the fitting process.

    Methods
    -------
    initialize_profile(profile_path, qmin=None, qmax=None, xmin=None, xmax=None, dx=None)
        Load and initialize the PDF profile from the given file path with
        some optional parameters.
    initialize_structures(structure_paths : list[str], run_parallel=True)
        Load and initialize the structures from the given file paths, and
        generate corresponding PDFGenerator objects.
    initialize_contribution(equation_string=None)
        Initialize the FitContribution object combining the PDF generators and
        the profile.
    initialize_recipe()
        Initialize the FitRecipe object for the fitting process.
    set_initial_variable_values(variable_name_to_value : dict)
        Update parameter values from the provided dictionary.
    refine_variables(variable_names: list[str])
        Refine the parameters specified in the list and in that order.
    get_variable_names()
        Get the names of all variables in the recipe.
    save_results(mode: str, filename: str=None)
        Save the fitting results.
    """  # noqa: E501

    def initialize_profile(
        self,
        profile_path: str,
        q_range=None,
        calculation_range=None,
    ):
        """Load and initialize the PDF profile from the given file path
        with some optional parameters.

        The target output, FitRecipe, requires a profile object, multiple
        PDFGenerator objects, and a FitContribution object combining them. This
        method initializes the profile object.

        Parameters
        ----------
        profile_path : str
            The path to the experimental PDF profile file.
        q_range: list or tuple of two floats.
            The qmin and qmax for PDF calculation. The default value is None,
            which means using the values parsed from the profile file.
        calculation_range : list or tuple of three floats.
            The rmin, rmax, and r step for PDF calculation. The default value
            is None, which means using the range parsed from the profile file.
        """
        profile = Profile()
        parser = PDFParser()
        parser.parseString(Path(profile_path).read_text())
        profile.loadParsedData(parser)
        if q_range is not None:
            profile.meta["qmin"] = q_range[0]
            profile.meta["qmax"] = q_range[1]
        if calculation_range is not None:
            if isinstance(calculation_range, (list, tuple)):
                calculation_range = {
                    "xmin": calculation_range[0],
                    "xmax": calculation_range[1],
                    "dx": calculation_range[2],
                }
            profile.setCalculationRange(**calculation_range)
        self.profile = profile

    def initialize_structures(
        self,
        structure_paths: list[str],
        run_parallel=True,
        spacegroups=None,
        names=None,
    ):
        """Load and initialize the structures from the given file paths,
        and generate corresponding PDFGenerator objects.

        The target output, FitRecipe, requires a profile object, multiple
        PDFGenerator objects, and a FitContribution object combining them. This
        method creates the PDFGenerator objects from the structure files.

        Must be called after initialize_profile.

        Parameters
        ----------
        structure_paths : list of str
            The list of paths to the structure files (CIF format).
        run_parallel : bool
        spacegroups : list of str or None
            The space group for each structure. If None, the space group will
            be determined automatically from the structure file. The default is
            None.
        names: list of str or None
            The names for each structure. If None, default names "G1", "G2",
            ... will be assigned. The default is None.

        Notes
        -----
        Planned features:
            - Support cif file manipulation.
                - Add/Remove atoms.
                - symmetry operations?
        """
        if isinstance(structure_paths, str):
            structure_paths = [structure_paths]
        structures = []
        spacegroups = []
        pdfgenerators = []
        if run_parallel:
            try:
                import multiprocessing
                from multiprocessing import Pool

                import psutil

                syst_cores = multiprocessing.cpu_count()
                cpu_percent = psutil.cpu_percent()
                avail_cores = numpy.floor(
                    (100 - cpu_percent) / (100.0 / syst_cores)
                )
                ncpu = int(numpy.max([1, avail_cores]))
                pool = Pool(processes=ncpu)
                self.pool = pool
            except ImportError:
                warnings.warn(
                    "\nYou don't appear to have the necessary packages for "
                    "parallelization. Proceeding without parallelization."
                )
                run_parallel = False
        for i, structure_path in enumerate(structure_paths):
            name = names[i] if names and i < len(names) else f"G{i+1}"
            stru_parser = getParser("cif")
            structure = stru_parser.parse(Path(structure_path).read_text())
            sg = getattr(stru_parser, "spacegroup", None)
            spacegroup = sg.short_name if sg is not None else "P1"
            structures.append(structure)
            spacegroups.append(spacegroup)
            pdfgenerator = PDFGenerator(name)
            pdfgenerator.setStructure(structure)
            if run_parallel:
                pdfgenerator.parallel(ncpu=ncpu, mapfunc=self.pool.map)
            pdfgenerators.append(pdfgenerator)
        self.spacegroups = spacegroups
        self.pdfgenerators = pdfgenerators

    def initialize_contribution(self, equation_string=None):
        """Initialize the FitContribution object combining the PDF
        generators and the profile.

        The target output, FitRecipe, requires a profile object, multiple
        PDFGenerator objects, and a FitContribution object combining them. This
        method creates the FitContribution object combining the profile and PDF
        generators.

        Must be called after initialize_profile and initialize_structures.

        Parameters
        ----------
        equation_string : str
            The equation string defining the contribution. The default
            equation will be generated based on the number of phases.
            e.g.
            for one phase: "s0*G1",
            for two phases: "s0*(s1*G1+(1-s1)*G2)",
            for three phases: "s0*(s1*G1+s2*G2+(1-(s1+s2))*G3)",
            ...

        Notes
        -----
        Planned features:
            - Support registerFunction for custom equations.
        """
        contribution = FitContribution("pdfcontribution")
        contribution.setProfile(self.profile)
        for pdfgenerator in self.pdfgenerators:
            contribution.addProfileGenerator(pdfgenerator)
        contribution.setEquation(equation_string)
        self.contribution = contribution
        return self.contribution

    def initialize_recipe(
        self,
    ):
        """Initialize the FitRecipe object for the fitting process.

        The target output, FitRecipe, requires a profile object, multiple
        PDFGenerator objects, and a FitContribution object combining them. This
        method creates the FitRecipe object combining the profile, PDF
        generators, and contribution.

        Except delta1, delta2, qdamp, qbroad, and the spacegroup parameters,
        other parameters are not added to the recipe by default.

        Must be called after initialize_contribution.

        Notes
        -----
        Planned features:
            - support instructions to
                - add variables
                - constrain variables of the scatters
                - change symmetry constraints
        """

        recipe = FitRecipe()
        recipe.addContribution(self.contribution)
        qdamp = recipe.newVar("qdamp", fixed=False, value=0.04)
        qbroad = recipe.newVar("qbroad", fixed=False, value=0.02)
        for i, (pdfgenerator, spacegroup) in enumerate(
            zip(self.pdfgenerators, self.spacegroups)
        ):
            for pname in [
                "delta1",
                "delta2",
            ]:
                par = getattr(pdfgenerator, pname)
                recipe.addVar(
                    par, name=f"{pdfgenerator.name}_{pname}", fixed=False
                )
            recipe.constrain(pdfgenerator.qdamp, qdamp)
            recipe.constrain(pdfgenerator.qbroad, qbroad)
            stru_parset = pdfgenerator.phase
            spacegroupparams = constrainAsSpaceGroup(stru_parset, spacegroup)
            for par in spacegroupparams.xyzpars:
                recipe.addVar(
                    par, name=f"{pdfgenerator.name}_{par.name}", fixed=False
                )
            for par in spacegroupparams.latpars:
                recipe.addVar(
                    par, name=f"{pdfgenerator.name}_{par.name}", fixed=False
                )
            for par in spacegroupparams.adppars:
                recipe.addVar(
                    par, name=f"{pdfgenerator.name}_{par.name}", fixed=False
                )
        recipe.fithooks[0].verbose = 0
        self.recipe = recipe

    def set_initial_variable_values(self, variable_name_to_value: dict):
        """Update parameter values from the provided dictionary.

        Parameters
        ----------
        variable_name_to_value : dict
            A dictionary mapping variable names to their new values.
        """
        for vname, vvalue in variable_name_to_value.items():
            self.recipe._parameters[vname].setValue(vvalue)

    def get_results(self):
        """Save the fitting results. Must be called after
        refine_variables.

        Returns
        -------
        dict
            The fitting results in a JSON-compatible dictionary format.
        """
        fit_results = FitResults(self.recipe)
        results_dict = {}
        results_dict["residual"] = fit_results.residual
        results_dict["contributions"] = (
            fit_results.residual - fit_results.penalty
        )
        results_dict["restraints"] = fit_results.penalty
        results_dict["chi2"] = fit_results.chi2
        results_dict["reduced_chi2"] = fit_results.rchi2
        results_dict["rw"] = fit_results.rw
        # variables
        results_dict["variables"] = {}
        for name, val, unc in zip(
            fit_results.varnames, fit_results.varvals, fit_results.varunc
        ):
            results_dict["variables"][name] = {
                "value": val,
                "uncertainty": unc,
            }
        # fixed variables
        results_dict["fixed_variables"] = {}
        if fit_results.fixednames is not None:
            for name, val in zip(
                fit_results.fixednames, fit_results.fixedvals
            ):
                results_dict["fixed_variables"][name] = {"value": val}
        # constraints
        results_dict["constraints"] = {}
        if fit_results.connames and fit_results.showcon:
            for con in fit_results.conresults.values():
                for i, loc in enumerate(con.conlocs):
                    names = [obj.name for obj in loc]
                    name = ".".join(names)
                    val = con.convals[i]
                    unc = con.conuncs[i]
                    results_dict["constraints"][name] = {
                        "value": val,
                        "uncertainty": unc,
                    }
        # covariance matrix
        results_dict["covariance_matrix"] = fit_results.cov.tolist()
        # certainty
        certain = True
        for con in fit_results.conresults.values():
            if (con.dy == 1).all():
                certain = False
        results_dict["certain"] = certain
        return results_dict

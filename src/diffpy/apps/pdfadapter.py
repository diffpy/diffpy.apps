import json
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
from scipy.optimize import least_squares


class PDFAdapter:
    """Build and configure a PDF fitting workflow backed by
    diffpy.srfit.

    The adapter assembles the profile, structure generators, contribution,
    and recipe objects required for a PDF refinement, then exposes the fit
    results in a JSON-compatible dictionary.

    Public methods
    --------------
    initialize_profile(profile_path: str, q_range=None, calculation_range=None)
        Load the experimental PDF profile.
    initialize_structures(
        structure_paths: list[str], run_parallel=True, spacegroups=None,
        names=None
    )
        Load structures and create PDFGenerator objects for them.
    initialize_contribution(equation_string=None)
        Create the FitContribution from the loaded profile and generators.
    initialize_recipe()
        Create the FitRecipe for the current contribution.
    set_initial_variable_values(variable_name_to_value: dict)
        Set recipe parameter values by name.
    get_results()
        Return the current fit results as a JSON-compatible dictionary.
    """

    def initialize_profile(
        self,
        profile_path: str,
        q_range=None,
        calculation_range=None,
    ):
        """Load the experimental PDF profile.

        Parameters
        ----------
        profile_path : str
            The path to the experimental PDF profile file.
        q_range : sequence of float, optional
            The two-element sequence containing qmin and qmax. When omitted,
            the values parsed from the profile file are used.
        calculation_range : list or tuple of three floats, optional
            The three-element sequence or mapping defining xmin, xmax, and dx
            for the calculation range. When omitted, the range parsed from the
            profile file is used.
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
        """Load structures and create PDFGenerator objects for them.

        This method should be called after initialize_profile.

        Parameters
        ----------
        structure_paths : list of str
            The paths to structure files in CIF format.
        run_parallel : bool
            If True, enable generator parallelization when the optional
            dependencies are available.
        spacegroups : list of str or None
            The space groups are inferred from the parsed CIF
            files when available and otherwise default to "P1". They will be
            used to impose symmetry constraints on the structure parameters
            during fitting.
        names : list of str or None
            The names assigned to the generators. Missing names default to
            "G1", "G2", and so on.
        """
        if isinstance(structure_paths, str):
            structure_paths = [structure_paths]
        structures = []
        have_spacegroups = False
        if spacegroups is not None:
            if len(spacegroups) != len(structure_paths):
                raise ValueError(
                    f"spacegroups list {spacegroups} must match "
                    f"structure_paths list {structure_paths}. "
                    "Please provide a space group for each structure or set "
                    "spacegroups to None to infer them automatically."
                )
            else:
                have_spacegroups = True
        if not have_spacegroups:
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
            if not have_spacegroups:
                spacegroups.append(spacegroup)
            else:
                if spacegroups[i] == "auto":
                    spacegroups[i] = spacegroup
            pdfgenerator = PDFGenerator(name)
            pdfgenerator.setStructure(structure)
            if run_parallel:
                pdfgenerator.parallel(ncpu=ncpu, mapfunc=self.pool.map)
            pdfgenerators.append(pdfgenerator)
        self.spacegroups = spacegroups
        self.pdfgenerators = pdfgenerators

    def initialize_contribution(self, equation=None):
        """Create the FitContribution from the loaded profile and
        generators.

        This method should be called after initialize_profile and
        initialize_structures.

        Parameters
        ----------
        equation : list of str, optional
            The list of a single equation passed to
            FitContribution.setEquation.

        Returns
        -------
        FitContribution
            The configured contribution object.
        """
        equation = equation[0] if equation is not None else None
        contribution = FitContribution("pdfcontribution")
        contribution.setProfile(self.profile)
        for pdfgenerator in self.pdfgenerators:
            contribution.addProfileGenerator(pdfgenerator)
        contribution.setEquation(equation)
        self.contribution = contribution
        return self.contribution

    def initialize_recipe(
        self,
    ):
        """Create the FitRecipe for the current contribution.

        This method should be called after initialize_contribution. The
        recipe includes shared qdamp and qbroad variables, per-generator
        delta1 and delta2 variables, and structure parameters
        constrained by the space group.
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

    def add_contribution_variables(self, variable_names):
        """Add contribution parameters to the recipe.

        Parameters
        ----------
        variable_names : list of str
            The names of the variables to be added to the recipe.
            e.g. 's0' for scale factor.
        """
        for var_name in variable_names:
            self.recipe.addVar(
                getattr(self.contribution, var_name),
                name=var_name,
                fixed=False,
            )

    def refine_variables(self, variable_names):
        """Refine the specified variables.

        Parameters
        ----------
        variable_names : list of str
            The names of the variables to be refined.
        """
        self.recipe.fix("all")
        for var in variable_names:
            self.recipe.free(var)
            least_squares(self.recipe.residual, self.recipe.values)

    def set_initial_variable_values(self, variable_name_to_value: dict):
        """Set recipe parameter values by name.

        Parameters
        ----------
        variable_name_to_value : dict
            Mapping from recipe variable names to new values.
        """
        for vname, vvalue in variable_name_to_value.items():
            self.recipe._parameters[vname].setValue(vvalue)

    def get_results(self):
        """Return the current fit results as a JSON-compatible
        dictionary.

        Returns
        -------
        dict
            Residual statistics, variable values, constraints, the covariance
            matrix, and a certainty flag.
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

    def save_results(self, result_path):
        with open(result_path, "w") as f:
            json.dump(self.get_results(), f, indent=4)

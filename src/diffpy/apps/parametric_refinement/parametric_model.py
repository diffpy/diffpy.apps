import logging
import re
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path

import networkx as nx
import numpy
from pyobjcryst import loadCrystal
from pyobjcryst.crystal import Crystal
from pyobjcryst.molecule import Molecule

from diffpy.srfit.fitbase import FitContribution
from diffpy.srfit.fitbase.parameter import Parameter, ParameterProxy
from diffpy.srfit.pdf.debyepdfgenerator import DebyePDFGenerator
from diffpy.srfit.pdf.pdfgenerator import PDFGenerator
from diffpy.srfit.structure import constrain_as_space_group
from diffpy.srfit.structure.diffpyparset import DiffpyStructureParSet
from diffpy.srfit.structure.objcrystparset import (
    ObjCrystCrystalParSet,
    ObjCrystMolAtomParSet,
    ObjCrystMoleculeParSet,
)
from diffpy.srfit.structure.srrealparset import SrRealParSet
from diffpy.structure import Structure
from diffpy.structure.parsers import get_parser

# NOTE: MCP server prefers logging for output
logger = logging.getLogger(__name__)


class ParametricModelABC(ABC):
    """Abstract base class for parametric models."""

    @abstractmethod
    def set_profile(self, profile, **kwargs):
        pass

    @property
    @abstractmethod
    def operator(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def residual(self):
        pass


class ParametricModelBase:
    """Base class for parametric models providing common functionality
    for managing parameters and constructing a parameter graph.

    Attributes
    ----------
    name : str
        The name of the parametric model.
    calc_obj : object
        The calculation object associated with the model.
    parset_obj : diffpy.srfit.fitbase.recipeorganizer.RecipeOrganizer
        The parameter set object associated with the model.
    _graph : networkx.DiGraph
        The directed graph representing the parameter hierarchy.
    """

    def __init__(self, name):
        self.name = name
        self.calc_obj = None
        self.parset_obj = None
        self._graph = nx.DiGraph()

    def _construct_parameter_graph(
        self, parset_obj, prefix="", old_graph=None
    ):
        parent_name = f"{prefix}{parset_obj.name}"
        self._graph.add_node(parent_name, obj=parset_obj)
        for par in parset_obj._iter_local_parameters(regexp=re.compile("")):
            par_node_id = f"{parent_name}.{par.name}"
            if not old_graph or par_node_id not in old_graph.nodes:
                self._graph.add_node(
                    par_node_id,
                    obj=par,
                    constrained_or_constant=False,
                )
            else:
                self._graph.add_node(
                    par_node_id,
                    obj=par,
                    constrained_or_constant=old_graph.nodes[par_node_id][
                        "constrained_or_constant"
                    ],
                )
            self._graph.add_edge(parent_name, par_node_id)
        for obj in parset_obj._iter_managed_parameter_containers():
            if hasattr(obj, "_iter_managed_parameter_containers"):
                child_name = f"{parent_name}.{obj.name}"
                # obj is handled as unconstrained by default
                self._graph.add_node(
                    child_name,
                    obj=None,
                    constrained_or_constant=False,
                )
                self._graph.add_edge(parent_name, child_name)
                self._construct_parameter_graph(
                    obj, prefix=f"{parent_name}.", old_graph=old_graph
                )

    @property
    def parameters(self):
        return {
            par_node_id: self._graph.nodes[par_node_id]["obj"]
            for par_node_id in self._graph.nodes
            if isinstance(self._graph.nodes[par_node_id]["obj"], Parameter)
        }

    @property
    def independent_parameters(self):
        return {
            par_node_id: self._graph.nodes[par_node_id]["obj"]
            for par_node_id in self._graph.nodes
            if isinstance(self._graph.nodes[par_node_id]["obj"], Parameter)
            and not (
                (
                    hasattr(self._graph.nodes[par_node_id]["obj"], "const")
                    and self._graph.nodes[par_node_id]["obj"].const
                )
                or self._graph.nodes[par_node_id]["obj"].constrained
                # NOTE: this is a workaround for the constraints not reflected
                #   in par.constrained
                or self._graph.nodes[par_node_id]["constrained_or_constant"]
            )
        }

    def _rebuild_graph(self):
        old_graph = self._graph
        self._graph = nx.DiGraph()
        self._construct_parameter_graph(
            self.parset_obj, prefix="", old_graph=old_graph
        )


ParametricModelABC.register(ParametricModelBase)


class ParametricModelEquation(ParametricModelBase, ParametricModelABC):
    """Parametric model based on an equation.

    Attributes
    ----------
    _contribution : diffpy.srfit.fitbase.FitContribution
        The fit contribution object associated with the model.
    calc_obj : diffpy.srfit.fitbase.FitContribution
        Alias for the fit contribution object.
    parset_obj : diffpy.srfit.fitbase.FitContribution
        Alias for the fit contribution object.
    _submodels : list
        List of registered submodels.
    """

    def __init__(self, name, equation_string):
        super().__init__(name=name)
        self._contribution = FitContribution(name)
        self.calc_obj = self._contribution
        self.parset_obj = self._contribution
        self._submodels = []
        self.set_equation(equation_string)

    def set_equation(self, equation_string):
        self.equation_string = equation_string
        self._contribution.set_equation(equation_string)
        self._rebuild_graph()

    @property
    def operator(self):
        return self._contribution._eq

    def set_profile(self, profile, xname=None, yname=None, dyname=None):
        self._contribution.set_profile(
            profile, xname=xname, yname=yname, dyname=dyname
        )
        for submodel in self._submodels:
            if hasattr(submodel, "set_profile"):
                submodel.set_profile(
                    profile, xname=xname, yname=yname, dyname=dyname
                )

    def evaluate(self):
        if (
            hasattr(self._contribution, "profile")
            and self._contribution.profile is not None
        ):
            yc = self._contribution._eq()
            self._contribution.profile.ycalc = yc
            return yc
        yc = self._contribution._eq()
        return yc

    def residual(self):
        return self._contribution.residual()

    def register_submodel(self, submodel, symbol=None):
        if symbol is None:
            symbol = submodel.name
        # add submodule's calculation model
        if symbol in self._contribution._parameters:
            self._contribution._remove_parameter(
                self._contribution._parameters[symbol]
            )
        if isinstance(submodel, ParametricModelPDF):
            if symbol != submodel.name:
                logger.warning(
                    f"ParametricModelPDF's name ({submodel.name}) does "
                    f"not match with the provided symbol ({symbol}). "
                    f"{submodel.name} will be used as the symbol."
                )
            self._contribution.add_profile_generator(submodel._generator)
        elif isinstance(submodel, ParametricModelFunction) or isinstance(
            submodel, ParametricModelEquation
        ):
            self._contribution._eqfactory.registerOperator(
                symbol, submodel.operator
            )
            self._contribution.add_parameter_set(submodel.parset_obj)
        else:
            raise NotImplementedError(
                "Only ParametricModelPDF, ParametricModelEquation, "
                "and ParametricModelFunction "
                "instances are supported to be registered as submodels."
            )
        self.set_equation(self.equation_string)
        # add submodel's subgraph
        submodel._rebuild_graph()
        if f"{self.name}.{submodel.name}" not in self._graph.nodes:
            self._graph.add_node(
                symbol,
                obj=submodel,
                constrained_or_constant=False,
            )
            self._graph.add_edge(self.name, symbol)
        subgraph = submodel._graph.copy()
        mapping = {node: f"{self.name}.{node}" for node in subgraph.nodes}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        self._graph = nx.compose(self._graph, subgraph)
        self._submodels.append(submodel)
        self._rebuild_graph()


class ParametricModelFunction(ParametricModelEquation):
    """Parametric model based on a function."""

    def __init__(self, name, function, symbol=None, argnames=None):
        """Initialize a ParametricModelFunction instance.

        function can be either a callable or a string representing
        the pre-defined function.
        One and only one of func or characteristic_func_name must be provided.

        Parameters
        ----------
        name: str
            The name of the parametric model function.
        symbol: str
            The symbol representing the function in the equation.
        function: str or callable
            The function associated with the model. Allowed str values are:
                "spherical_particle",
                "spheroidal_particle",
                "lognormal_spherical_particle",
                "sheet_particle",
                "shell_particle",
                "SASCF",
                "sphericalCF",
                "spheroidalCF",
                "spheroidalCF2",
                "lognormalSphericalCF",
                "sheetCF",
                "shellCF",
                "shellCF2",
        argnames: list of str, optional
            The names of the arguments for the function.
        """
        if isinstance(function, str):
            import diffpy.srfit.pdf.characteristicfunctions

            function = getattr(
                diffpy.srfit.pdf.characteristicfunctions,
                function,
            )
        if symbol is None:
            symbol = function.__name__
        ParametricModelBase.__init__(self, name=name)
        self._contribution = FitContribution(name)
        self.calc_obj = self._contribution
        self.parset_obj = self._contribution
        self._submodels = []
        self._function = function
        self._argnames = argnames
        self._contribution.register_function(
            function=function, name=symbol, argnames=argnames
        )
        self.set_equation(symbol)
        self._rebuild_graph()


class ParametricModelPDF(ParametricModelBase, ParametricModelABC):
    """Parametric model based on a calculator for generating pair
    distribution function (PDF).

    NOTE: qmin, qmax, stype(scattering type) are meta handled throughout
    the loaded profile in the refinement session

    Attributes
    ----------
    _generator : PDFGenerator or DebyePDFGenerator
        The generator object used internally for PDF calculations.
    calc_obj: PDFGenerator or DebyePDFGenerator
        alias for _generator.
    parset_obj: FitContribution
        The parameter set object associated with the model.
    space_group_symbol: str
        The space group symbol of the structure.
    sgpar_names: list of str
        The names of the space group parameters.
    """

    def __init__(
        self,
        name,
        structure,
        finite,
        spacegroup_symbol="P1",
        run_parallel=True,
    ):
        """Create a ParametricModelPDF instance from a structure object.

        structure can be a raw diffpy.structure/pyobjcryst structure, or
        an existing DiffpyStructureParSet/ObjCrystCrystalParSet phase
        (e.g. shared from another ParametricModelPDF).
        """
        ParametricModelBase.__init__(self, name=name)

        if not finite:
            self._generator = PDFGenerator(name)
        else:
            self._generator = DebyePDFGenerator(name)
        if isinstance(structure, SrRealParSet):
            self._generator.setPhase(structure)
        elif isinstance(structure, (Structure, Crystal, Molecule)):
            self._generator.setStructure(structure)
        self.calc_obj = self._generator
        self.parset_obj = self._generator
        self._contribution = self.calc_obj

        self._set_parallel(run_parallel)
        self.space_group_symbol = spacegroup_symbol
        self.sgpar_names = []
        self.free_sgpar_names = []
        if isinstance(self._generator.phase, DiffpyStructureParSet):
            self.use_uiso = True
        elif isinstance(self._generator.phase, ObjCrystCrystalParSet):
            self.use_uiso = False
        else:
            logger.warning(
                "Unknown phase type for the generator. "
                "use_uiso is set to False by default."
            )
            self.use_uiso = False
        self._rebuild_graph()

    def _rebuild_graph(self):
        ParametricModelBase._rebuild_graph(self)
        self._prepare_graph()

    def set_profile(self, profile, *args, **kwargs):
        # generator doesn't accept xname, yname, dyname arguments
        self._generator.set_profile(profile)

    @property
    def operator(self):
        logging.error("Accessing the operator of the generator.")
        raise NotImplementedError()

    def evaluate(self):
        if (
            not hasattr(self._generator, "profile")
            or self._generator.profile is None
        ):
            raise RuntimeError(
                "Unable to evaluate because the profile is not set. "
                "Please set the profile before evaluating."
            )
        return self._generator.operation()

    def residual(self):
        ycalc = self.calc_obj.operation()
        yobs = self.calc_obj.profile.ypar.value
        dyobs = self.calc_obj.profile.dypar.value
        return (ycalc - yobs) / dyobs

    def _prepare_graph(self):
        if self.use_uiso:
            dependent_par_names = [
                r"\.U21$",
                r"\.U31$",
                r"\.U32$",  # U21=U12, U31=U13, U32=U23
                r"\.Biso",
                r"\.B\d{2}",  # Bij = Uij * 8 * pi^2
                r"\.occupancy$",  # occupancy=oc
            ]
        else:
            dependent_par_names = [
                r"\.B21$",
                r"\.B31$",
                r"\.B32$",
                r"\.Uiso",
                r"\.U\d{2}",
                r"\.occupancy$",  # occupancy=oc
            ]
        regex = re.compile("|".join(dependent_par_names))
        for par_name in self.parameters.keys():
            if regex.search(par_name):
                self._graph.nodes[par_name]["constrained_or_constant"] = True

    def _set_parallel(self, run_parallel):
        if run_parallel:
            try:
                import multiprocessing
                from multiprocessing import Pool

                import psutil
            except ImportError:
                logger.warning(
                    "Parallelization is disabled because the required "
                    "packages are not installed."
                )
                return
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = numpy.floor(
                (100 - cpu_percent) / (100.0 / syst_cores)
            )
            ncpu = int(numpy.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            self._generator.parallel(ncpu=ncpu, mapfunc=pool.map)

    def constrain_symmetry(self, spacegroup_symbol=None):
        if spacegroup_symbol is None:
            spacegroup_symbol = self.space_group_symbol
        space_group_parset = constrain_as_space_group(
            self._generator.phase, spacegroup_symbol
        )
        self.space_group_symbol = spacegroup_symbol
        # hide constrained parameters in the graph
        if self.use_uiso:
            symmetry_par_names = [
                r"\.a$",
                r"\.b$",
                r"\.c$",
                r"\.alpha$",
                r"\.beta$",
                r"\.gamma$",
                r"\.x$",
                r"\.y$",
                r"\.z$",
                r"\.Uiso$",
                r"\.U11$",
                r"\.U22$",
                r"\.U33$",
                r"\.U12$",
                r"\.U13$",
                r"\.U23$",
            ]
        else:
            symmetry_par_names = [
                r"\.a$",
                r"\.b$",
                r"\.c$",
                r"\.alpha$",
                r"\.beta$",
                r"\.gamma$",
                r"\.x$",
                r"\.y$",
                r"\.z$",
                r"\.Biso$",
                r"\.B11$",
                r"\.B22$",
                r"\.B33$",
                r"\.B12$",
                r"\.B13$",
                r"\.B23$",
            ]
        free_variables = []
        if isinstance(self._generator.phase, DiffpyStructureParSet):
            for latpar in space_group_parset.latpars:
                free_variables.append(latpar)
            for adpar in space_group_parset.adppars:
                free_variables.append(adpar)
            for xyzpar in space_group_parset.xyzpars:
                free_variables.append(xyzpar)
        elif isinstance(self._generator.phase, ObjCrystCrystalParSet):
            for par in self._generator.phase.sgpars:
                free_variables.append(par)
        for i in range(len(free_variables)):
            while isinstance(free_variables[i], ParameterProxy):
                free_variables[i] = free_variables[i].par
            organized_name = ".".join(
                [
                    obj.name
                    for obj in self._generator._locate_managed_object(
                        free_variables[i]
                    )
                ]
            )
            self.sgpar_names.append(organized_name)
        symmetry_par_regex = re.compile("|".join(symmetry_par_names))
        for par_name, par in self.parameters.items():
            if symmetry_par_regex.search(par_name):
                while isinstance(par, ParameterProxy):
                    par = par.par
                if par not in free_variables:
                    self._graph.nodes[par_name][
                        "constrained_or_constant"
                    ] = True
                else:
                    self.free_sgpar_names.append(par_name)
                self.sgpar_names.append(par_name)

    def check_molecule_or_molatom(func):
        @wraps(func)  # preserves name, docstring, signature metadata
        def wrapper(self, *args, **kwargs):
            if not (
                isinstance(self._generator.phase, ObjCrystMoleculeParSet)
                or isinstance(self._generator.phase, ObjCrystMolAtomParSet)
            ):
                logging.warning(
                    "The method %s is only applicable to "
                    "ObjCrystMoleculeParSet or "
                    "ObjCrystMolAtomParSet phases.",
                    func.__name__,
                )
                return None
            else:
                return func(self, *args, **kwargs)

        return wrapper

    @check_molecule_or_molatom
    def add_bond_length_parameter(
        self,
        par_name,
        atom1,
        atom2,
        value=None,
        const=None,
        parent_node_name=None,
    ):
        phase = self._generator.phase
        par = phase.addBondLengthParameter(
            par_name, atom1, atom2, value, const
        )
        new_node_name = f"{parent_node_name}.{par_name}"
        self._graph.add_node(
            new_node_name, obj=par, constrained_or_constant=False
        )
        self._graph.add_edge(parent_node_name, new_node_name)

    @check_molecule_or_molatom
    def add_bond_angle_parameter(
        self,
        par_name,
        atom1,
        atom2,
        atom3,
        value=None,
        const=None,
        parent_node_name=None,
    ):
        phase = self._generator.phase
        par = phase.addBondAngleParameter(
            par_name, atom1, atom2, atom3, value, const
        )
        new_node_name = f"{parent_node_name}.{par_name}"
        self._graph.add_node(
            new_node_name, obj=par, constrained_or_constant=False
        )
        self._graph.add_edge(parent_node_name, new_node_name)

    @check_molecule_or_molatom
    def add_dihedral_angle_parameter(
        self,
        par_name,
        atom1,
        atom2,
        atom3,
        atom4,
        value=None,
        const=None,
        parent_node_name=None,
    ):
        phase = self._generator.phase
        par = phase.addDihedralAngleParameter(
            par_name, atom1, atom2, atom3, atom4, value, const
        )
        new_node_name = f"{parent_node_name}.{par_name}"
        self._graph.add_node(
            new_node_name, obj=par, constrained_or_constant=False
        )
        self._graph.add_edge(parent_node_name, new_node_name)

    @check_molecule_or_molatom
    def restrain_bond_length_parameter(
        self, par, length, sigma, delta, scaled=False
    ):
        phase = self._generator.phase
        phase.restrainBondLengthParameter(par, length, sigma, delta, scaled)

    @check_molecule_or_molatom
    def restrain_bond_angle_parameter(
        self, par, angle, sigma, delta, scaled=False
    ):
        phase = self._generator.phase
        phase.restrainBondAngleParameter(par, angle, sigma, delta, scaled)

    @check_molecule_or_molatom
    def restrain_dihedral_angle_parameter(
        self, par, angle, sigma, delta, scaled=False
    ):
        phase = self._generator.phase
        phase.restrainDihedralAngleParameter(par, angle, sigma, delta, scaled)


# NOTE: certain space groups require dual origin handling.
DUAL_ORIGIN_SG_NUMBERS = {
    48,
    50,
    59,
    68,
    70,
    85,
    86,
    88,
    125,
    126,
    129,
    130,
    133,
    134,
    137,
    138,
    141,
    142,
    201,
    203,
    222,
    224,
    227,
    228,
}


def create_pdf_model_from_file(
    name,
    structure_file_path,
    finite,
    library="Diffpy",
    run_parallel=True,
):
    """Create a ParametricModelPDF by parsing a structure file."""
    stru_parser = get_parser("auto")
    structure = stru_parser.parse(Path(structure_file_path).read_text())
    sg = getattr(stru_parser, "spacegroup", None)
    spacegroup_symbol = sg.short_name if sg is not None else "P1"
    if (
        sg is not None and sg.number in DUAL_ORIGIN_SG_NUMBERS
    ) or library == "ObjCryst":
        structure = loadCrystal(structure_file_path)
    return ParametricModelPDF(
        name,
        structure,
        spacegroup_symbol=spacegroup_symbol,
        finite=finite,
        run_parallel=run_parallel,
    )


def create_pdf_model_from_model(name, from_model, finite, run_parallel=True):
    """Create a ParametricModelPDF sharing the phase of from_model."""
    return ParametricModelPDF(
        name,
        from_model.calc_obj.phase,
        spacegroup_symbol=from_model.space_group_symbol,
        finite=finite,
        run_parallel=run_parallel,
    )


def create_pdf_model_from_code(
    name,
    code,
    finite,
    spacegroup_symbol="P1",
    global_namespace=None,
    local_structure_name="structure",
    run_parallel=True,
):
    """Create a ParametricModelPDF by executing code that builds a
    structure.

    The code must assign the structure/crystal object to a variable
    named structure_name (default "structure") in its local namespace.
    """
    if global_namespace is None:
        global_namespace = {}
    local_namespace = {}
    exec(code, global_namespace, local_namespace)
    if local_structure_name not in local_namespace:
        raise ValueError(
            f"Structure named {local_structure_name} not "
            "found in the executed code."
        )
    structure = local_namespace[local_structure_name]
    if not type(structure).__module__.startswith("pyobjcryst"):
        if "spacegroup_symbol" not in local_namespace:
            logging.warning(
                "diffpy.structure.Structure doesn't contain spacegroup "
                "information. Please provide the 'spacegroup_symbol' "
                "variable explicitly in the executed code or the default "
                f"'spacegroup_symbol' {spacegroup_symbol} will be used."
            )
        spacegroup_symbol = local_namespace.get(
            "spacegroup_symbol", spacegroup_symbol
        )
    else:
        crystal = (
            structure.GetCrystal()
            if hasattr(structure, "GetCrystal")
            else structure
        )
        if hasattr(crystal, "get_space_group"):
            spacegroup_symbol = crystal.get_space_group().get_name()
        else:
            logging.warning(
                "Could not determine a space group for the structure named "
                f"{local_structure_name}. The default 'spacegroup_symbol' "
                f"{spacegroup_symbol} will be used."
            )

    return ParametricModelPDF(
        name,
        structure,
        spacegroup_symbol=spacegroup_symbol,
        finite=finite,
        run_parallel=run_parallel,
    )

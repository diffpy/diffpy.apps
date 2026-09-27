import logging
import re
from functools import wraps
from pathlib import Path

import networkx as nx
from pyobjcryst import loadCrystal

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
from diffpy.structure.parsers import get_parser

# NOTE: MCP server prefers logging for output
logger = logging.getLogger(__name__)


class ParametricModel:
    def __init__(self, name):
        self.name = name
        self._contribution = FitContribution(name)
        self.calc_obj = self._contribution
        self._graph = nx.DiGraph()
        # all submodels will share the same profile
        self._submodels = []

    def _construct_parameter_graph(
        self, parameterset, prefix="", old_graph=None
    ):
        parent_name = f"{prefix}{parameterset.name}"
        self._graph.add_node(parent_name, obj=parameterset)
        for par in parameterset._iter_local_parameters(regexp=re.compile("")):
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
        for obj in parameterset._iter_managed_parameter_containers():
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

    def register_submodel(self, submodel, symbol=None):
        if not isinstance(self, ParametricModelEquation):
            raise ValueError(
                "Submodels can only be registered to "
                "ParametricModelEquation instance."
            )
        if symbol is None:
            symbol = submodel.name
        if symbol in self._contribution._parameters:
            self._contribution._remove_parameter(
                self._contribution._parameters[symbol]
            )
        if isinstance(submodel, ParametricModelPDF):
            if symbol != submodel.name:
                logger.warning(
                    f"ParametricModelPDF's name ({submodel.name}) does "
                    f"not match with the provided symbol ({symbol}) ",
                )
            self._contribution.add_profile_generator(submodel.calc_obj)
        elif isinstance(submodel, ParametricModelEquation) or isinstance(
            submodel, ParametricModelFunction
        ):
            self._contribution._eqfactory.registerOperator(
                symbol, submodel._contribution._eq
            )
            self._contribution.add_parameter_set(submodel._contribution)
        else:
            raise NotImplementedError(
                "Only ParametricModelPDF, ParametricModelEquation, "
                "and ParametricModelFunction "
                "instances are supported to be registered as submodels."
            )
        if self.equation_str is not None:
            self._contribution.set_equation(self.equation_str)
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

    def set_profile(self, profile):
        self._contribution.set_profile(profile)
        for submodel in self._submodels:
            if hasattr(submodel, "set_profile"):
                submodel.set_profile(profile)
        self._rebuild_graph()

    def _rebuild_graph(self):
        old_graph = self._graph
        self._graph = nx.DiGraph()
        self._construct_parameter_graph(
            self._contribution, prefix="", old_graph=old_graph
        )

    def evaluate(self):
        raise NotImplementedError(
            "The evaluate method must be implemented by subclasses."
        )

    def residual(self):
        raise NotImplementedError(
            "The residual method must be implemented by subclasses."
        )


class ParametricModelEquation(ParametricModel):
    def __init__(self, name, equation_str=None):
        super().__init__(name=name)
        self.equation_str = None
        if equation_str:
            self.set_equation(equation_str)

    def set_equation(self, equation_str):
        self.equation_str = equation_str
        self._contribution.set_equation(equation_str)
        self._rebuild_graph()

    def get_equation(self):
        return self.equation_str

    def evaluate(self):
        yc = self._contribution._eq()
        if (
            hasattr(self._contribution, "profile")
            and self._contribution.profile is not None
        ):
            self._contribution.profile.ycalc = yc
        return yc

    def residual(self):
        return self._contribution.residual()


class ParametricModelFunction(ParametricModel):
    def __init__(self, name, function, argnames=None):
        """
        Initialize a ParametricModelFunction instance.

        function can be either a callable or a string representing
        the pre-defined function.
        One and only one of func or characteristic_func_name must be provided.
        Allowed value for characteristic_func_name:
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
        """
        super().__init__(name=name)
        if isinstance(function, str):
            import diffpy.srfit.pdf.characteristicfunctions

            function = getattr(
                diffpy.srfit.pdf.characteristicfunctions,
                function,
            )
        self._contribution.register_function(function, argnames=argnames)
        self._contribution.set_equation(function.__name__)
        self.calc_obj = function
        self._rebuild_graph()

    def set_profile(self, profile, xname=None, yname=None, dyname=None):
        self._contribution.set_profile(
            profile, xname=xname, yname=yname, dyname=dyname
        )
        # no submodel is allowed ParametricModelFunction

    def evaluate(self):
        return self._contribution._eq()


class ParametricModelPDF(ParametricModel):
    # NOTE: qmin, qmax, stype(scattering type) are meta handled
    #   throughout the loaded profile in the refinement session
    def __init__(self, name, structure, spacegroup_symbol="P1", finite=False):
        """
        Create a ParametricModelPDF instance from a structure object.

        structure can be a raw diffpy.structure/pyobjcryst structure, or
        an existing DiffpyStructureParSet/ObjCrystCrystalParSet phase
        (e.g. shared from another ParametricModelPDF).
        """
        super().__init__(name=name)

        if not finite:
            self.calc_obj = PDFGenerator(name)
        else:
            self.calc_obj = DebyePDFGenerator(name)
        if isinstance(
            structure, (DiffpyStructureParSet, ObjCrystCrystalParSet)
        ):
            # already a phase parset: share it instead of rewrapping it
            self.calc_obj.setPhase(structure)
        else:
            self.calc_obj.setStructure(structure)
        self.space_group_symbol = spacegroup_symbol
        self.sgpar_names = []
        self._rebuild_graph()

    def _hide_dependent_parameters(self, use_uiso=True):
        if use_uiso:
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

    def constrain_symmetry(self, spacegroup_symbol=None, use_uiso=True):
        if spacegroup_symbol is None:
            spacegroup_symbol = self.space_group_symbol
        if isinstance(self.calc_obj.phase, DiffpyStructureParSet):
            space_group_parset = constrain_as_space_group(
                self.calc_obj.phase, spacegroup_symbol
            )
            self._hide_dependent_parameters(use_uiso=use_uiso)
        elif isinstance(self.calc_obj.phase, ObjCrystCrystalParSet):
            if use_uiso is not False:
                use_uiso = False
                logger.warning(
                    "ObjCrystCrystalParSet prefers using the letter B instead "
                    "U for ADP parameters."
                )
            self._hide_dependent_parameters(use_uiso=use_uiso)
        else:
            raise ValueError(
                "Unsupported calculation object type."
                "Currently supported types are "
                "DiffpyStructureParSet and ObjCrystCrystalParSet."
            )
        # hide constrained parameters in the graph
        if use_uiso:
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
        if isinstance(self.calc_obj.phase, DiffpyStructureParSet):
            for latpar in space_group_parset.latpars:
                free_variables.append(latpar)
            for adpar in space_group_parset.adppars:
                free_variables.append(adpar)
            for xyzpar in space_group_parset.xyzpars:
                free_variables.append(xyzpar)
        elif isinstance(self.calc_obj.phase, ObjCrystCrystalParSet):
            for par in self.calc_obj.phase.sgpars:
                free_variables.append(par)
        for i in range(len(free_variables)):
            while isinstance(free_variables[i], ParameterProxy):
                free_variables[i] = free_variables[i].par
            organized_name = ".".join(
                [
                    obj.name
                    for obj in self.calc_obj._locate_managed_object(
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

    def check_molecule_or_molatom(func):
        @wraps(func)  # preserves name, docstring, signature metadata
        def wrapper(self, *args, **kwargs):
            if not (
                isinstance(self.calc_obj.phase, ObjCrystMoleculeParSet)
                or isinstance(self.calc_obj.phase, ObjCrystMolAtomParSet)
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
        phase = self.calc_obj.phase
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
        phase = self.calc_obj.phase
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
        phase = self.calc_obj.phase
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
        phase = self.calc_obj.phase
        phase.restrainBondLengthParameter(par, length, sigma, delta, scaled)

    @check_molecule_or_molatom
    def restrain_bond_angle_parameter(
        self, par, angle, sigma, delta, scaled=False
    ):
        phase = self.calc_obj.phase
        phase.restrainBondAngleParameter(par, angle, sigma, delta, scaled)

    @check_molecule_or_molatom
    def restrain_dihedral_angle_parameter(
        self, par, angle, sigma, delta, scaled=False
    ):
        phase = self.calc_obj.phase
        phase.restrainDihedralAngleParameter(par, angle, sigma, delta, scaled)

    def _rebuild_graph(self):
        old_graph = self._graph
        self._graph = nx.DiGraph()
        self._construct_parameter_graph(
            # PDFGenerator itself holds parameters
            self.calc_obj,
            prefix="",
            old_graph=old_graph,
        )

    def set_profile(self, profile):
        self.calc_obj.set_profile(profile)
        # no submodel is allowed ParametricModelPDF

    def evaluate(self):
        return self.calc_obj.operation()

    def residual(self):
        ycalc = self.calc_obj.operation()
        yobs = self.calc_obj.profile.ypar.value
        dyobs = self.calc_obj.profile.dypar.value
        return (ycalc - yobs) / dyobs


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
    name, structure_file_path, library="Diffpy", finite=False
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
        name, structure, spacegroup_symbol=spacegroup_symbol, finite=finite
    )


def create_pdf_model_from_model(name, from_model):
    """Create a ParametricModelPDF sharing the phase of from_model."""
    return ParametricModelPDF(
        name,
        from_model.calc_obj.phase,
        spacegroup_symbol=from_model.space_group_symbol,
    )


def create_pdf_model_from_code(
    name,
    code,
    spacegroup_symbol="P1",
    global_namespace={},
    local_structure_name="structure",
    finite=False,
):
    """Create a ParametricModelPDF by executing code that builds a structure.

    The code must assign the structure/crystal object to a variable named
    structure_name (default "structure") in its local namespace.
    """
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
        if hasattr(crystal, "GetSpaceGroup"):
            spacegroup_symbol = crystal.GetSpaceGroup().GetName()
        else:
            logging.warning(
                "Could not determine a space group for the structure named "
                f"{local_structure_name}. The default 'spacegroup_symbol' "
                f"{spacegroup_symbol} will be used."
            )

    return ParametricModelPDF(
        name, structure, spacegroup_symbol=spacegroup_symbol, finite=finite
    )

import logging
import re
from pathlib import Path

import networkx as nx
from pyobjcryst import loadCrystal

from diffpy.srfit.fitbase import FitContribution
from diffpy.srfit.fitbase.parameter import Parameter, ParameterProxy
from diffpy.srfit.pdf.pdfgenerator import PDFGenerator
from diffpy.srfit.structure import constrain_as_space_group
from diffpy.srfit.structure.diffpyparset import DiffpyStructureParSet
from diffpy.srfit.structure.objcrystparset import ObjCrystCrystalParSet
from diffpy.structure.parsers import get_parser

# NOTE: MCP server prefers logging for output
logger = logging.getLogger(__name__)


class ParametricModel:
    def __init__(self, name):
        self.name = name
        self.calc_obj = FitContribution(name)
        self._graph = nx.DiGraph()
        # all submodels will share the same profile
        self._submodels = []

    def _construct_parameter_graph(
        self, parameterset, prefix="", old_graph=None
    ):
        parent_name = f"{prefix}{parameterset.name}"
        self._graph.add_node(parent_name, parameter=parameterset)
        for par in parameterset._iter_local_parameters(regexp=re.compile("")):
            par_node_id = f"{parent_name}.{par.name}"
            if not old_graph or par_node_id not in old_graph.nodes:
                self._graph.add_node(
                    par_node_id,
                    parameter=par,
                    constrained_or_constant=False,
                )
            else:
                self._graph.add_node(
                    par_node_id,
                    parameter=par,
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
                    parameter=None,
                    constrained_or_constant=False,
                )
                self._graph.add_edge(parent_name, child_name)
                self._construct_parameter_graph(
                    obj,
                    prefix=f"{parent_name}.",
                )

    def register_submodel(self, submodel, symbol=None):
        if not isinstance(self, ParametricModelEquation):
            raise ValueError(
                "Submodels can only be registered to "
                "ParametricModelEquation instance."
            )
        if symbol is None:
            symbol = submodel.name
        if symbol in self.calc_obj._parameters:
            self.calc_obj._remove_parameter(self.calc_obj._parameters[symbol])
        if isinstance(submodel, ParametricModelPDF):
            if symbol != submodel.name:
                logger.warning(
                    f"ParametricModelPDF's name ({submodel.name}) does "
                    f"not match with the provided symbol ({symbol}) ",
                )
            self.calc_obj.add_profile_generator(submodel.calc_obj)
        elif isinstance(submodel, ParametricModelEquation):
            self.calc_obj._eqfactory.registerOperator(
                symbol, submodel.calc_obj._eq
            )
            self.calc_obj.add_parameter_set(submodel.calc_obj)
        else:
            raise NotImplementedError(
                "Only ParametricModelPDF and ParametricModelEquation "
                "instances are supported to be registered as submodels."
            )
        if self.equation_str is not None:
            self.calc_obj.set_equation(self.equation_str)
        self._submodels.append(submodel)
        self._rebuild_graph()

    def process_meta_data(self, meta):
        if hasattr(self.calc_obj, "process_meta_data"):
            self.calc_obj.process_meta_data(meta)

    @property
    def parameters(self):
        return {
            par_node_id: self._graph.nodes[par_node_id]["parameter"]
            for par_node_id in self._graph.nodes
            if isinstance(
                self._graph.nodes[par_node_id]["parameter"], Parameter
            )
        }

    @property
    def independent_parameters(self):
        return {
            par_node_id: self._graph.nodes[par_node_id]["parameter"]
            for par_node_id in self._graph.nodes
            if isinstance(
                self._graph.nodes[par_node_id]["parameter"], Parameter
            )
            and not (
                (
                    hasattr(
                        self._graph.nodes[par_node_id]["parameter"], "const"
                    )
                    and self._graph.nodes[par_node_id]["parameter"].const
                )
                or self._graph.nodes[par_node_id]["parameter"].constrained
                # NOTE: this is a workaround for the constraints not reflected
                #   in par.constrained
                or self._graph.nodes[par_node_id]["constrained_or_constant"]
            )
        }

    def set_profile(self, profile):
        self.calc_obj.set_profile(profile)
        for submodel in self._submodels:
            if hasattr(submodel, "set_profile"):
                submodel.set_profile(profile)
        self._rebuild_graph()

    def _rebuild_graph(self):
        old_graph = self._graph
        self._graph.clear()
        self._construct_parameter_graph(
            self.calc_obj, prefix="", old_graph=old_graph
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
    def __init__(self, name, equation_str=None, from_model_name=None):
        super().__init__(name=name)
        self.equation_str = None
        if from_model_name is not None:
            for name, obj in from_model_name.calc_obj.__dict__.items():
                if name not in ["name", "profile", "_observers"]:
                    setattr(self.calc_obj, name, obj)
        if equation_str:
            self.set_equation(equation_str)

    @property
    def _contribution(self):
        return self.calc_obj

    def set_equation(self, equation_str):
        self.equation_str = equation_str
        self.calc_obj.set_equation(equation_str)
        self._rebuild_graph()

    def get_equation(self):
        return self.equation_str

    def evaluate(self):
        yc = self.calc_obj._eq()
        if (
            hasattr(self.calc_obj, "profile")
            and self.calc_obj.profile is not None
        ):
            self.calc_obj.profile.ycalc = yc
        return yc

    def residual(self):
        return self.calc_obj.residual()


class ParametricModelPDF(ParametricModel):
    # NOTE: qmin, qmax, stype(scattering type) are meta handled
    #   throughout the loaded profile in the refinement session
    def __init__(
        self,
        name,
        structure_file_path=None,
        from_model_name=None,
        structure_lib="Diffpy",
    ):
        super().__init__(name=name)
        self.calc_obj = PDFGenerator(name)
        if structure_file_path is not None:
            stru_parser = get_parser("auto")
            structure = stru_parser.parse(
                Path(structure_file_path).read_text()
            )
            sg = getattr(stru_parser, "spacegroup", None)
            self.space_group_symbol = sg.short_name if sg is not None else "P1"
            if structure_lib == "Diffpy":
                self.calc_obj.setStructure(structure)
            elif structure_lib == "PyObjcryst":
                structure = loadCrystal(structure_file_path)
                self.calc_obj.setStructure(structure)
            else:
                raise ValueError(
                    f"Unsupported structure library: {structure_lib} "
                    "Please use 'Diffpy' or 'PyObjcryst'."
                )
        elif from_model_name is not None:
            self.calc_obj.setPhase(from_model_name.calc_obj.phase)
            self.space_group_symbol = from_model_name.space_group_symbol
        else:
            raise ValueError(
                "Either structure_file or from_model must be provided."
            )
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

    def set_profile(self, profile):
        self.calc_obj.set_profile(profile)
        self._yname = self.calc_obj.profile.ypar.name
        self._dyname = self.calc_obj.profile.dypar.name
        # no submodel is allowed ParametricModelPDF

    def evaluate(self):
        return self.calc_obj.operation()

    def residual(self):
        ycalc = self.calc_obj.operation()
        yobs = self.calc_obj.profile.ypar.value
        dyobs = self.calc_obj.profile.dypar.value
        return (ycalc - yobs) / dyobs

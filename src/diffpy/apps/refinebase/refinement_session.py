import uuid
from collections import OrderedDict
from collections.abc import Callable

import numpy
from scipy.optimize import leastsq

from diffpy.apps.refinebase.parametric_model import (
    ParametricModelEquation,
    ParametricModelPDF,
)
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    FitResults,
    Profile,
)


class RefinementSession:
    def __init__(self):
        self.recipes_dict = OrderedDict()
        self.profiles_dict = OrderedDict()
        self.models_dict = OrderedDict()

    def add_profile_from_file(
        self,
        profile_path: str,
        profile_name: str = None,
        xname: str = "x",
        yname: str = "y",
        dyname: str = "dy",
    ):
        if profile_name is not None and profile_name in self.profiles_dict:
            raise ValueError(f"Profile with ID {profile_name} already exists.")
        if profile_name is None:
            profile_name = str(uuid.uuid4())
        profile = Profile()
        if profile_path.endswith(".dat"):
            profile.loadtxt(profile_path)
        else:
            from diffpy.srfit.pdf import PDFParser

            parser = PDFParser()
            parser.parse_file(profile_path)
            profile.load_parsed_data(parser)
        profile.xpar.name = xname
        profile._xname = xname
        profile.ypar.name = yname
        profile._yname = yname
        profile.dypar.name = dyname
        profile._dyname = dyname
        self.profiles_dict[profile_name] = profile

    def add_profile_from_arrays(
        self,
        xarray,
        yarray,
        dx=None,
        dy=None,
        profile_name: str = None,
        xname: str = "x",
        yname: str = "y",
        dyname: str = "dy",
    ):
        if profile_name is not None and profile_name in self.profiles_dict:
            raise ValueError(f"Profile with ID {profile_name} already exists.")
        if profile_name is None:
            profile_name = str(uuid.uuid4())
        profile = Profile()
        profile.setObservedProfile(xarray, yarray, dx=dx, dy=dy)
        profile.xpar.name = xname
        profile._xname = xname
        profile.ypar.name = yname
        profile._yname = yname
        profile.dypar.name = dyname
        profile._dyname = dyname
        self.profiles_dict[profile_name] = profile

    def remove_profile(self, profile_name: str):
        self._get_profile(profile_name)
        del self.profiles_dict[profile_name]

    def check_profile_meta(self, profile_name: str):
        profile = self._get_profile(profile_name)
        return profile.meta

    def update_profile_meta(self, profile_name: str, meta: dict):
        profile = self._get_profile(profile_name)
        profile.meta.update(meta)

    def set_profile_calculation_range(
        self,
        profile_name: str,
        xmin=None,
        xmax=None,
        dx=None,
    ):
        profile = self._get_profile(profile_name)
        profile.set_calculation_range(xmin, xmax, dx)

    def set_profile_calculation_points(self, profile_name: str, x):
        profile = self._get_profile(profile_name)
        profile.set_calculation_points(x)

    def add_equation_model(self, model_name: str, equation_str=None):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelEquation,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        if equation_str is not None:
            model = ParametricModelEquation(model_name, equation_str)
        else:
            raise ValueError(
                "Either equation_str or from_model must be provided."
            )
        self.models_dict[model_name] = model

    def add_pdf_model(
        self,
        model_name: str,
        structure_file_path=None,
        from_model_name=None,
        code=None,
        library="Diffpy",
        global_namespace={},
        local_structure_name="structure",
        finite=False,
    ):
        from diffpy.apps.refinebase.parametric_model import (
            create_pdf_model_from_code,
            create_pdf_model_from_file,
            create_pdf_model_from_model,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        if structure_file_path is not None:
            pdf_model = create_pdf_model_from_file(
                model_name,
                structure_file_path,
                library=library,
                finite=finite,
            )
        elif from_model_name is not None:
            if from_model_name not in self.models_dict:
                raise ValueError(
                    f"Model with ID {from_model_name} does not exist."
                )
            from_model = self.models_dict[from_model_name]
            pdf_model = create_pdf_model_from_model(
                model_name,
                from_model,
            )
        elif code is not None:
            pdf_model = create_pdf_model_from_code(
                model_name,
                code,
                global_namespace=global_namespace,
                local_structure_name=local_structure_name,
                finite=finite,
            )
        else:
            raise ValueError(
                "Either structure_file_path, from_model_name, or code "
                "must be provided."
            )
        self.models_dict[model_name] = pdf_model

    def add_function_model(
        self,
        model_name: str,
        function: Callable | str,
        argnames: list[str] | None = None,
    ):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelFunction,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        function_model = ParametricModelFunction(
            model_name, function, argnames=argnames
        )
        self.models_dict[model_name] = function_model

    def remove_model(self, model_name: str):
        if model_name not in self.models_dict:
            raise ValueError(f"Model with ID {model_name} does not exist.")
        del self.models_dict[model_name]

    def combine_models(
        self,
        parent_model_name: str,
        child_model_names: list[str],
    ):
        if parent_model_name not in self.models_dict:
            raise ValueError(
                f"Parent model '{parent_model_name}' not found in the session."
            )
        for child_model_name in child_model_names:
            if child_model_name not in self.models_dict:
                raise ValueError(
                    f"Child model '{child_model_name}' not "
                    "found in the session."
                )
        parent_model = self.models_dict[parent_model_name]
        for child_model_name in child_model_names:
            child_model = self.models_dict[child_model_name]
            parent_model.register_submodel(child_model)

    def set_model_equation(self, model_name: str, equation: str):
        model = self._get_equation_model(model_name)
        model.set_equation(equation)

    def set_model_residual_equation(
        self, model_name: str, residual_equation: str
    ):
        model = self._get_equation_model(model_name)
        model.set_residual_equation(residual_equation)

    def get_model_residual_equation(self, model_name: str) -> str:
        model = self._get_equation_model(model_name)
        return model.get_residual_equation()

    def set_model_profile(self, model_name: str, profile_name: str):
        if model_name not in self.models_dict:
            raise ValueError(f"Model '{model_name}' not found in the session.")
        if profile_name not in self.profiles_dict:
            raise ValueError(
                f"Profile '{profile_name}' not found in the session."
            )
        model = self.models_dict[model_name]
        profile = self.profiles_dict[profile_name]
        model.set_profile(profile)

    def get_model_residual(self, model_name: str):
        model = self._get_model(model_name)
        if not hasattr(model, "residual"):
            raise ValueError(
                f"Model '{model_name}' does not have a residual method."
            )
        return model.residual()

    def get_model_evaluation(self, model_name: str):
        model = self._get_model(model_name)
        if not hasattr(model, "evaluate"):
            raise ValueError(
                f"Model '{model_name}' does not have an evaluate method."
            )
        return model.evaluate()

    def constrain_pdf_model_space_group_symmetry(
        self, model_name, space_group=None
    ):
        model = self._get_pdf_model(model_name)
        model.constrain_symmetry(space_group)

    def _get_model(self, model_name):
        if model_name not in self.models_dict:
            raise ValueError(f"Model with ID {model_name} does not exist.")
        return self.models_dict[model_name]

    def _get_pdf_model(self, model_name):
        model = self._get_model(model_name)
        if not isinstance(model, ParametricModelPDF):
            raise ValueError(
                f"Model '{model_name}' is not a ParametricModelPDF instance."
            )
        return model

    def _get_equation_model(self, model_name):
        model = self._get_model(model_name)
        if not isinstance(model, ParametricModelEquation):
            raise ValueError(
                f"Model '{model_name}' is not a "
                "ParametricModelEquation instance."
            )
        return model

    def _get_profile(self, profile_name):
        if profile_name not in self.profiles_dict:
            raise ValueError(f"Profile with ID {profile_name} does not exist.")
        return self.profiles_dict[profile_name]

    def _get_object(self, obj_name):
        obj_name_splitted = obj_name.split(".")
        model = self._get_pdf_model(obj_name_splitted[0])
        if obj_name not in model._graph.nodes:
            raise ValueError(
                f"Object with ID {obj_name} does not exist "
                f"in the model {model.name}."
            )
        return model._graph.nodes[obj_name]["obj"]

    def add_pdf_bond_length_parameter(
        self,
        model_name,
        par_name,
        obj1_name,
        obj2_name,
        value=None,
        const=None,
    ):
        model = self._get_pdf_model(model_name)
        atom1 = self._get_object(obj1_name)
        atom2 = self._get_object(obj2_name)
        parent_node_name = ".".join(obj1_name.split(".")[:-1])
        model.add_bond_length_parameter(
            par_name,
            atom1,
            atom2,
            value=value,
            const=const,
            parent_node_name=parent_node_name,
        )

    def add_pdf_bond_angle_parameter(
        self,
        model_name,
        par_name,
        obj1_name,
        obj2_name,
        obj3_name,
        value=None,
        const=None,
    ):
        model = self._get_pdf_model(model_name)
        atom1 = self._get_object(obj1_name)
        atom2 = self._get_object(obj2_name)
        atom3 = self._get_object(obj3_name)
        parent_node_name = ".".join(obj1_name.split(".")[:-1])
        model.add_bond_angle_parameter(
            par_name,
            atom1,
            atom2,
            atom3,
            value=value,
            const=const,
            parent_node_name=parent_node_name,
        )

    def add_pdf_dihedral_angle_parameter(
        self,
        model_name,
        par_name,
        obj1_name,
        obj2_name,
        obj3_name,
        obj4_name,
        value=None,
        const=None,
    ):
        model = self._get_pdf_model(model_name)
        atom1 = self._get_object(obj1_name)
        atom2 = self._get_object(obj2_name)
        atom3 = self._get_object(obj3_name)
        atom4 = self._get_object(obj4_name)
        parent_node_name = ".".join(obj1_name.split(".")[:-1])
        model.add_dihedral_angle_parameter(
            par_name,
            atom1,
            atom2,
            atom3,
            atom4,
            value=value,
            const=const,
            parent_node_name=parent_node_name,
        )

    def restrain_pdf_bond_length_parameter(
        self, variable_name, length, sigma, delta, scaled=False
    ):
        model = self._get_pdf_model(variable_name.split(".")[0])
        par = self.get_variable(variable_name)["obj"]
        return model.restrain_bond_length_parameter(
            par, length, sigma, delta, scaled=scaled
        )

    def restrain_pdf_bond_angle_parameter(
        self, variable_name, angle, sigma, delta, scaled=False
    ):
        model = self._get_pdf_model(variable_name.split(".")[0])
        par = self.get_variable(variable_name)["obj"]
        return model.restrain_bond_angle_parameter(
            par, angle, sigma, delta, scaled=scaled
        )

    def restrain_pdf_dihedral_angle_parameter(
        self, variable_name, angle, sigma, delta, scaled=False
    ):
        model = self._get_pdf_model(variable_name.split(".")[0])
        par = self.get_variable(variable_name)["obj"]
        return model.restrain_dihedral_angle_parameter(
            par, angle, sigma, delta, scaled=scaled
        )

    def set_variables_value(self, name_value_dict):
        for variable_name, value in name_value_dict.items():
            variable = self.get_variable(variable_name)["obj"]
            variable.value = value

    def get_variable(self, variable_name):
        objs = variable_name.split(".")
        if (
            objs[0] not in self.models_dict
            or variable_name not in self.models_dict[objs[0]].parameters
        ):
            for recipe in self.recipes_dict.values():
                if variable_name in recipe._parameters:
                    variable_obj = recipe._parameters[variable_name]
                    return {
                        "name": variable_name,
                        "value": variable_obj.value,
                        "obj": variable_obj,
                    }
            raise ValueError(
                f"Variable '{variable_name}' not found in "
                f"the model '{objs[0]}'."
            )
        variable_obj = self.models_dict[objs[0]].parameters[variable_name]

        return {
            "name": variable_name,
            "value": variable_obj.value,
            "obj": variable_obj,
        }

    def get_model_obj(self, model_name, obj_name):
        model = self._get_model(model_name)
        if obj_name not in model._graph.nodes:
            raise ValueError(
                f"Object '{obj_name}' not found inside model '{model_name}'."
            )
        return model._graph.nodes[obj_name]["obj"]

    def _solve(
        self,
        name,
        profiles,
        models,
        variable_names,
        constraints=None,
        restraints=None,
        weights=None,
        residual_equations=None,
        metas=None,
        verbose_iterations=0,
    ):
        # NOTE: restraints to be implemented
        recipe = FitRecipe()
        self.recipes_dict[name] = recipe
        if weights is None:
            weights = numpy.ones(len(profiles))
        if residual_equations is None:
            residual_equations = ["chiv"] * len(profiles)
        if metas is not None:
            for i in range(len(metas)):
                profiles[i].meta.update(metas[i])
        for i in range(len(models)):
            if isinstance(models[i], ParametricModelEquation):
                models[i].set_profile(profiles[i])
                models[i]._contribution.set_residual_equation(
                    residual_equations[i]
                )
                recipe.add_contribution(
                    models[i]._contribution, weight=weights[i]
                )
            elif isinstance(models[i], ParametricModelPDF):
                contribution = FitContribution(models[i].name)
                contribution.add_profile_generator(models[i].calc_obj)
                contribution.set_profile(profiles[i])
                contribution.set_residual_equation(residual_equations[i])
                models[i].set_profile(profiles[i])
                recipe.add_contribution(contribution, weight=weights[i])

        if constraints:
            for var_name, value in constraints[0].items():
                if var_name not in recipe._parameters:
                    recipe.create_new_variable(var_name, value)
            for constraint_name, constraint_eq_str in constraints[1].items():
                constraint_var = self.get_variable(constraint_name)["obj"]
                recipe.add_constraint(constraint_var, constraint_eq_str)

        variables = [self.get_variable(name)["obj"] for name in variable_names]
        variable_names = ["_".join(name.split(".")) for name in variable_names]
        for i, var in enumerate(variables):
            if var in recipe._parameters.values():
                continue
            recipe.add_variable(var, name=variable_names[i])

        recipe.free("all")
        leastsq(recipe.residual, recipe.getValues())
        # NOTE: non-scalar value will raise error in `get_results_string`
        try:
            result_string = FitResults(recipe).get_results_string()
        except TypeError:
            result_string = "Refinement Finished."
        return result_string

    def solve(
        self,
        profile_names,
        model_names,
        variable_names=[],
        residual_equations=None,
        constraints=None,
        restraints=None,
        name=uuid.uuid4(),
        weights=None,
        metas=None,
        include_sgpars=False,
        verbose_iterations=0,
    ):
        profiles = []
        for profile_name in profile_names:
            if profile_name not in self.profiles_dict:
                raise ValueError(
                    f"Profile '{profile_name}' not found in the session."
                )
            profiles.append(self.profiles_dict[profile_name])

        models = []
        for model_name in model_names:
            if model_name not in self.models_dict:
                raise ValueError(
                    f"Model '{model_name}' not found in the session."
                )
            models.append(self.models_dict[model_name])

        if include_sgpars:
            for model in models:
                if isinstance(model, ParametricModelPDF):
                    for sgpar_name in model.sgpar_names:
                        if sgpar_name in variable_names:
                            continue
                        variable_names.append(sgpar_name)
                else:
                    for submodel in model._submodels:
                        if isinstance(submodel, ParametricModelPDF):
                            for sgpar_name in submodel.sgpar_names:
                                if sgpar_name in variable_names:
                                    continue
                                variable_names.append(sgpar_name)

        return self._solve(
            profiles=profiles,
            models=models,
            variable_names=variable_names,
            residual_equations=residual_equations,
            constraints=constraints,
            restraints=restraints,
            name=name,
            weights=weights,
            metas=metas,
            verbose_iterations=verbose_iterations,
        )

    def plot(self):
        # NOTE: to be implemented
        for id, recipe in self.recipes_dict.items():
            recipe.plot_recipe()

    def clear(self):
        self.profiles_dict.clear()
        self.models_dict.clear()
        self.recipes_dict.clear()

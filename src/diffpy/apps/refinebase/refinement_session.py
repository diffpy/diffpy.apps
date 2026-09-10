import uuid
from collections import OrderedDict
from functools import wraps

import numpy
from scipy.optimize import least_squares

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

    def check_profile_exists(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            profile_name = args[0] if args else kwargs.get("profile_name")
            if profile_name not in self.profiles_dict:
                raise ValueError(
                    f"Profile with ID {profile_name} does not exist."
                )
            return method(self, *args, **kwargs)

        return wrapper

    def check_model_exists(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            model_name = args[0] if args else kwargs.get("model_name")
            if model_name not in self.models_dict:
                raise ValueError(f"Model with ID {model_name} does not exist.")
            return method(self, *args, **kwargs)

        return wrapper

    def add_profile_from_file(
        self, profile_path: str, profile_name: str = None
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
        self.profiles_dict[profile_name] = profile

    def add_profile_from_arrays(
        self, xarray, yarray, dx=None, dy=None, profile_name: str = None
    ):
        if profile_name is not None and profile_name in self.profiles_dict:
            raise ValueError(f"Profile with ID {profile_name} already exists.")
        if profile_name is None:
            profile_name = str(uuid.uuid4())
        profile = Profile()
        profile.setObservedProfile(xarray, yarray, dx=dx, dy=dy)
        self.profiles_dict[profile_name] = profile

    @check_profile_exists
    def remove_profile(self, profile_name: str):
        del self.profiles_dict[profile_name]

    @check_profile_exists
    def check_profile_meta(self, profile_name: str):
        profile = self.profiles_dict[profile_name]
        return profile.meta

    @check_profile_exists
    def update_profile_meta(self, profile_name: str, meta: dict):
        profile = self.profiles_dict[profile_name]
        profile.meta.update(meta)

    @check_profile_exists
    def set_profile_calculation_range(
        self,
        profile_name: str,
        xmin=None,
        xmax=None,
        dx=None,
    ):
        profile = self.profiles_dict[profile_name]
        profile.set_calculation_range(xmin, xmax, dx)

    @check_profile_exists
    def set_profile_calculation_points(self, profile_name: str, x):
        profile = self.profiles_dict[profile_name]
        profile.set_calculation_points(x)

    def add_equation_model(
        self, model_name: str, equation_str=None, from_model_name=None
    ):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelEquation,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        if equation_str is not None:
            model = ParametricModelEquation(model_name, equation_str)
        elif from_model_name is not None:
            if from_model_name not in self.models_dict:
                raise ValueError(
                    f"Model with ID {from_model_name} does not exist."
                )
            model = ParametricModelEquation(
                model_name, from_model_name=from_model_name
            )
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
        structure_lib="Diffpy",
    ):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelPDF,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        if structure_file_path is not None:
            pdf_model = ParametricModelPDF(
                model_name,
                structure_file_path=structure_file_path,
                structure_lib=structure_lib,
            )
        elif from_model_name is not None:
            if from_model_name not in self.models_dict:
                raise ValueError(
                    f"Model with ID {from_model_name} does not exist."
                )
            pdf_model = ParametricModelPDF(
                model_name, from_model_name=self.models_dict[from_model_name]
            )
        else:
            raise ValueError(
                "Either structure_file_path or from_model must be provided."
            )
        self.models_dict[model_name] = pdf_model

    def remove_model(self, model_name: str):
        if model_name not in self.models_dict:
            raise ValueError(f"Model with ID {model_name} does not exist.")
        del self.models_dict[model_name]

    def combine_models(
        self,
        parent_model_name: str,
        child_model_names: list[str],
        symbol: str = None,
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
            parent_model.register_submodel(child_model, symbol)

    @check_model_exists
    def set_model_equation(self, model_name: str, equation: str):
        model = self.models_dict[model_name]
        if not isinstance(model, ParametricModelEquation):
            raise ValueError(
                f"Model '{model_name}' is not a "
                "ParametricModelEquation instance."
            )
        model.set_equation(equation)

    @check_model_exists
    def set_model_residual_equation(
        self, model_name: str, residual_equation: str
    ):
        model = self.models_dict[model_name]
        if not isinstance(model, ParametricModelEquation):
            raise ValueError(
                f"Model '{model_name}' is not a "
                "ParametricModelEquation instance."
            )
        model.set_residual_equation(residual_equation)

    @check_model_exists
    def get_model_residual_equation(self, model_name: str) -> str:
        model = self.models_dict[model_name]
        if not isinstance(model, ParametricModelEquation):
            raise ValueError(
                f"Model '{model_name}' is not a "
                "ParametricModelEquation instance."
            )
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

    @check_model_exists
    def get_model_residual(self, model_name: str):
        model = self.models_dict[model_name]
        if not hasattr(model, "residual"):
            raise ValueError(
                f"Model '{model_name}' does not have a residual method."
            )
        return model.residual()

    @check_model_exists
    def get_model_evaluation(self, model_name: str):
        model = self.models_dict[model_name]
        if not hasattr(model, "evaluate"):
            raise ValueError(
                f"Model '{model_name}' does not have an evaluate method."
            )
        return model.evaluate()

    @check_model_exists
    def constrain_pdf_model_space_group_symmetry(
        self, model_name, space_group=None
    ):
        model = self.models_dict[model_name]
        if not isinstance(model, ParametricModelPDF):
            raise ValueError(
                f"Model '{model_name}' is not a ParametricModel instance."
            )
        model.constrain_symmetry(space_group)

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
                models[i].calc_obj.set_residual_equation(residual_equations[i])
                recipe.add_contribution(models[i].calc_obj, weight=weights[i])
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

        residual_fn = recipe.residual
        if verbose_iterations:
            call_counter = {"n": 0}

            def _verbose_residual(p, _orig=residual_fn, _counter=call_counter):
                r = _orig(p)
                _counter["n"] += 1
                if _counter["n"] <= verbose_iterations:
                    names = recipe.getNames()
                    print(f"--- solve() call {_counter['n']} ---")
                    for pname, pval in zip(names, p):
                        print(f"    {pname} = {pval}")
                    print(
                        "    sum(residual**2) = "
                        f"{numpy.sum(numpy.asarray(r) ** 2):.6f}"
                    )
                return r

            residual_fn = _verbose_residual

        recipe.fix("all")
        for i in range(len(variable_names)):
            recipe.free(variable_names[i])
            least_squares(
                recipe.residual,
                recipe.getValues(),
            )
        return FitResults(recipe).get_results_string()

    def solve(
        self,
        profile_names,
        model_names,
        variable_names,
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

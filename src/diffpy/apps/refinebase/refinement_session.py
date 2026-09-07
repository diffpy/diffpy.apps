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
        xmin,
        xmax,
        dx=None,
    ):
        profile = self.profiles_dict[profile_name]
        profile.set_calculation_range(xmin, xmax, dx)

    @check_profile_exists
    def set_profile_calculation_points(self, profile_name: str, x):
        profile = self.profiles_dict[profile_name]
        profile.set_calculation_points(x)

    def add_model_from_equation(self, model_name: str, equation_str):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelEquation,
        )

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        model = ParametricModelEquation(model_name, equation_str)
        self.models_dict[model_name] = model

    def add_model_from_structure_file(
        self, model_name: str, structure_file_path: str
    ):
        from diffpy.apps.refinebase.parametric_model import (
            ParametricModelPDF,
        )
        from diffpy.structure import Structure

        if model_name in self.models_dict:
            raise ValueError(f"Model with ID {model_name} already exists.")
        stru = Structure()
        stru.read(structure_file_path)
        pdf_model = ParametricModelPDF(model_name, structure=stru)
        self.models_dict[model_name] = pdf_model

    def remove_model(self, model_name: str):
        if model_name not in self.models_dict:
            raise ValueError(f"Model with ID {model_name} does not exist.")
        del self.models_dict[model_name]

    def combine_models(
        self, parent_model_name: str, child_model_name: str, symbol: str = None
    ):
        if parent_model_name not in self.models_dict:
            raise ValueError(
                f"Parent model '{parent_model_name}' not found in the session."
            )
        if child_model_name not in self.models_dict:
            raise ValueError(
                f"Child model '{child_model_name}' not found in the session."
            )
        parent_model = self.models_dict[parent_model_name]
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
        self, model_name, space_group
    ):
        model = self.models_dict[model_name]
        if not isinstance(model, ParametricModelPDF):
            raise ValueError(
                f"Model '{model_name}' is not a ParametricModel instance."
            )
        model.constrain_symmetry(space_group)

    def set_variable_value(self, variable_name, value):
        variable = self.get_variable(variable_name)["obj"]
        variable.value = value

    def get_variable(self, variable_name):
        objs = variable_name.split(".")
        if objs[0] not in self.models_dict:
            raise ValueError(f"Model '{objs[0]}' not found in the session.")
        if variable_name not in self.models_dict[objs[0]].parameters:
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
        profiles,
        models,
        variables,
        id=uuid.uuid4(),
        weights=None,
        initial_values=None,
        metas=None,
    ):
        recipe = FitRecipe()
        self.recipes_dict[id] = recipe
        if weights is None:
            weights = numpy.ones(len(profiles)) / len(profiles)
        if metas is not None:
            for i in range(len(metas)):
                profiles[i].meta.update(metas[i])
        for i in range(len(models)):
            if isinstance(models[i], ParametricModelEquation):
                models[i].set_profile(profiles[i])
                recipe.add_contribution(
                    models[i]._contribution, weight=weights[i]
                )
            elif isinstance(models[i], ParametricModelPDF):
                contribution = FitContribution(models[i].name)
                contribution.add_profile_generator(models[i].calc_obj)
                contribution.set_profile(profiles[i])
                models[i].set_profile(profiles[i])
                recipe.add_contribution(contribution, weight=weights[i])

        # Add variables
        if initial_values is not None:
            for var, val in zip(variables, initial_values):
                var.value = val
        if len(set([var.name for var in variables])) != len(variables):
            raise ValueError(
                "Duplicate variable names found. Please ensure that "
                "each variable to be refined has a unique name."
            )
        for var in variables:
            recipe.add_variable(var)
        # Refine the recipe
        recipe.fix("all")
        for i in range(len(variables)):
            recipe.free(variables[i].name)
            least_squares(recipe.residual, recipe.getValues(), x_scale="jac")
        return FitResults(recipe).get_results_string()

    def solve(
        self,
        profile_names,
        model_names,
        variable_names,
        id=None,
        weights=None,
        initial_values=None,
        metas=None,
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

        variables = []
        for variable_name in variable_names:
            variables.append(self.get_variable(variable_name)["obj"])

        return self._solve(
            profiles=profiles,
            models=models,
            variables=variables,
            id=id,
            weights=weights,
            initial_values=initial_values,
            metas=metas,
        )

    def plot(self):
        # NOTE: to be implemented
        for id, recipe in self.recipes_dict.items():
            recipe.plot_recipe()

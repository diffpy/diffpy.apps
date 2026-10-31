import inspect
import json
import logging
from functools import wraps
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from diffpy.apps.parametric_refinement.refinement_session import (
    RefinementSession,
)

session = RefinementSession()
mcp = MCPServer("diffpy.apps")
logger = logging.getLogger(__name__)


def tool_errors(func):
    """Log exceptions and expose their messages as MCP ToolErrors."""

    def convert(exc):
        logger.exception("Tool %s failed", func.__name__)
        return ToolError(f"{type(exc).__name__}: {exc}")

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ToolError:
                raise
            except Exception as exc:
                raise convert(exc) from exc

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise convert(exc) from exc

    return sync_wrapper


@mcp.tool()
@tool_errors
async def add_profile_from_file(
    profile_path: Annotated[
        str, Field(description="Path to the profile file")
    ],
    profile_name: Annotated[
        str, Field(description="Unique name for the profile")
    ],
    xname: Annotated[
        str, Field(description="Name of the x-axis for the profile")
    ] = "x",
    yname: Annotated[
        str, Field(description="Name of the y-axis for the profile")
    ] = "y",
    dyname: Annotated[
        str, Field(description="Name of the y-uncertainty for the profile")
    ] = "dy",
) -> str:
    """Add a profile to the refinement session from a file."""
    session.add_profile_from_file(
        profile_path,
        profile_name=profile_name,
        xname=xname,
        yname=yname,
        dyname=dyname,
    )
    return f"Profile {profile_name} added successfully."


@mcp.tool()
@tool_errors
async def add_profile_from_arrays(
    profile_name: Annotated[
        str, Field(description="Unique name for the profile")
    ],
    xarray: Annotated[list, Field(description="X-values of the profile")],
    yarray: Annotated[list, Field(description="Y-values of the profile")],
    dy: Annotated[
        list | None, Field(description="Uncertainties in the y-values")
    ] = None,
    xname: Annotated[
        str, Field(description="Name of the x-axis for the profile")
    ] = "x",
    yname: Annotated[
        str, Field(description="Name of the y-axis for the profile")
    ] = "y",
    dyname: Annotated[
        str, Field(description="Name of the y-uncertainty for the profile")
    ] = "dy",
) -> str:
    """Add a profile to the refinement session from arrays."""
    session.add_profile_from_arrays(
        xarray,
        yarray,
        dy=dy,
        profile_name=profile_name,
        xname=xname,
        yname=yname,
        dyname=dyname,
    )
    return f"Profile {profile_name} added successfully."


@mcp.tool()
@tool_errors
async def set_profile_calculation_range(
    profile_name: Annotated[
        str,
        Field(
            description="Name of the profile to set the calculation range for"
        ),
    ],
    xmin: Annotated[
        float | None, Field(description="Start of the calculation range")
    ] = None,
    xmax: Annotated[
        float | None, Field(description="End of the calculation range")
    ] = None,
    dx: Annotated[
        float | None, Field(description="Step size for the calculation range")
    ] = None,
) -> str:
    """Set the calculation range for a profile in the refinement
    session."""
    session.set_profile_calculation_range(
        profile_name, xmin=xmin, xmax=xmax, dx=dx
    )
    return (
        f"Calculation range for profile {profile_name} "
        f"set to ({xmin}, {xmax}) successfully."
    )


@mcp.tool()
@tool_errors
async def check_profile_meta(
    profile_name: Annotated[
        str, Field(description="Name of the profile to check metadata for")
    ],
) -> dict:
    """Check the metadata for a profile in the refinement session."""
    return session.check_profile_meta(profile_name)


@mcp.tool()
@tool_errors
async def update_profile_meta(
    profile_name: Annotated[
        str, Field(description="Name of the profile to update")
    ],
    meta: Annotated[dict, Field(description="Profile metadata to update")],
) -> str:
    """Update metadata for a profile in the refinement session."""
    session.update_profile_meta(profile_name, meta)
    return f"Metadata for profile {profile_name} updated successfully."


@mcp.tool()
@tool_errors
async def remove_profile(
    profile_name: Annotated[
        str, Field(description="Name of the profile to remove")
    ],
) -> str:
    """Remove a profile from the refinement session."""
    session.remove_profile(profile_name)
    return f"Profile {profile_name} removed successfully."


@mcp.tool()
@tool_errors
async def add_equation_model(
    equation_string: Annotated[
        str, Field(description="Equation for the parametric model")
    ],
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
) -> str:
    """Add an equation-based parametric model to the refinement session.

    Parameters
    ----------
    model_name : str
        Name of the parametric model to add.
    equation_string : str
        Equation for the parametric model. e.g. "a * x + b"
    """
    session.add_equation_model(
        equation_string=equation_string,
        model_name=model_name,
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
@tool_errors
async def add_pdf_model(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    structure_file_path: Annotated[
        str | None, Field(description="Path to the structure file")
    ] = None,
    from_model_name: Annotated[
        str | None,
        Field(
            description="Name of the existing model to base the new model on"
        ),
    ] = None,
    finite: Annotated[
        bool,
        Field(
            description=(
                "Whether to use DebyePDFGenerator instead of PDFGenerator"
            )
        ),
    ] = False,
    code: Annotated[
        str | None,
        Field(
            description="Code that builds a structure to base the new model on"
        ),
    ] = None,
    library: Annotated[
        str | None,
        Field(description="Structure library to use ('Diffpy' or 'ObjCryst')"),
    ] = "Diffpy",
    global_namespace: Annotated[
        dict | None, Field(description="Global namespace to execute 'code' in")
    ] = None,
    local_structure_name: Annotated[
        str,
        Field(description="Name of the structure variable assigned by 'code'"),
    ] = "structure",
    run_parallel: Annotated[
        bool,
        Field(description="Whether to run the PDF model in parallel"),
    ] = True,
) -> str:
    """Add a structure-file-based parametric model to the refinement
    session.

    Parameters
    ----------
    model_name : str
        Name of the parametric model to add.
    structure_file_path : str, optional
        Path to the structure file.
    from_model_name : str, optional
        Name of the existing model to base the new model on.
    code : str, optional
        Code that builds a structure to base the new model on.
    library : str, optional
        Structure library to use ('Diffpy' or 'ObjCryst').
    global_namespace : dict, optional
        Global namespace to execute 'code' in.
    local_structure_name : str, optional
        Name of the structure variable assigned by 'code'.
    finite : bool, optional
        Whether to use DebyePDFGenerator instead of PDFGenerator.
    run_parallel : bool, optional
        Whether to run the PDF model in parallel.

    Notes
    -----
    'from_model_name' is used to create a new model referencing to an
    existing computation object. For example, when the same phase's signal
    is observed in multiple profiles, 'from_model_name' allows the refinement
    backend to modify the same structure across multiple profiles.

    Exactly one of 'structure_file_path', 'from_model_name', or 'code'
    must be provided.
    """
    session.add_pdf_model(
        model_name=model_name,
        structure_file_path=structure_file_path,
        from_model_name=from_model_name,
        code=code,
        library=library,
        global_namespace=global_namespace,
        local_structure_name=local_structure_name,
        finite=finite,
        run_parallel=run_parallel,
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
@tool_errors
async def add_function_model(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    function: Annotated[
        Literal[
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
        ],
        Field(description="Name of the predefined function for the model"),
    ],
    argnames: Annotated[
        list[str] | None, Field(description="Argument names for the function")
    ] = None,
) -> str:
    """Add a function model to the refinement session.

    Allowed value for function:
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
    session.add_function_model(
        model_name=model_name,
        function=function,
        argnames=argnames,
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
@tool_errors
async def set_model_equation(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    equation: Annotated[
        str, Field(description="New equation for the parametric model")
    ],
) -> str:
    """Set the equation for an existing parametric model in the
    session."""
    session.set_model_equation(model_name=model_name, equation=equation)
    return f"Equation for model {model_name} set successfully."


@mcp.tool()
@tool_errors
async def get_model_evaluation(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    data_path: Annotated[
        str,
        Field(
            description=(
                "The path to the file where the evaluation will be written"
            )
        ),
    ],
    profile_name: Annotated[
        str | None,
        Field(
            description=(
                "Name of the profile to use for the evaluation. "
                "If None, temporary profile will be created for the "
                "evaluation. "
                "If both profile_name and xmin/xmax/dx are provided, "
                "the profile's calculation range will be updated accordingly."
            )
        ),
    ] = None,
    xmin: Annotated[
        float | None,
        Field(
            description=(
                "Minimum x value for the evaluation "
                "if the profile is not set for the model"
            )
        ),
    ] = None,
    xmax: Annotated[
        float | None,
        Field(
            description=(
                "Maximum x value for the evaluation "
                "if the profile is not set for the model"
            )
        ),
    ] = None,
    dx: Annotated[
        float | None,
        Field(
            description=(
                "Step size for the evaluation "
                "if the profile is not set for the model"
            )
        ),
    ] = None,
    meta: Annotated[
        dict | None,
        Field(
            description=(
                "Metadata to associate with the evaluation. "
                "If None, no additional metadata will be added."
            )
        ),
    ] = None,
    xname: Annotated[
        str,
        Field(
            description=(
                "Name of the x variable for the evaluation "
                "if the profile is not set for the model"
            )
        ),
    ] = "x",
) -> str:
    """Get the current evaluation of a parametric model."""
    r, ycalc = session.get_model_evaluation(
        model_name=model_name,
        profile_name=profile_name,
        xmin=xmin,
        xmax=xmax,
        dx=dx,
        xname=xname,
        meta=meta,
    )
    evaluation = {"r": r.tolist(), "ycalc": ycalc.tolist()}
    with open(data_path, "w") as f:
        json.dump(evaluation, f)
    return (
        f"Evaluation for model {model_name} "
        f"written to {data_path} successfully."
    )


@mcp.tool()
@tool_errors
async def set_model_profile(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    profile_name: Annotated[
        str, Field(description="Name of the profile to set on the model")
    ],
) -> str:
    """Set the profile for a parametric model in the refinement
    session."""
    session.set_model_profile(model_name, profile_name)
    return f"Profile {profile_name} set for model {model_name} successfully."


@mcp.tool()
@tool_errors
async def get_model_residual(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    data_path: Annotated[
        str,
        Field(
            description=(
                "The path to the file where the residual will be written"
            )
        ),
    ],
) -> str:
    """Get the current residual of a parametric model."""
    residual = session.get_model_residual(model_name=model_name).tolist()
    with open(data_path, "w") as f:
        json.dump(residual, f)
    return (
        f"Residual for model {model_name} written to {data_path} successfully."
    )


@mcp.tool()
@tool_errors
async def get_profile_data(
    profile_name: Annotated[
        str, Field(description="Name of the profile to retrieve")
    ],
    data_path: Annotated[
        str,
        Field(
            description=(
                "The path to the file where the profile data will be written"
            )
        ),
    ],
) -> str:
    """Get the details of a specific profile in the refinement
    session."""
    profile = session.profiles_dict[profile_name]
    data = {"xobs": profile.xobs.tolist(), "yobs": profile.yobs.tolist()}
    with open(data_path, "w") as f:
        json.dump(data, f)
    return (
        f"Data for profile {profile_name} written to {data_path} successfully."
    )


@mcp.tool()
@tool_errors
async def remove_model(
    model_name: Annotated[
        str, Field(description="Name of the model to remove")
    ],
) -> str:
    """Remove a model from the refinement session."""
    session.remove_model(model_name)
    return f"Model {model_name} removed successfully."


@mcp.tool()
@tool_errors
async def constrain_pdf_model_space_group_symmetry(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    space_group: Annotated[
        str | None, Field(description="Space group to constrain the model to")
    ] = None,
) -> str:
    """Constrain a parametric model to a specific space group symmetry.

    If no space group is provided, the model will be constrained to its
    current space group parsed from its structure.
    """
    session.constrain_pdf_model_space_group_symmetry(model_name, space_group)
    return (
        f"Model {model_name} constrained to space group "
        f"{session.models_dict[model_name].space_group_symbol} "
        f"successfully."
    )


@mcp.tool()
@tool_errors
async def list_profiles() -> list[str]:
    """List all profiles in the refinement session."""
    return [str(profile_id) for profile_id in session.profiles_dict.keys()]


@mcp.tool()
@tool_errors
async def list_models() -> list[str]:
    """List all models in the refinement session."""
    return [str(model_id) for model_id in session.models_dict.keys()]


@mcp.tool()
@tool_errors
async def combine_models(
    parent_model_name: Annotated[
        str,
        Field(
            description=(
                "Name of the parent model. Must be an equation model "
                "(from add_equation_model)."
            )
        ),
    ],
    child_model_names: Annotated[
        list[str],
        Field(
            description="Names of the models the parent's equation refers to."
        ),
    ],
) -> str:
    """Combine two parametric models by registering the child to the
    parent model."""
    session.combine_models(parent_model_name, child_model_names)
    return (
        f"Models {parent_model_name} and "
        f"{child_model_names} combined successfully."
    )


@mcp.tool()
@tool_errors
async def set_variables_value(
    name_value_dict: Annotated[
        dict,
        Field(
            description=(
                "Mapping of variable names to the values to set them to"
            )
        ),
    ],
) -> str:
    """Set the value of variables in the current session."""
    session.set_variables_value(name_value_dict)

    return f"Variables '{name_value_dict}' are set."


@mcp.tool()
@tool_errors
async def get_variable(
    variable_name: Annotated[str, Field(description="Name of the variable")],
) -> str:
    """Get the value of a specific variable in the current session."""
    variable = session.get_variable(variable_name)
    return f"Variable '{variable_name}': {variable['value']}"


@mcp.tool()
@tool_errors
async def list_model_parameters(
    model_name: Annotated[
        str, Field(description="Name of the parametric model")
    ],
    independent_only: Annotated[
        bool, Field(description="Whether to list only independent parameters")
    ] = True,
) -> str:
    """List all parameters of a specific parametric model."""
    if model_name not in session.models_dict:
        raise ValueError(f"Model with ID {model_name} does not exist.")

    model = session.models_dict[model_name]
    if independent_only:
        parameters = {
            node_id: par.value
            for node_id, par in model.independent_parameters.items()
        }
    else:
        parameters = {
            node_id: par.value for node_id, par in model.parameters.items()
        }
    return f"Parameters for model '{model_name}': {parameters}"


@mcp.tool()
@tool_errors
async def clear() -> str:
    """Clear the current refinement session."""
    session.clear()
    return "Refinement session cleared successfully."


@mcp.tool()
@tool_errors
async def list_recipe_parameters(
    recipe_name: Annotated[str, Field(description="Name of the recipe")],
) -> str:
    """List all parameters of a specific recipe."""
    if recipe_name not in session.recipes_dict:
        raise ValueError(f"Recipe with ID {recipe_name} does not exist.")

    recipe = session.recipes_dict[recipe_name]
    parameters = {var.name: var.value for var in recipe._parameters.values()}
    return f"Parameters for recipe '{recipe_name}': {parameters}"


@mcp.tool()
@tool_errors
async def solve(
    name: Annotated[str, Field(description="Name of the refinement recipe.")],
    profile_names: Annotated[
        list[str],
        Field(description="List of profile IDs to use in the refinement"),
    ],
    model_names: Annotated[
        list[str],
        Field(description="List of model IDs to use in the refinement"),
    ],
    variable_names: Annotated[
        list[str], Field(description="List of variable names to refine")
    ],
    residual_equations: Annotated[
        list[str] | None,
        Field(description="List of residual equations for each profile"),
    ] = None,
    constraints: Annotated[
        list[dict] | None,
        Field(
            description=(
                "First dict is new_variable-initial value pair, "
                "and the second dict is variable-constraint_equation pair."
            )
        ),
    ] = None,
    bounds: Annotated[
        dict | None,
        Field(description="Dictionary of bounds for variables or equations"),
    ] = None,
    weights: Annotated[
        list[float] | None,
        Field(description="List of weights for each refinement profile"),
    ] = None,
    metas: Annotated[
        list[dict] | None,
        Field(description="List of metadata dictionaries for each profile"),
    ] = None,
    include_sgpars: Annotated[
        bool,
        Field(
            description=(
                "Whether to also include sgpars from the models automatically"
            )
        ),
    ] = False,
    method: Annotated[
        str, Field(description="Optimizer method: trf, dogbox, or lm")
    ] = "trf",
    x_scale: Annotated[
        float | str, Field(description="Optimizer parameter scaling, or 'jac'")
    ] = 1.0,
    max_nfev: Annotated[
        int | None, Field(description="Maximum number of function evaluations")
    ] = None,
    ftol: Annotated[
        float, Field(description="Cost-function termination tolerance")
    ] = 1e-8,
    xtol: Annotated[
        float, Field(description="Step-size termination tolerance")
    ] = 1e-8,
    background: Annotated[
        bool, Field(description="Run asynchronously and return a job ID")
    ] = False,
) -> dict | str:
    """Initiate a refinement using the specified profiles, models, and
    variables.

    Parameters
    ----------
    profile_names : list[str]
        List of profile IDs to use in the refinement.
    model_names : list[str]
        List of model IDs to use in the refinement.
    variable_names : list[str]
        List of variable names to refine.
    residual_equations : list[str], optional
        List of residual equations for each profile.
    constraints : list[dict], optional
        First dict is new_variable-initial value pair,
        and the second dict is variable-constraint_equation pair.
    bounds : dict, optional
        Dictionary of bounds for variables or equations.
        e.g. {"variable_name":
        {
            "lower_bound": 0,
            "upper_bound": 10,
            "uncertainty": 1,
            "scaled": False
        }}
        # start copied from diffpy.srfit docstring
        scaled : bool, optional
            If True, the restraint penalty is scaled by the unrestrained
            point-average chi^2 (chi^2/numpoints) (default is False).
        params : dict, optional
            The dictionary of Parameters, indexed by name, that are used in
            `param_or_eq` (if an equation string is used) but are not part
            of the RecipeOrganizer (default is {}).
        # end copied from diffpy.srfit docstring
    name : str
        Name of the refinement recipe.
    weights : list[float], optional
        List of weights for each refinement profile.
    metas : list[dict], optional
        List of metadata dictionaries for each profile.
    include_sgpars : bool, optional
        Whether to also include sgpars from the models automatically.
    method : str, optional
        Least-squares optimizer method.
    x_scale : float or str, optional
        Parameter scaling used by the optimizer.
    max_nfev : int, optional
        Maximum number of residual evaluations.
    ftol, xtol : float, optional
        Optimizer termination tolerances.
    background : bool, optional
        If True, return a job ID immediately and run the solve in the
        background.

    Notes
    -----
    The length of the profile_names, model_names, residual_equations,
    and weights should be the same.

    Constraints should be a list of two dictionaries.
    The first dictionary should contain new variable names as keys and
    their initial values as values.
    The second dictionary should contain variable names as keys and their
    constraint equations as values.
    """
    out_string = session.solve(
        profile_names,
        model_names,
        variable_names,
        residual_equations=residual_equations,
        constraints=constraints,
        bounds=bounds,
        name=name,
        weights=weights,
        metas=metas,
        include_sgpars=include_sgpars,
        method=method,
        x_scale=x_scale,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        background=background,
    )

    return out_string


@mcp.tool()
@tool_errors
async def get_solve_status(
    job_id: Annotated[str, Field(description="Background refinement job ID")],
) -> dict:
    """Return the status and result of a background refinement job."""
    return session.get_job_status(job_id)


@mcp.tool()
@tool_errors
async def list_running_jobs() -> list[dict]:
    """List pending and currently running background refinements."""
    return session.list_running_jobs()


if __name__ == "__main__":
    mcp.run()

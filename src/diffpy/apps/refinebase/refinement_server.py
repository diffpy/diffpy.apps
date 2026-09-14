import inspect
import json
import logging
import uuid
from functools import wraps
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from diffpy.apps.refinebase.refinement_session import RefinementSession

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


@mcp.prompt()
async def refinement_playbook():
    file_path = str(Path(__file__).parent / "refinement_playbook.md")
    with open(file_path, "r") as f:
        return f.read()


@mcp.tool()
@tool_errors
async def add_profile_from_file(
    profile_path: Annotated[str, "Path to the profile file"],
    profile_name: Annotated[str, "Unique name for the profile"] = None,
) -> str:
    """Add a profile to the refinement session from a file."""
    session.add_profile_from_file(profile_path, profile_name=profile_name)
    return f"Profile {profile_name} added successfully."


@mcp.tool()
@tool_errors
async def add_profile_from_arrays(
    xarray: Annotated[list, "X-values of the profile"],
    yarray: Annotated[list, "Y-values of the profile"],
    dx: Annotated[list, "Uncertainties in the x-values"] = None,
    dy: Annotated[list, "Uncertainties in the y-values"] = None,
    profile_name: Annotated[str, "Unique name for the profile"] = None,
) -> str:
    """Add a profile to the refinement session from arrays."""
    session.add_profile_from_arrays(
        xarray, yarray, dx=dx, dy=dy, profile_name=profile_name
    )
    return f"Profile {profile_name} added successfully."


@mcp.tool()
@tool_errors
async def set_profile_calculation_range(
    profile_name: Annotated[
        str, "Name of the profile to set the calculation range for"
    ],
    xmin: Annotated[float, "Start of the calculation range"] = None,
    xmax: Annotated[float, "End of the calculation range"] = None,
    dx: Annotated[float, "Step size for the calculation range"] = None,
) -> str:
    """Set the calculation range for a profile in the refinement session."""
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
    profile_name: Annotated[str, "Name of the profile to check metadata for"],
) -> dict:
    """Check the metadata for a profile in the refinement session."""
    return session.check_profile_meta(profile_name)


@mcp.tool()
@tool_errors
async def update_profile_meta(
    profile_name: Annotated[str, "Name of the profile to update"],
    meta: Annotated[dict, "Profile metadata to update"],
) -> str:
    """Update metadata for a profile in the refinement session."""
    session.update_profile_meta(profile_name, meta)
    return f"Metadata for profile {profile_name} updated successfully."


@mcp.tool()
@tool_errors
async def remove_profile(
    profile_name: Annotated[str, "Name of the profile to remove"],
) -> str:
    """Remove a profile from the refinement session."""
    session.remove_profile(profile_name)
    return f"Profile {profile_name} removed successfully."


@mcp.tool()
@tool_errors
async def add_equation_model(
    model_name: Annotated[str, "Name of the parametric model"] = uuid.uuid4(),
    equation_str: Annotated[str, "Equation for the parametric model"] = None,
) -> str:
    """Add an equation-based parametric model to the refinement session.

    Parameters
    ----------
    model_name : str
        Name of the parametric model to add.
    equation_str : str, optional
        Equation for the parametric model. e.g. "a * x + b"
    """
    session.add_equation_model(
        equation_str=equation_str,
        model_name=model_name,
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
@tool_errors
async def add_pdf_model(
    model_name: Annotated[str, "Name of the parametric model"] = uuid.uuid4(),
    structure_file_path: Annotated[str, "Path to the structure file"] = None,
    from_model_name: Annotated[
        str, "Name of the existing model to base the new model on"
    ] = None,
) -> str:
    """
    Add a structure-file-based parametric model to the refinement session.

    Parameters
    ----------
    model_name : str
        Name of the parametric model to add.
    structure_file_path : str, optional
        Path to the structure file.
    from_model_name : str, optional

    Notes
    -----
    'from_model_name' is used to create a new model referencing to an
    existing computation object. For example, when the same phase's signal
    is observed in multiple profiles, 'from_model_name' allows the refinement
    backend to modify the same structure across multiple profiles.
    """
    session.add_pdf_model(
        model_name=model_name,
        structure_file_path=structure_file_path,
        from_model_name=from_model_name,
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
@tool_errors
async def set_model_equation(
    model_name: Annotated[str, "Name of the parametric model"],
    equation: Annotated[str, "New equation for the parametric model"],
) -> str:
    """Set the equation for an existing parametric model in the session."""
    session.set_model_equation(model_name=model_name, equation=equation)
    return f"Equation for model {model_name} set successfully."


@mcp.tool()
@tool_errors
async def get_model_evaluation(
    model_name: Annotated[str, "Name of the parametric model"],
    data_path: Annotated[str, "Path to the data to retrieve"],
) -> str:
    """Get the current evaluation of a parametric model."""
    evaluation = session.get_model_evaluation(model_name=model_name).tolist()
    with open(data_path, "w") as f:
        json.dump(evaluation, f)
    return (
        f"Evaluation for model {model_name} "
        f"written to {data_path} successfully."
    )


@mcp.tool()
@tool_errors
async def get_model_residual(
    model_name: Annotated[str, "Name of the parametric model"],
    data_path: Annotated[str, "Path to the data to retrieve"],
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
    profile_name: Annotated[str, "Name of the profile to retrieve"],
    data_path: Annotated[str, "Path to the data to retrieve"],
) -> str:
    """Get the details of a specific profile in the refinement session."""
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
    model_name: Annotated[str, "Name of the model to remove"],
) -> str:
    """Remove a model from the refinement session."""
    session.remove_model(model_name)
    return f"Model {model_name} removed successfully."


@mcp.tool()
@tool_errors
async def constrain_pdf_model_space_group_symmetry(
    model_name: Annotated[str, "Name of the parametric model"],
    space_group: Annotated[
        str, "Space group to constrain the model to"
    ] = None,
) -> str:
    """Constrain a parametric model to a specific space group symmetry.

    If no space group is provided, the model will be constrained to
    its current space group parsed from its structure.
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
    parent_model_name: Annotated[str, "Name of the parent parametric model"],
    child_model_names: Annotated[
        list[str], "Names of the child parametric models"
    ],
    symbol: Annotated[
        str, "Symbol to use for child model in the parent model's equation"
    ],
) -> str:
    """
    Combine two parametric models by registering the child to the parent model.
    """
    session.combine_models(parent_model_name, child_model_names, symbol)
    return (
        f"Models {parent_model_name} and "
        f"{child_model_names} combined successfully."
    )


@mcp.tool()
@tool_errors
async def set_variables_value(
    name_value_dict: Annotated[
        dict, "Mapping of variable names to the values to set them to"
    ],
) -> str:
    """
    Set the value of a specific variable in a parametric model.
    """
    session.set_variables_value(name_value_dict)

    return f"Variables '{name_value_dict}' are set."


@mcp.tool()
@tool_errors
async def get_variable(
    variable_name: Annotated[str, "Name of the variable"],
) -> str:
    """
    Get the value of a specific variable in a parametric model.
    """
    variable = session.get_variable(variable_name)
    return f"Variable '{variable_name}': {variable['value']}"


@mcp.tool()
@tool_errors
async def list_model_parameters(
    model_name: Annotated[str, "Name of the parametric model"],
    independent_only: Annotated[
        bool, "Whether to list only independent parameters"
    ] = True,
) -> str:
    """
    List all parameters of a specific parametric model.
    """
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
    """
    Clear the current refinement session.
    """
    session.clear()
    return "Refinement session cleared successfully."


@mcp.tool()
@tool_errors
async def list_recipe_parameters(
    recipe_name: Annotated[str, "Name of the recipe"],
) -> str:
    """
    List all parameters of a specific recipe.
    """
    if recipe_name not in session.recipes_dict:
        raise ValueError(f"Recipe with ID {recipe_name} does not exist.")

    recipe = session.recipes_dict[recipe_name]
    parameters = {var.name: var.value for var in recipe._parameters.values()}
    return f"Parameters for recipe '{recipe_name}': {parameters}"


@mcp.tool()
@tool_errors
async def solve(
    profile_names: Annotated[
        list[str], "List of profile IDs to use in the refinement"
    ],
    model_names: Annotated[
        list[str], "List of model IDs to use in the refinement"
    ],
    variable_names: Annotated[list[str], "List of variable names to refine"],
    residual_equations: Annotated[
        list[str], "List of residual equations for each profile"
    ] = None,
    constraints: Annotated[
        list[dict],
        (
            "First dict is new_variable-initial value pair, "
            "and the second dict is variable-constraint_equation pair."
        ),
    ] = None,
    restraints: Annotated[
        list[str], "List of restraints to apply during the refinement"
    ] = None,
    name: Annotated[str, "Name of the refinement session"] = None,
    weights: Annotated[
        list[float], "List of weights for each refinement profile"
    ] = None,
    metas: Annotated[
        list[dict], "List of metadata dictionaries for each profile"
    ] = None,
    include_sgpars: Annotated[
        bool, "Whether to also include sgpars from the models automatically"
    ] = False,
) -> str:
    """
    Initiate a refinement using the specified profiles, models, and variables.

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
    restraints : list[str], optional
        List of restraints to apply during the refinement.
    name : str, optional
        Name of the refinement session.
    weights : list[float], optional
        List of weights for each refinement profile.
    metas : list[dict], optional
        List of metadata dictionaries for each profile.
    include_sgpars : bool, optional
        Whether to also include sgpars from the models automatically.

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
        restraints=restraints,
        name=name,
        weights=weights,
        metas=metas,
        include_sgpars=include_sgpars,
    )

    return out_string


if __name__ == "__main__":
    mcp.run()

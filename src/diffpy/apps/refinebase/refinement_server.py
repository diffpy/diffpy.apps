import json
import uuid
from typing import Annotated

from mcp.server import MCPServer

from diffpy.apps.refinebase.refinement_session import RefinementSession

session = RefinementSession()
mcp = MCPServer("diffpy.apps")


@mcp.prompt()
def refine_general():
    return """
1. Load the profile
2. Ensure the profile's meta is consistent with the experiment settings.
3. Add the parametric model
4. Check the tunable independent parameters
5. Initialize necessary parameters
6. Perform the refinement
"""


@mcp.prompt()
def refine_include_pdf():
    return """
It follows the general refinement procedure, except some additional procedures
to set up the PDF model.
1. Load the profile
2. Ensure the profile's meta is consistent with the experiment settings.
3. Add the parametric PDF model
4. Constrain the parameters of the PDF model as needed according to it's
    spacegroup symmetry.
5. Add another parametric equation model if needed to consider
additional factors, e.g. scale, and combine them with the PDF model.
6. Check the tunable independent parameters
7. Initialize necessary parameters
8. Perform the refinement
"""


@mcp.prompt()
def plot_get_data():
    return """
After a complete refinement, the data to plot is stored in:
1. profile data
2. model.evaluation
3. model.residual
"""


@mcp.tool()
async def add_profile_from_file(
    profile_path: Annotated[str, "Path to the profile file"],
    profile_name: Annotated[str, "Unique name for the profile"] = None,
) -> str:
    """Add a profile to the refinement session from a file."""
    session.add_profile_from_file(profile_path, profile_name=profile_name)
    return f"Profile {profile_name} added successfully."


@mcp.tool()
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
async def set_profile_calculation_range(
    profile_name: Annotated[
        str, "Name of the profile to set the calculation range for"
    ],
    xmin: Annotated[float, "Start of the calculation range"],
    xmax: Annotated[float, "End of the calculation range"],
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
async def check_profile_meta(
    profile_name: Annotated[str, "Name of the profile to check metadata for"],
) -> dict:
    """Check the metadata for a profile in the refinement session."""
    return session.check_profile_meta(profile_name)


@mcp.tool()
async def update_profile_meta(
    profile_name: Annotated[str, "Name of the profile to update"],
    meta: Annotated[dict, "Profile metadata to update"],
) -> str:
    """Update metadata for a profile in the refinement session."""
    session.update_profile_meta(profile_name, meta)
    return f"Metadata for profile {profile_name} updated successfully."


@mcp.tool()
async def remove_profile(
    profile_name: Annotated[str, "Name of the profile to remove"],
) -> str:
    """Remove a profile from the refinement session."""
    session.remove_profile(profile_name)
    return f"Profile {profile_name} removed successfully."


@mcp.tool()
async def add_model_from_equation(
    equation_str: Annotated[str, "Equation for the parametric model"],
    model_name: Annotated[str, "Name of the parametric model"] = uuid.uuid4(),
) -> str:
    """Add an equation-based parametric model to the refinement session."""
    session.add_model_from_equation(
        equation_str=equation_str, model_name=model_name
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
async def add_model_from_structure_file(
    structure_file_path: Annotated[str, "Path to the structure file"],
    model_name: Annotated[str, "Name of the parametric model"] = uuid.uuid4(),
) -> str:
    """Add a structure-file-based parametric model to the session."""
    session.add_model_from_structure_file(
        structure_file_path=structure_file_path, model_name=model_name
    )
    return f"Model {model_name} added successfully."


@mcp.tool()
async def set_model_equation(
    model_name: Annotated[str, "Name of the parametric model"],
    equation: Annotated[str, "New equation for the parametric model"],
) -> str:
    """Set the equation for an existing parametric model in the session."""
    session.set_model_equation(model_name=model_name, equation=equation)
    return f"Equation for model {model_name} set successfully."


@mcp.tool()
async def set_model_residual_equation(
    model_name: Annotated[str, "Name of the parametric model"],
    residual_equation: Annotated[
        str, "New residual equation for the parametric model"
    ],
) -> str:
    """Set the residual equation for an existing parametric model."""
    session.set_model_residual_equation(
        model_name=model_name, residual_equation=residual_equation
    )
    return f"Residual equation for model {model_name} set successfully."


@mcp.tool()
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
async def remove_model(
    model_name: Annotated[str, "Name of the model to remove"],
) -> str:
    """Remove a model from the refinement session."""
    session.remove_model(model_name)
    return f"Model {model_name} removed successfully."


@mcp.tool()
async def constrain_pdf_model_space_group_symmetry(
    model_name: Annotated[str, "Name of the parametric model"],
    space_group: Annotated[str, "Space group to constrain the model to"],
) -> str:
    """Constrain a parametric model to a specific space group symmetry."""
    session.constrain_pdf_model_space_group_symmetry(model_name, space_group)
    return (
        f"Model {model_name} constrained to space group {space_group} "
        f"successfully."
    )


@mcp.tool()
async def list_profiles() -> list[str]:
    """List all profiles in the refinement session."""
    return [str(profile_id) for profile_id in session.profiles_dict.keys()]


@mcp.tool()
async def list_models() -> list[str]:
    """List all models in the refinement session."""
    return [str(model_id) for model_id in session.models_dict.keys()]


@mcp.tool()
async def combine_models(
    parent_model_name: Annotated[str, "Name of the parent parametric model"],
    child_model_name: Annotated[str, "Name of the child parametric model"],
    symbol: Annotated[
        str, "Symbol to use for child model in the parent model's equation"
    ],
) -> str:
    """
    Combine two parametric models by registering the child to the parent model.
    """
    session.combine_models(parent_model_name, child_model_name, symbol)
    return (
        f"Models {parent_model_name} and "
        f"{child_model_name} combined successfully."
    )


@mcp.tool()
async def set_variable_value(
    variable_name: Annotated[str, "Name of the variable to set"],
    value: Annotated[float, "Value to set for the variable"],
) -> str:
    """
    Set the value of a specific variable in a parametric model.
    """
    session.set_variable_value(variable_name, value)

    return f"Variable '{variable_name}' is set to {value}."


@mcp.tool()
async def get_variable(
    variable_name: Annotated[str, "Name of the variable"],
) -> str:
    """
    Get the value of a specific variable in a parametric model.
    """
    variable = session.get_variable(variable_name)
    return f"Variable '{variable_name}': {variable['value']}"


@mcp.tool()
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
async def solve(
    profile_names: Annotated[
        list[str], "List of profile IDs to use in the refinement"
    ],
    model_names: Annotated[
        list[str], "List of model IDs to use in the refinement"
    ],
    variable_names: Annotated[list[str], "List of variable names to refine"],
    weights: Annotated[list[float], "List of weights for each profile"] = None,
    initial_values: Annotated[
        list[float], "List of initial values for each variable"
    ] = None,
) -> str:
    """
    Perform a refinement using the specified profiles, models, and variables.
    """
    out_string = session.solve(
        profile_names,
        model_names,
        variable_names,
        weights=weights,
        initial_values=initial_values,
    )

    return out_string


if __name__ == "__main__":
    mcp.run()

:tocdepth: -1

DiffPy Macro Interface
=======================

Diffpy macro files can be executed using the ``runmacro`` application.

.. code-block:: bash

    diffpy-app runmacro <macro_file>

To follow the example,

1. download the example files from :download:`here <../example/runmacro_example.zip>`

2. extract the downloaded files and navigate to the extracted directory

.. code-block:: bash

    mv /path/to/runmacro runmacro_example.zip working_directory
    cd working_directory
    unzip runmacro_example.zip

3. run the macro using the ``runmacro`` application

.. code-block:: bash

    diffpy-app runmacro example_macro.dp-in

How to write macro
~~~~~~~~~~~~~~~~~~

A macro file is a list of calls to the methods of the refinement session,
one call per line or spread over several lines. Arguments are written as
``name=value``. Supported values are numbers, quoted strings, ``True``,
``False``, ``None``, lists ``[...]``, and dictionaries ``{"key": value}``.
Everything after ``#`` is a comment. Calls are executed in order, and the
macro is rejected before running if any method name is unknown.

Let's still use the Ni example created earlier, but this time we will
write the macro file from scratch.

1. Load the profile and set the calculation range.

.. code-block:: text

    add_profile_from_file(
        profile_name="ni_profile", profile_path="path/to/profile.gr"
    )
    set_profile_calculation_range(
        profile_name="ni_profile", xmin=1.5, xmax=20, dx=0.01
    )
    update_profile_meta(profile_name="ni_profile", meta={"qmin": 0.1})

``ni_profile`` is the identifier used in the macro to refer to the profile.
The profile file is a ``.gr`` file containing the experimental data.
``xmin``, ``xmax`` and ``dx`` set the range and step size of the calculation,
and ``qmin`` in the profile meta sets the minimum Q used in the refinement.

2. Create the PDF model from the structure file.

.. code-block:: text

    add_pdf_model(
        model_name="pdf", structure_file_path="path/to/structure.cif", finite=False
    )
    constrain_pdf_model_space_group_symmetry(
        model_name="pdf", space_group="Fm-3m"
    )

The structure file is a ``.cif`` file representing the atomic arrangement in
your sample. The space group symmetry ``Fm-3m`` is preserved during the
refinement.

3. Set the initial values of the variables.

.. code-block:: text

    set_variables_value(
        name_value_dict={
            "pdf.phase.lattice.a": 3.52,
            "pdf.scale": 0.4,
            "pdf.phase.Ni0.Uiso": 0.005,
            "pdf.delta2": 2,
            "pdf.qdamp": 0.04,
            "pdf.qbroad": 0.02,
        }
    )

Variable names follow the format ``model_name.parameter``. Parameters of the
structure are under ``model_name.phase``, for example
``pdf.phase.Ni0.Uiso`` is the Uiso parameter of the atom ``Ni0``.

4. Run the refinement.

.. code-block:: text

    solve(
        profile_names=["ni_profile"],
        model_names=["pdf"],
        variable_names=[
            "pdf.phase.lattice.a",
            "pdf.scale",
            "pdf.phase.Ni0.Uiso",
            "pdf.delta2",
            "pdf.qdamp",
            "pdf.qbroad",
        ],
    )

Only the variables listed in ``variable_names`` are refined.

.. note::

    All commands from in a macro file are also tools inside
    ``cmi-agentic-ui`` MCP server as they are essentially wrappers around the
    same underlying methods. Please refer to :doc:`diffpy-agent`
    for more information.

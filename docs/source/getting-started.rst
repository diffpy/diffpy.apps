:tocdepth: -1

.. index:: getting-started

.. _getting-started:

================
Getting started
================

``diffpy.apps`` provides user applications to help with tasks using
diffpy packages. This page contains the instructions for all applications
available, including:

- :ref:`runmacro`

.. _runmacro:

Use ``runmacro`` to start refinement with a macro file
------------------------------------------------------

The ``runmacro`` application allows users to execute macros written in the
diffpy macro language.

.. code-block:: bash

    diffpy.app runmacro <macro_file.dp-in>

To follow the example,

1. download the example files from :download:`here <../example/runmacro_example.zip>`

2. extract the downloaded files and navigate to the extracted directory

.. code-block:: bash

    mv /path/to/runmacro runmacro_example.zip working_directory
    cd working_directory
    unzip runmacro_example.zip

3. run the macro using the ``runmacro`` application

.. code-block:: bash

    diffpy.app runmacro example_macro.dp-in

How to write macro
~~~~~~~~~~~~~~~~~~

Let's still use the Ni example created earlier, but this time we will
write the macro file from scratch.

1. Prepare the structure and profile file you want to work with.

.. note::

    The structure file is a ``.cif`` file representing the atomic arrangement in
    your sample, and the profile file is a ``.gr`` file containing the
    experimental data.

Start the macro file with the following two lines:

.. code-block:: text

    load structure G1 from "path/to/structure.cif"
    load profile exp_ni from "path/to/profile.gr"

``G1`` and ``exp_ni`` are the identifiers used in the macro to refer to the
structure and profile files, respectively. Quotes ``""`` are must to specify
the file paths.

2. (Optional) Declare the space group of the structure.

.. code-block:: text

    set G1 spacegroup as Fm-3m

``G1`` is the identifier for the structure file loaded earlier. Space group
symmetry ``Fm-3m`` is preserved during the refinement. You can also set
it to be ``auto`` to use the one automatically parsed from the structure file.
But if space group is not provided, nor can it be determined from the structure
file, it will be considered as ``P1`` space group.

3. (Optional) Set the calculation parameters for the refinement.

.. code-block:: text

    set exp_ni calculation_range as 0.5 20.0 0.01  # r_min, r_max, r_step
    set exp_ni q_range as 0.5 20.0  # q_min, q_max

``exp_ni`` is the identifier for the profile file loaded earlier.
``calculation_range`` specifies the range and step size for the calculation,
while ``q_range`` specifies the range of Q values to be used in the refinement.
If calculation parameters are not set, it will use the ones
that are defined in the profile file.


4. Create the refinement equation.

.. code-block:: text

    create equation variables s0
    set equation as "G1*s0"

``G1`` is the identifier for the structure file loaded earlier. In the
equation, it represents the PDF data generated from the structure `G1`.
``s0`` is created to count for the scaling factor.

5. Store the results.

.. code-block:: text

    save to "output_results.json"

The results of the refinement will be saved to a file
named ``output_results.json``.

6. List variables to be refined.

.. code-block:: text

    variables:
    ---
    - G1.a: 3.52
    - s0: 0.4
    - G1.Uiso_0: 0.005
    - G1.delta2: 2
    - qdamp: 0.04
    - qbroad: 0.02
    ---

Only variables listed in this section will be refined during the
execution of the macro, and the variables will also be refined in that order.
Variables with initial values specified here will be used as the
starting point for the refinement.

.. note::
    The naming of variables follows the format
    ``structure_identifier.parameter``,
    ``profile_parameter``, or
    ``equation_parameter``.

    For parameters belonging to a specific atom in the parameter,
    the naming follows the format ``structure_identifier.parameter_atomindex``.
    e.g. ``G1.Uiso_0`` here refers to the Uiso parameter of the first atom in
    the structure ``G1``.

    For constrained parameters, it will use the first parameter in the
    constraint. e.g. Here, lattice parameter ``a=b=c``,
    and ``Usio_0=Uiso_i, i=1,2,3``, ``a`` and ``Uiso_0`` are used as the
    reference variables.

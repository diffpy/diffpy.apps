=============
Release notes
=============

.. current developments

0.2.3
=====


0.2.3
=====

**Added:**

* Add diffpy-app agentify list and support -skill flag for both diffpy-app agentify and diffpy-app agentify list.
* Update the grammar of macro to be compatible with cmi-agentic-ui.
* ``clear`` accepts ``recipe_names``, ``profile_names`` and ``model_names`` to remove only the named items.
* Reuse constraints variables across recipe to avoid rebuilding recipe every time.


0.2.2
=====


0.2.1
=====


0.2.0
=====

**Added:**

* Add generator parallelism.
* Add soft bounds.
* add a tool to download cif file from database.
* support load PDF model from pyobjcryst code.
* support molecule PDF refinement.
* Added support for nanoparticle pdf refinement.
* Add more flexible interface to utilize ``diffpy.srfit``.

**Fixed:**

* Update the skill name and allow multiple skills.


0.1.1
=====

**Added:**

* Ensure ``git`` is installed and internet connection for ``agentify``.


0.1.0
=====

**Added:**

* Also add ``diffpy.apps`` as an entry point.
* Add ``runmacro`` in the ``README``.
* Support single line comment in the macro file.
* Add "agentify" app to deploy ``diffpy.cmi`` agent skills.
* Add ``diffpy.app runmacro <file>`` command.
* Add `DSL` support.

**Changed:**

* Use ``python=3.13`` as default before ``python=3.14`` is supported by ``diffpy.cmi``.

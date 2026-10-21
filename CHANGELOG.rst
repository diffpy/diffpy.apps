=============
Release notes
=============

.. current developments

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

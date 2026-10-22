#######
|title|
#######

.. |title| replace:: diffpy.apps documentation

``diffpy.apps`` - User applications to help with tasks using diffpy packages

| Software version |release|
| Last updated |today|.

===============
Getting started
===============

Welcome to the ``diffpy.apps`` documentation!

``diffpy.apps`` provides user a simpler interface to interact with diffpy
packages while retaining most of their flexibility. Currently it provides:

1. Macro interface for PDF refinements. See :doc:`diffpy-macro`.
2. Agent interface for PDF refinements. See :doc:`diffpy-agent`.


=======
Authors
=======

``diffpy.apps`` is developed by Yuchen Xiao, Simon Billinge and members of the Billinge group. This project is maintained by Simon Billinge. For a detailed list of contributors see
https://github.com/diffpy/diffpy.apps/graphs/contributors.

============
Installation
============

We recommend installing ``diffpy.apps`` using `conda <https://docs.conda.io/en/latest/>`_.

.. code-block:: bash

   conda create -n diffpy.apps_env python=3.13
   conda activate diffpy.apps_env
   conda install -c conda-forge diffpy.apps


================
Acknowledgements
================

``diffpy.apps`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.

=================
Table of contents
=================
.. toctree::
   :maxdepth: 2

   diffpy-agent
   diffpy-macro
   release
   license

=======
Indices
=======

* :ref:`genindex`
* :ref:`search`

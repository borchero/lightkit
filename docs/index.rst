LightKit
========

LightKit provides simple utilities for working with PyTorch and PyTorch Lightning. At the moment,
it provides three simple features:

- A data loader for tabular data that is orders of magnitude faster than PyTorch's builtin data
  loader for medium-sized datasets and beyond.
- A mixin for modules that allows to save not only weights, but also the configuration to the file
  system such that it is easier to retrieve trained models.
- A typed base class for estimators that enables users to easily create estimators with PyTorch and
  PyTorch Lightning which are fully compatible with Scikit-learn.

Installation
------------

LightKit is available via ``pip``:

.. code-block:: python

    pip install lightkit

If you are using `Poetry <https://python-poetry.org/>`_:

.. code-block:: python

    poetry add lightkit

Reference
---------

.. toctree::
   :maxdepth: 2

   sites/api

Index
^^^^^

- :ref:`genindex`

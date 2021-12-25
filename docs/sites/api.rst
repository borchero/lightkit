API Reference
=============

Estimator
---------

.. currentmodule:: lightkit

.. rubric:: Classes
.. autosummary::
    :toctree: generated/estimator
    :nosignatures:
    :caption: Estimator

    BaseEstimator
    ConfigurableBaseEstimator

.. currentmodule:: lightkit.estimator

.. rubric:: Mixins
.. autosummary::
    :toctree: generated/estimator
    :nosignatures:
    :template: classes/mixin.rst

    TransformerMixin
    PredictorMixin


Modules
-------

.. currentmodule:: lightkit.nn

.. rubric:: Classes
.. autosummary::
    :toctree: generated/nn
    :nosignatures:
    :caption: Modules
    :template: classes/mixin.rst

    Configurable


Data Handling
-------------

.. currentmodule:: lightkit.data

.. rubric:: Classes
.. autosummary::
    :toctree: generated/data
    :nosignatures:
    :caption: Data Handling
    :template: classes/no_method.rst

    DataLoader

    :template: autosummary/class.rst

    RangeBatchSampler

.. rubric:: Type Aliases
.. autosummary::
    :toctree: generated/data
    :nosignatures:
    :template: classes/type_alias.rst

    TensorLike
    DataLoaderLike


.. rubric:: Functions
.. autosummary::
    :toctree: generated/data
    :nosignatures:

    dataset_from_tensors
    collate_tuple
    collate_tensor

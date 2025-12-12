.. meta::
   :description: Create functional and beautiful websites for your documentation with Sphinx and the Awesome Sphinx Theme.
   :twitter:description: Create functional and beautiful websites for your documentation with Sphinx and the Awesome Sphinx Theme.


stable-datasets
===============

.. rst-class:: lead

   A comprehensive collection of stable, reproducible datasets for machine learning research.

Welcome to the docs for stable-datasets.
We recommend using ``python>=3.10``, and installation using ``uv``:

.. tab-set::

    .. tab-item:: uv

        .. code-block:: bash

            uv add stable-datasets

    .. tab-item:: pip

        .. code-block:: bash

            pip install stable-datasets


If you would like to start testing or contribute to ``stable-datasets`` then please install this project from source with:

.. code-block:: bash

    git clone https://github.com/rbalestr-lab/stable-datasets.git --single-branch
    cd stable-datasets
    pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   datasets/index
   introduction/quickstart
   introduction/showcase
   introduction/contributing

Citation
--------

If you find this library useful in your research, please consider citing us:

.. code-block:: bibtex

    @misc{stable-datasets,
      author = {},
      title = {},
      year = {2025},
      howpublished = {}
    }
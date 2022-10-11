
Benchmark Crowdsourcing
=====================
|Build Status| |Python 3.6+|



Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tanglef/benchmark_crowdsourcing
   $ benchopt run benchmark_crowdsourcing

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_crowdsourcing -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tanglef/benchmark_crowdsourcing/workflows/Tests/badge.svg
   :target: https://github.com/tanglef/benchmark_crowdsourcing/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/

.. BlueFridgeNonLin documentation master file, created by
   sphinx-quickstart on Sun Mar 23 21:06:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BlueFridgeNonLin documentation
==============================


This documentation is for the Blue Fridge non-Hermitian 4+5 qubit setup, built while we're still using GPIB and the WX waveform generator. It serves as a snapshot of the control and analysis infrastructure as we transition toward a more modular and object-oriented codebase.

We're using the Sphinx documentation tool to build and maintain this documentation.

This project represents our first step toward **modular, maintainable, and scalable** code for multi-qubit experiments. We're introducing `Qubit` and `Gate` objects to ensure the codebase is:

- Scalable across different experimental platforms
- Consistent across experiment types
- Easier to maintain and debug

Ultimately, this modularization will support faster iteration and clearer abstractions in complex quantum experiments.



Add more content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


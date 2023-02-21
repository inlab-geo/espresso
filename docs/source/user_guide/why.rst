=================
What is Espresso?
=================

Espresso was conceived by `Andrew Valentine <https://valentineap.github.io/>`_ & `Malcolm Sambridge <http://rses.anu.edu.au/~malcolm/>`_. As researchers working on new inversion and inference algorithms, they often needed example problems that could be used to test and illustrate their ideas. However, finding good examples is laborious: you must identify an appropriate problem, obtain relevant datasets and simulation codes, and learn how these work --- and then the example may turn out not to perform as hoped. As a result, individual researchers tend to be familiar with a small number of problems that they use and re-use, and progress often ends up siloed within specific research domains. 

Espresso aims to address this issue, by making it easy for researchers to access and explore a varied range of exemplars. Espresso is a Python package that provides access to a suite of example problems. In this context, an 'example problem' means:

- A (real or synthetic) data set, arranged in a single vector :math:`\mathbf{d_0}`; and
- A simulation package, or 'forward model', which takes in a model vector, :math:`\mathbf{m}`, and computes a simulated version of the data vector :math:`\mathbf{g}(\mathbf{m})`;

Many problems also provide additional capabilities and information, such as the ability to compute gradients (:math:`\nabla_\mathbf{m}\, \mathbf{g}`), suggestions for problem-appropriate objective functions and prior distributions, and visualisation routines.


Espresso can be applied for testing, validation and benchmarking of algorithms across a wide range of settings where one aims to estimate or otherwise characterise :math:`\mathbf{m^\prime}` such that :math:`\mathbf{d_0}\approx\mathbf{g}(\mathbf{m^\prime})`, including:

- Optimisation;
- Inversion;
- Bayesian inference;
- Machine learning.

The package primarily focusses on problems where :math:`\mathbf{g}(\mathbf{m})` is a model for some system or process relevant to the earth sciences.

The key design goal for Espresso is 'uniformity of interface': switching from one problem to another should require changing only one line of code, and examples should be accessible to users without knowledge of the domain or specific implementation. We rely on domain-expert contributors to make decisions about data selection and processing, parameter settings, and so on. Most Espresso examples will  be cut-down, restricted versions of real-world inference problems, and our interface requirements are likely to be limiting. We regard this as a feature, not a bug. Espresso provides a quick, easy route to explore new domains and identify problems that might repay more focussed attention.
# FMM Tomography

<!-- Please write anything you'd like to explain about the forward problem here -->

The wave front tracker routines solves boundary value ray tracing problems into 2D heterogeneous wavespeed media, defined by continuously varying velocity model calculated by 2D cubic B-splines.

## Fast Marching wavefront tracking
 
The method used here is *Fast Marching wavefront tracking*. In this algorithm the entire first arriving wavefronts from each source are calculated across the entire model in the form of a *travel time field* one per source. Travel times and ray paths are then calculated by tracing rays back from a desired receiver position to the source along the perpendicular to the wavefronts. This uses the *Fast Marching method* which is stable in highly heterogeneous velocity models. Here we implement a version of Fast Marching with source grid refinement for improved accuracy, as described in the papers below. 

*Rawlinson, N., de Kool, M. and Sambridge, M., 2006. "Seismic wavefront tracking in 3-D heterogeneous media: applications with multiple data classes", Explor. Geophys., 37, 322-330.*

*de Kool, M., Rawlinson, N. and Sambridge, M. 2006. "A practical grid based method for tracking multiple refraction and reflection phases in 3D heterogeneous media", Geophys. J. Int., 167, 253-270.*

The python implementation here is a *wrapper* around the FMM code `fm2dss.f90` of Nick Rawlinson.


## Theoretical background

The goal in travel-time tomography is to infer details about the velocity structure of a medium, given measurements of the minimum time taken for a wave to propagate from source to receiver. For seismic travel times, as we change our model, the route of the fastest path from source to receiver also changes. This makes the problem nonlinear, as raypaths also depend on the sought after velocity or slowness model. 

Provided the 'true' velocity structure is not *too* dissimilar from our initial guess, travel-time tomography can be treated as a weakly non-linear problem. In this notebook we optionally treat the ray paths as fixed, and so it becomes a linear problem, or calculate rays in the velociuty model.

The travel-time of an individual ray can be computed as 

$$t = \int_\mathrm{path} \frac{1}{v(\mathbf{x})}\,\mathrm{d}\mathbf{x}$$

This points to an additional complication: even for a fixed path, the relationship between velocities and observations is not linear. However, if we define the 'slowness' to be the inverse of velocity, $s(\mathbf{x}) = v^{-1}(\mathbf{x})$, we can write
$$t = \int_\mathrm{path} {s(\mathbf{x})}\,\mathrm{d}\mathbf{x}$$
which *is* linear.


We will assume that the object we are interested in is 2-dimensional slowness field. If we discretize this model, with $N_x$ cells in the $x$-direction and $N_y$ cells in the $y$-direction, we can express $s(\mathbf{x})$ as an $N_x \times N_y$ vector $\boldsymbol{s}$. 

**For the linear case**, this is related to the data by

$$d_i = A_{ij}s_j $$

where $d_i$ is the travel time of the ith path, and where $A_{ij}$ represents the path length in cell $j$ of the discretized model.

**For the nonlinear case**, this is related to the data by

$$\delta d_i = A_{ij}\delta s_j $$

where $\delta d_i$ is the difference in travel time, of the ith path, between the observed time and the travel time in the reference model. Here $A_{ij}$ represents the path length in cell $j$ of the discretized model. The parameters $\delta s_j$ are slowness perturbations to the reference model.


## Library usage

All the methods comply to `cofi-espresso` standards. Additionally, you can display 
paths together with the model when running `fmm_instance.plot_model(model, paths=True)`,
where `fmm_instance` is an instance of class `FmmTomography`.

<!-- ## Getting started

To complete this contribution, here are some ideas on what to do next:

- [ ] **Modify [README.md](README.md)**. Document anything you'd like to add for this problem
  (in this README.md file). Some recommended parts include:
   - What this test problem is about
   - What you would recommend inversion practitioners to notice
   - etc.
- [ ] **Modify [LICENCE](LICENCE)**. The default one we've used is a 2-clauss BSD licence. 
   Feel free to replace the content with a licence that suits you best.
- [ ] **Write code in [fmm_tomography.py](fmm_tomography.py) (and [\_\_init\_\_.py](__init__.py) if
   necessary)**. Some basic functions have been defined in the template - these are the
   standard interface we'd like to enforce in Espresso. You'll see
   clearly some functionalities that are required to implement and others that are
   optional.
   - If you would like to load data from files, please use our 
     [utility functions](https://cofi-espresso.readthedocs.io/en/latest/user_guide/api/generated/cofi_espresso.utils.html) 
     to get absoluate path before calling your load function.
- [ ] **Validate and build your contribution locally**. We have seperate scripts for 
   validation and packaging. Check 
   [how to test building your contribution](README.md#how-to-test-building-your-contribution-with-cofi-espresso) 
   for details.
- [ ] **Delete / comment out these initial instructions**. They are for your own reference
   so feel free to delete them or comment them out once you've finished the above
   checklist. -->


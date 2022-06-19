"""Eletrical Resistivity Tomography Inversion with PyGIMLi + CoFI

This script runs:
- ERT problem defined with PyGIMLi, and
- Newton's optimisation method with CoFI

"""

############# 0. Import modules #######################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygimli
from pygimli import meshtools
from pygimli.physics import ert

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "pygimli_ert"
_solver_name = "newton_opt"
_file_prefix = f"{_problem_name}_{_solver_name}"


def scheme_fwd():                   # Dipole Dipole (dd) measuring scheme
    scheme = ert.createData(elecs=np.linspace(start=0, stop=50, num=51),schemeName='dd')
    return scheme

def geometry_true():                # piecewise linear complex
    world = meshtools.createWOrld(start=[-55, 0], end=[105, -80], worldMarker=True)
    conductive_anomaly = meshtools.createCircle(pos=[10, -7], radius=5, marker=2)
    plc = meshtools.mergePLC((world, conductive_anomaly))    
    return plc

def mesh_fwd(scheme, plc):          # forward mesh
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        plc.createNode(s + [0.0, -0.2])
    mesh_coarse = meshtools.createMesh(plc, quality=33)
    fmesh = mesh_coarse.createH2()
    return fmesh

def markers_to_resistivity():       # set resistivity values in each region
    rhomap = [[1, 200],
              [2,  50],]
    return rhomap

def model_vec(rhomap, fmesh):       # create true model vector
    model_true = pygimli.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)
    return model_true


def _post_plot(ax, title):
    ax[0].set_title(title)
    if save_plot:
        plt.savefig(ax, f"{_figs_prefix}_{title}")

def plot_mesh(mesh, title):
    ax = pygimli.show(mesh)
    _post_plot(ax, title)
    
def plot_model(mesh, model, label, title):
    ax = pygimli.show(mesh, data=model, label=label)
    _post_plot(ax, title)

def plot_data(survey, label, title):
    ax=ert.showERTData(survey,label=label)
    _post_plot(ax, title)

def mesh_inv_triangular(scheme):
    world = meshtools.createWorld(start=[-15, 0], end=[65, -30], worldMarker=False, marker=2)
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        world.createNode(s + [0.0, -0.4])
    mesh_coarse = mt.createMesh(world, quality=33)
    imesh = mesh_coarse.createH2()
    for nr, c in enumerate(imesh.cells()):
        c.setMarker(nr)
    return imesh


def main(output_dir="."):
    _figs_prefix = f"{output_dir}/{_file_prefix}"
    
    ######### 1. Define the problem ###################################################
    
    # PyGIMLi - define measuring scheme, geometry, forward mesh and true model
    scheme = scheme_fwd()
    geometry = geometry_true()
    fmesh = mesh_fwd(scheme, geometry)
    rhomap = markers_to_resistivity()
    model_true = model_vec(rhomap, fmesh)
    
    # PyGIMLi - plot the compuational mesh and the true model
    plot_mesh(fmesh, "Computational Mesh")
    plot_model(fmesh, model_true, "$\Omega m$", "Resistivity")

    # PyGIMLi - generate data
    survey = ert.simulate(fmesh, res=rhomap, scheme=scheme)
    plot_data(survey, "$\Omegam$", "Aparent Resistivity")
    y_obs = np.log(survey['rhoa'].array())

    # PyGIMLi - create mesh for inversion
    imesh_tri = mesh_inv_triangular(scheme)
    plot_mesh(imesh_tri, "Inversion Mesh")
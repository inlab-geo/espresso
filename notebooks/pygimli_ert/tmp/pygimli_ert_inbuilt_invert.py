import matplotlib.pyplot as plt
import numpy as np

import pygimli
from pygimli import meshtools
from pygimli.physics import ert


############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = ert.createData(elecs=np.linspace(start=0, stop=50, num=51), schemeName="dd")

# simulation mesh
world = meshtools.createWorld(start=[-55,0], end=[105,-80], worldMarker=True)
conductive_anomaly = meshtools.createCircle(pos=[10,-7], radius=5, marker=2)
geom = world + conductive_anomaly
ax = pygimli.show(geom)
ax[0].figure.savefig("figs/true_geometry")
for s in scheme.sensors():          # local refinement 
    geom.createNode(s + [0.0, -0.2])
rhomap = [[1, 200], [2,  50],]
mesh = meshtools.createMesh(geom, quality=33)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].figure.savefig("figs/true_model_coarse")
# mesh = mesh.createH2()
# ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
# ax[0].figure.savefig("figs/true_model")

# generate data
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, seed=42)
data.remove(data['rhoa'] < 0)
log_data = np.log(data['rhoa'].array())
ax = ert.show(data)
ax[0].figure.savefig("figs/data")

############# Inverted by PyGIMLi solvers #############################################

mgr = ert.ERTManager(data, verbose=True, useBert=True)
inv = mgr.invert(lam=20, verbose=True)
fig = mgr.showResultAndFit()
fig.savefig("figs/invert_result")

print(inv)
print(type(inv))

# # plot inferred model
# ax = pygimli.show(mesh, data=inv, label=r"$\Omega m$")
# ax[0].set_title("Inferred model")
# ax[0].figure.savefig("figs/pygimli_ert_inbuild_inferred")

# # plot synthetic data
# data = ert.simulate(mesh, scheme=scheme, res=inv)
# data.remove(data['rhoa'] < 0)
# log_data = np.log(data['rhoa'].array())
# ax = ert.show(data)
# ax[0].figure.savefig("figs/data_synth_inbuild_inferred")

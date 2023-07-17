import numpy as np
import pygimli
from pygimli.physics import ert
from pygimli import meshtools


# Dipole Dipole (dd) measuring scheme
scheme = ert.createData(elecs=np.linspace(start=0,stop=50,num=51), schemeName="dd")

# true geometry, forward mesh and true model
world = meshtools.createWorld(start=[-55,0], end=[105,-80], worldMarker=True)
for s in scheme.sensors():          # local refinement 
    world.createNode(s + [0.0, -0.1])
conductive_anomaly = meshtools.createCircle(pos=[10,-7], radius=5, marker=2)
geom = world + conductive_anomaly
rhomap = [[1, 200], [2,  50],]
mesh = meshtools.createMesh(geom, quality=33)

# generate synthetic data
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1, noise_abs=1e-6, seed=42)
data.remove(data["rhoa"] < 0)
log_data = np.log(data["rhoa"].array())
data_err = data["rhoa"] * data["err"]
data_err_log = np.log(data_err)
data_cov_inv = np.eye(log_data.shape[0]) / (data_err_log ** 2)

data.save("3_ert_data.dat")
np.savetxt("3_ert_data_log.txt", log_data)
np.savetxt("3_ert_data_cov_inv.txt", data_cov_inv)

"""Linear XRay Tomography Solved with a Linear System Solver with CoFI

"""
from pathlib import Path
from cofi import BaseProblem, InversionOptions, Inversion

from xray_tomography_lib import *

############# Initialisation for XRT problem ##########################################

# load data
current_dir = Path(__file__).resolve().parent
sourcedat=np.loadtxt(current_dir / 'datasets/ttimes/sources_InLab_lrt_s500.dat')
recdat = np.loadtxt(current_dir / 'datasets/ttimes/receivers_InLab_lrt_r36.dat')
ttdat = np.loadtxt(current_dir / 'datasets/ttimes/ttimes_InLab_lrt_s500_r36.dat')
recs = recdat.T[1:].T # set up receivers
srcs = sourcedat.T[1:].T # set up sources
nr,ns = np.shape(recs)[0],np.shape(srcs)[0] # number of receivers and sources
print(' New data set has:\n',np.shape(recs)[0],
      ' receivers\n',np.shape(sourcedat)[0],
      ' sources\n',np.shape(ttdat)[0],' travel times')

rays = (ttdat[:,1] + ttdat[:,0]*nr).astype(int) # find rays from travel time file
d = ttdat[rays,2] # set up data for linear tomography

# build ray path matrix
# evaluate travel times for all straight ray paths in 50x50 slowness model
mref = np.ones([50,50])             # build homogeneous slowness model to calculate raypaths in
paths = buildPaths(srcs,recs)   # calculate straight paths between all source receiver pairs 
ttimes, A = tracer(mref,paths)  # calculate raypath matrix
A = A[rays]                         # take only rays (rows) of matrix corresponding to data
print(' Seismic travel times min ',np.min(d),' max ',np.max(d),' mean ',np.mean(d))
ndata = np.shape(A)[0]


############# Define problem with CoFI ################################################
xrt_problem = BaseProblem()
# TODO continue this
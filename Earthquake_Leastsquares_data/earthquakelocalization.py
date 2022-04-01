import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from Earthquake import plotcovellipse as pc
import math
import scipy.io as sio
from Earthquake import eqlocate as eq
from Earthquake.plotcovellipse import plot_point_cov,plot_cov_ellipse
import pickle

######################################################

#start=[-10.0, 5.0, 5.0, 0.0]



def test_init():
    pickle_off = open("Earthquake/eqdata.pickle","rb")
    [data,datan,rec_loc] = pickle.load(pickle_off)
    vel = 5.4
    nit = 8 # number of iterations
    n_used = 10
    model=[-10.0, 5.0, 5.0, 0.0]


class init_routine():
    pickle_off = open("Earthquake/eqdata.pickle","rb")
    [data,datan,rec_loc] = pickle.load(pickle_off)
    vel = 5.4
    nit = 8 # number of iterations
    n_used = 10
    model=[-10.0, 5.0, 5.0, 0.0]
    print('Observed travel times (eql.data):\n',data)
    print('Observed travel times with noise (eql.datan):\n',datan)

def forward(init_obj):
    rec_loc=init_obj.rec_loc
    model=init_obj.model
    vel=init_obj.vel
    d = np.zeros(len(rec_loc))     # define distance matrix shape
    for i in range(len(rec_loc)):
        dx = rec_loc[i,0]-model[0]
        dy = rec_loc[i,1]-model[1]
        dz = rec_loc[i,2]-model[2]
        d[i] = np.sqrt(dx*dx+dy*dy+dz*dz)
    synthetic = model[3] + d/vel
    gradient=[]
    return synthetic, gradient

def plot_model(init_obj):
    rec_loc=init_obj.rec_loc
    fig, ax = plt.subplots(figsize=(9,6))
    plt.title('Station locations')
    ax.scatter(rec_loc.T[0],rec_loc.T[1],color='k')
    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def calcG(sol,model):
    vel = 5.4
    d = np.zeros(len(rec_loc))     # define distance matrix shape
    G = np.zeros((len(rec_loc),4)) # define G matrix shape
    for i in range(len(rec_loc)):
        dx = rec_loc[i,0]-sol[0]
        dy = rec_loc[i,1]-sol[1]
        dz = rec_loc[i,2]-sol[2]
        d[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
        G[i,0] = -dx/(d[i]*vel)
        G[i,1] = -dy/(d[i]*vel)
        G[i,2] = -dz/(d[i]*vel)
        G[i,3] = 1
    return G


def myFancySolverRoutine(model, data, synthetics):
    xsol = np.zeros([nit,4])
    xsol[0,:] = start
    n_used =10
    t = t[:n_used]
    Cdinv = np.eye(len(t))/(0.2*0.2)
    r = []
#    for it in range(nit): # loop over number of iterations
    r = synthetics-data
    #print(r)
    chisq = np.dot(r.T,r)
    print(chisq)
    r2 = synthetics-datan
    #print(r)
    chisq2 = np.dot(r2.T,r2)
    print(chisq2)
    Cdinv = np.eye(len(data))/(0.2*0.2)
    G = calcG(start,model)
    print('CD inverse \n',Cdinv)
    print('G matrix at initial guess \n',G)

#class forward:
#    def __init__(self, model, start):
#        self.model = model
#        self.start = start
#    
#    vel = 5.4
#    d = np.zeros(len(model.sta))     # define distance matrix shape
#    for i in range(len(model.sta)):
#        dx = model.sta[i,0]-start[0]
#        dy = model.sta[i,1]-start[1]
#        dz = model.sta[i,2]-start[2]
#        d[i] = np.sqrt(dx*dx+dy*dy+dz*dz)
#    tpred = start[3] + d/vel
#    gradient=[]

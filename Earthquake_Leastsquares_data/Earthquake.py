import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from Earthquake import plotcovellipse as pc
import math
import scipy.io as sio
from Earthquake import eqlocate as eq
from Earthquake.plotcovellipse import plot_point_cov,plot_cov_ellipse
import pickle

class Earthquake():
    pickle_off = open("Earthquake/eqdata.pickle","rb")
    [data,datan,rec_loc] = pickle.load(pickle_off)
    vel = 5.4
    nit = 8 # number of iterations
    n_used = 10
    # model=[-10.0, 5.0, 5.0, 0.0]
    
    
def init_routine(eql_basics):
    try:
        model=eql_basics.model
    except:
        model=[-10.0, 5.0, 5.0, 0.0] # create starting model, if not set in eql_basics
    return model

def forward(eql_basics, model):
    rec_loc=eql_basics.rec_loc
    vel=eql_basics.vel
    d = np.zeros(len(rec_loc))     # define distance matrix shape
    for i in range(len(rec_loc)):
        dx = rec_loc[i,0]-model[0]
        dy = rec_loc[i,1]-model[1]
        dz = rec_loc[i,2]-model[2]
        d[i] = np.sqrt(dx*dx+dy*dy+dz*dz)
    synthetic = model[3] + d/vel
    gradient=[]
    return synthetic, gradient

def solver(eql_basics, model_start, synthetic, gradient):
    rec_loc=eql_basics.rec_loc
    nit=eql_basics.nit
    n_used=eql_basics.n_used
    try :
        noise=eql_basics.noise
    except:
        noise=0
        
    data=eql_basics.data + np.random.normal(0,noise*np.max(eql_basics.data),len(eql_basics.data))
    
    model_recovered = np.zeros([nit,4])
    model_recovered[0,:] = model_start
    t = data[:n_used] # 
    Cdinv = np.eye(len(t))/(0.2*0.2)
    r = []
    for it in range(nit): # loop over number of iterations
    # calculate travel times of current guess
        tpred = calct( model_recovered[it],rec_loc[:n_used]) # calculate predicted travel times from current location
        G = calcG(model_recovered[it],rec_loc[:n_used])      # calculate new G matrix for current location
        residuals = t-tpred       # calculate residuals between observed and predicted travel times
        chisq =np.dot(residuals.T,np.dot(Cdinv,residuals))
        print(' Iteration ',it,'Chisq ',chisq,' Current location','{:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format(*tuple(model_recovered[it])))
        r = np.append(r,residuals)
        
        A = np.dot(np.transpose(G), Cdinv)
        GtG = np.dot(A, G) # G^T C_inv G^T
        GtGinv = np.linalg.inv(GtG) # Inverse of G^TC_inv G
        B = np.dot(GtGinv,A) # Least squares operator 
        dm = np.dot(B,residuals) # find update to location solution  
        if(it!=nit-1):model_recovered[it+1] = model_recovered[it] + dm
    r = r.reshape(nit,len(rec_loc[:n_used]))
    
    result=resultmaker(model_recovered,r, chisq)
    return result

def plot_model(eql_basics, result):
    nit=eql_basics.nit  
    n_used=eql_basics.n_used
    rec_loc=eql_basics.rec_loc
    r=result.r
    xsol=result.model_recovered
    
    print('\nStation arrival time residuals for each iteration\n')
    print('         ',('{:7d} '*nit).format(*range(nit)))
    for i in range(n_used):
        print('Station ',i,('{:7.4f} '*nit).format(*tuple(r.T[i])))

    fig, ax = plt.subplots(figsize=(12,9))
    plt.title('')
    ax.scatter([10],[0] ,500, color="r",  marker=(5, 2),label='Earthquake location')
    ax.scatter(rec_loc.T[0],rec_loc.T[1],marker='^', color='k',label='Stations')
    #ax.scatter(rec_loc.T[0],rec_loc.T[1],10, np.linspace(1,10,len(eql_basics.rec_loc)))
    plt.plot(xsol.T[0][:nit],xsol.T[1][:nit],':') 
    ax.scatter(xsol.T[0][:nit],xsol.T[1][:nit], 50, np.linspace(1,10,len(xsol.T[0][:nit])),cmap='rainbow')
    # Put large marker on last location:
    # ax.scatter(xsol.T[0][-1],xsol.T[1][-1] ,500, color="r",  marker=(5, 2))
    #plt.plot(xsol.T[0][-1],xsol.T[1][-1],'ro')
    #plt.plot(xsol.T[0][0],xsol.T[1][0],'go')
    for i in range(len(rec_loc)):
        plt.text(rec_loc.T[0][i]+1,rec_loc.T[1][i],str(i))
    plt.text(xsol.T[0][0]-3,xsol.T[1][0],'start')
    plt.text(xsol.T[0][-1]-1,xsol.T[1][-1]+2,'end')
    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper left")
    plt.show()    


# Secondary functions etc..

def calcG(sol,sta):
    vel = 5.4
    d = np.zeros(len(sta))     # define distance matrix shape
    G = np.zeros((len(sta),4)) # define G matrix shape
    for i in range(len(sta)):
        dx = sta[i,0]-sol[0]
        dy = sta[i,1]-sol[1]
        dz = sta[i,2]-sol[2]
        d[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
        G[i,0] = -dx/(d[i]*vel)
        G[i,1] = -dy/(d[i]*vel)
        G[i,2] = -dz/(d[i]*vel)
        G[i,3] = 1
    return G

def calct(sol,rec_loc):
    vel = 5.4
    d = np.zeros(len(rec_loc))     # define distance matrix shape
    for i in range(len(rec_loc)):
        dx = rec_loc[i,0]-sol[0]
        dy = rec_loc[i,1]-sol[1]
        dz = rec_loc[i,2]-sol[2]
        d[i] = np.sqrt(dx*dx+dy*dy+dz*dz)
    tpred = sol[3] + d/vel
    return tpred

class resultmaker():
    def __init__(self, model_recovered,r, chisq):
        self.model_recovered=model_recovered
        self.r=r
        self.chisq=chisq


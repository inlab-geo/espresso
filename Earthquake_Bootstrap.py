import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import pickle
from Earthquake_Bootstrap import plotcovellipse as pc
from Earthquake_Bootstrap import eqlocate as eq





class Earthquake():
    pickle_eq = open("Earthquake_Bootstrap/loctim.pickle","rb")
    [la,lo,el,ts,vp] = pickle.load(pickle_eq)
    # load border.xy
    pickle_b = open("Earthquake_Bootstrap/border.pickle","rb")
    [borderx,bordery] = pickle.load(pickle_b)
    
    nBoot = 5000 # Number of bootstrap samples

    # vel = 5.4
    # nit = 8 # number of iterations
    # n_used = 10
    # model=[-10.0, 5.0, 5.0, 0.0]




def forward(eql_basics, model_start):
    gradient=[]
    x0=model_start[0]
    y0=model_start[1]
    z0=model_start[2]
    tol=[0.1,0.1]
    vp=eql_basics.vp
    la=eql_basics.la
    lo=eql_basics.lo
    el=eql_basics.el
    ts=eql_basics.ts
    sols, res =eq.eqlocate(x0,y0,z0,ts,la,lo,el,vp,tol) # here sols are the iterative solutions found,
    synthetic=eql_basics.ts-res
    return synthetic, gradient
    
    
def init_routine(eql_basics):
    try:
        model=eql_basics.model
    except:
        model=[9, 46.8, -10] # create starting model, if not set in eql_basics
    return model



def solver(eql_basics, model_start, synthetic, gradient):
    x0=model_start[0]
    y0=model_start[1]
    z0=model_start[2]
    tol=[0.1,0.1]
    vp=eql_basics.vp
    la=eql_basics.la
    lo=eql_basics.lo
    el=eql_basics.el
    ts=eql_basics.ts
    nBoot=eql_basics.nBoot
    sols, res =eq.eqlocate(x0,y0,z0,ts,la,lo,el,vp,tol) # here sols are the iterative solutions found
    bootstrap_solutions = np.zeros((nBoot,4))
    for i in range(nBoot):
        yBoot = synthetic + np.random.choice(res,size=len(res),replace=True) # random sample residuals with replacement
        solsB, resB =eq.eqlocate(x0,y0,z0,yBoot,la,lo,el,vp,tol) # here sols are the iterative solutions found,
        bootstrap_solutions[i] = solsB[-1] # bootstrap solution
    
    bootstrap_cov=np.cov(bootstrap_solutions.T)
    tpred=ts-res
    model_all=sols
    orig_t_all=sols[:,0]
    orig_t_final=sols[-1,0] # origin time
    model_final=[sols[-1,1], sols[-1,2], sols[-1,3]] # x,y,z,
    result=resultmaker(bootstrap_solutions, bootstrap_cov, res, model_all, model_final, tpred, orig_t_final, orig_t_all)
    print ('Earthquake location (iterative least square solution):\n', model_final)
    print ('Event time (seconds after 16:30)',orig_t_final)

    return result

def plot_model(eql_basics, result):
    solBoot=result.bootstrap_solutions
    orig_t_final=result.orig_t_final
    orig_t_all=result.orig_t_all
    model_final=result.model_final
    borderx=eql_basics.borderx
    bordery=eql_basics.bordery
    la=eql_basics.la
    lo=eql_basics.lo
    sols=result.model_all
    xfinal=result.model_final[0]
    yfinal=result.model_final[1]
    cov=result.bootstrap_cov

    plt.figure(figsize=(9,6))
    plt.plot(borderx,bordery,'r-')
    plt.scatter(lo,la,s=50,marker='^',label="Stations")
    plt.plot(sols[:,1],sols[:,2],'o-y') # solution updates
    plt.plot(sols[0,1],sols[0,2],'ok',label="Initial guess") # initial guess
    plt.plot(xfinal,yfinal,'or',label="Final guess")
    plt.xlim([5.5,11])
    plt.ylim([45.5,48])
    plt.legend(loc="upper left")
    plt.title("Earthquake location", fontsize=20)
    plt.show()


    fig = plt.figure(figsize=(9,6))

    ax1 = plt.subplot(221)
    xp, yp = solBoot.T[0], solBoot.T[1]
    ax1.plot(xp, yp, 'k+')
    ax1.plot(orig_t_final,model_final[0], 'ro',label="Least squares solution")
    ax1.set_xlabel('Origin Time')
    ax1.set_ylabel('Longitude')
    plt.legend(loc="upper left")

    ax2 = plt.subplot(222)
    xp, yp =  solBoot.T[1], solBoot.T[2]
    ax2.plot(xp, yp, 'k+')
    ax2.plot(model_final[0],model_final[1], 'ro')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    ax3 = plt.subplot(223)
    xp, yp =  solBoot.T[0], solBoot.T[2]
    ax3.plot(xp, yp, 'k+')
    ax3.plot(orig_t_final,model_final[1], 'ro')
    ax3.set_xlabel('Origin Time')
    ax3.set_ylabel('Latitude')
    #plt.tight_layout()
    fig.suptitle("Bootstrap solutions", fontsize=20)
    plt.show()
    
    sol=[result.orig_t_final, result.model_final[0], result.model_final[1],result.model_final[2]]
    # Bootstrap mean
    print(' Bootstrap mean earthquake location:\n',np.mean(solBoot[1:4],axis=0))
    print(' Bootstrap mean event time (seconds after 16:30):\n',np.mean(solBoot[0],axis=0))
    print(' Bootstrap covariance:\n',cov)
    bcsol = sol - (np.mean(solBoot,axis=0)-sol)
    print(' Bootstrap bias corrected solution:\n',bcsol)
    p = np.percentile(solBoot,[2.5,97.5],axis=0)
    print(' Bootstrap 95% Confidence intervals: ')
    print(" Parameter 1 {:7.3f} [{:7.3f}, {:7.3f}]".format(bcsol[0],p[0,0],p[1,0]))
    print(" Parameter 2 {:7.3f} [{:7.3f}, {:7.3f}]".format(bcsol[1],p[0,1],p[1,1]))
    print(" Parameter 3 {:7.3f} [{:7.3f}, {:7.3f}]".format(bcsol[2],p[0,2],p[1,2]))


class resultmaker():
    def __init__(self, bootstrap_solutions, bootstrap_cov, res, model_all, model_final, tpred, orig_t_final, orig_t_all):
        self.bootstrap_solutions=bootstrap_solutions
        self.bootstrap_cov=bootstrap_cov
        self.res=res
        self.model_all=model_all
        self.model_final=model_final
        self.tpred=tpred
        self.orig_t_final=orig_t_final
        self.orig_t_all=orig_t_all


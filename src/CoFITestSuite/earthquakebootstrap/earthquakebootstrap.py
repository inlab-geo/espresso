import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import pickle
from cofitestsuite.earthquakebootstrap import plotcovellipse as pc
from cofitestsuite.earthquakebootstrap import eqlocate as eq
import os




class basics():
    """
    Creates a class object containing basic information about the inversion test problem.

    Each attribute in this class can be set prior to the forward calculation to change the output.

    Attributes:
    --------------------

    Changeable:
    :param nBoot: Set the number of bootstrap iterations, i.e. for how many data subsets the inversion is performed.
    :type nBoot: int
    :param vel: Set the wave speed of the Earth crust. Use assumes homogeneous crustal Earth model
        with constant wave speed, in km/s. Values can be changed for each recording location.
    :type vp: numpy array
    --------------------

    Others:
    :param la: Latitude of receiver locations, in degree (WGS84?)
    :type la: float
    :param lo: Longitude of receiver locations, in degree (WGS84?)
    :type lo: float
    :param el: Elevation of receiver locations, im m
    :type el: float
    :param ts: Recording time, in seconds after 16:30
    :type ts: float
    :param borderx: Latitude of border of Switzerland, in degree (WGS84?)
    :type borderx: float
    :param bordery: Longitude of border of Switzerland, in degree (WGS84?)
    :type bordery: float


    --------------------
    """
    def data_path(filename):
        path_to_current_file = os.path.realpath(__file__)
        current_directory = os.path.split(path_to_current_file)[0]
        data_path = os.path.join(current_directory, filename)
        return data_path


    nBoot = 5000 # Number of bootstrap samples

    pickle_eq = open(data_path("loctim.pickle"),"rb")
    [la,lo,el,ts,vp] = pickle.load(pickle_eq)
    del pickle_eq

    pickle_b = open(data_path("border.pickle"),"rb")
    [borderx,bordery] = pickle.load(pickle_b)
    del pickle_b


def forward(eql_basics, model_start):
    """
    Calculates the gravitational force of each recording location based on the input model.

    Arguments:
    -------------

    :param eql_basics: Basic parameters of the inversion test problem
    :type eql_basics: class
    :param model: Contains starting coordinates for the Earthquake location.
    :type model: numpy array
    :param synthetics: Contains synthetic data created with the forward calulation
    :type synthetics: class
        - :param sols: Contains the iterative solutions of the inversion (coordinates)
          :type sols: numpy array
        - :param res: The observed arrival time - minus the predicted arrival time
          :type res: numpy array
        - :param tpred: Contains the predicted arrival times for comparison with origin time
          :type tpred: numpy array
    :param gradient: Empty variable in this inversion test problem.
    :type gradient: list (empty)
    -------------
    """

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
    sols, res =eq.eqlocate(x0,y0,z0,ts,la,lo,el,vp,tol) # here sols are the iterative solutions found
    tpred=eql_basics.ts-res
    synthetic=synth(sols, res, tpred)
    return synthetic, gradient


def init_routine(eql_basics):
    """
    Returns a starting model for the forward calculation.

    If eql_basics.model is set, it returns that as the starting model. If eql_basics.model is
    not set, it returns a default starting model.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class

    -------------
    """
    try:
        model=eql_basics.model
    except:
        model=[9, 46.8, -10] # create starting model, if not set in eql_basics
    return model



def solver(eql_basics, model_start, synthetic, gradient):
    """
    Performs the inversion. Returns a recovered model that is a
    regularised least squares solution given the data and the starting model.

    Further obtains about the results uncertainty (covariance).

    Arguments:
    -------------

    :param eql_basics: Basic parameters of the inversion test problem
    :type eql_basics: class
    :param model: Contains starting values for the Earthquake location.
    :type model: numpy array
    :param synthetics: Contains synthetic data created with the forward calulation
    :type synthetics: class
        - :param sols: Contains the iterative solutions of the inversion (coordinates)
          :type sols: numpy array
        - :param res: The observed arrival time - minus the predicted arrival time
          :type res: numpy array
        - :param tpred: Contains the predicted arrival times
          :type tpred: numpy array
    :param gradient: Empty variable in this inversion test problem.
    :type gradient: list (empty)

    :param result: Contains the result of the inversion and relevant information to understand it.
    :type result: class
        - :param bootstrap_solutions: Contains earthquake location and origin time of all bootstrap iterations in (t,x,y,z)
          :type bootstrap_solutions: numpy array
        - :param bootstrap_cov: Contains covariance values (earthquake location and origin time) of all bootstrap iterations
          :type bootstrap_cov: numpy array
        - :param model_all: Contains the iterative model solutions for earthquake location and origin time in (t,x,y,z)
          :type model_all: numpy array
        - :param model_final: Contains the final earthquake location in (x,y,z) coordinates, in degree (WGS84?)
          :type model_final: numpy array
        - :param orig_t_final: Contains the final origin time estimation, in seconds after 16:30
          :type orig_t_final: float
        - :param orig_t_all: Contains all origin time estimatoions, in seconds after 16:30
          :type orig_t_all: numpy array

    -------------
    """

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
        yBoot = synthetic.tpred + np.random.choice(res,size=len(res),replace=True) # random sample residuals with replacement
        solsB, resB =eq.eqlocate(x0,y0,z0,yBoot,la,lo,el,vp,tol) # here sols are the iterative solutions found,
        bootstrap_solutions[i] = solsB[-1] # bootstrap solution

    bootstrap_cov=np.cov(bootstrap_solutions.T)
    #tpred=ts-res
    model_all=sols
    orig_t_all=sols[:,0]
    orig_t_final=sols[-1,0] # origin time
    model_final=[sols[-1,1], sols[-1,2], sols[-1,3]] # x,y,z,
    result=resultclass(bootstrap_solutions, bootstrap_cov, res, model_all, model_final, orig_t_final, orig_t_all)
    print ('Earthquake location (iterative least square solution):\n', model_final)
    print ('Event time (seconds after 16:30)',orig_t_final)

    return result

def plot_model(eql_basics, result):
    """
    Visualises the recovered model and provides information about its covariance.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class
    :param result: Contains the result of the inversion and relevant information to understand it.
    :type result: class
        - :param bootstrap_solutions: Contains earthquake location and origin time of all bootstrap iterations in (t,x,y,z)
          :type bootstrap_solutions: numpy array
        - :param bootstrap_cov: Contains covariance values (earthquake location and origin time) of all bootstrap iterations
          :type bootstrap_cov: numpy array
        - :param model_all: Contains the iterative model solutions for earthquake location and origin time in (t,x,y,z)
          :type model_all: numpy array
        - :param model_final: Contains the final earthquake location in (x,y,z) coordinates, in degree (WGS84?)
          :type model_final: numpy array
        - :param orig_t_final: Contains the final origin time estimation, in seconds after 16:30
          :type orig_t_final: float
        - :param orig_t_all: Contains all origin time estimatoions, in seconds after 16:30
          :type orig_t_all: numpy array

    --------------------
    """


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

class resultclass():
    """
    Class object containing  the result of the inversion and relevant information to understand it.

    Attributes:
    ----------------
    :param bootstrap_solutions: Contains earthquake location and origin time of all bootstrap iterations in (t,x,y,z)
    :type bootstrap_solutions: numpy array
    :param bootstrap_cov: Contains covariance values (earthquake location and origin time) of all bootstrap iterations
    :type bootstrap_cov: numpy array
    :param model_all: Contains the iterative model solutions for earthquake location and origin time in (t,x,y,z)
    :type model_all: numpy array
    :param model_final: Contains the final earthquake location in (x,y,z) coordinates, in degree (WGS84?)
    :type model_final: numpy array
    :param orig_t_final: Contains the final origin time estimation, in seconds after 16:30
    :type orig_t_final: float
    :param orig_t_all: Contains all origin time estimatoions, in seconds after 16:30
    :type orig_t_all: numpy array
    ----------------

    """

    def __init__(self, bootstrap_solutions, bootstrap_cov, res, model_all, model_final,  orig_t_final, orig_t_all):
        self.bootstrap_solutions=bootstrap_solutions
        self.bootstrap_cov=bootstrap_cov
        self.res=res
        self.model_all=model_all
        self.model_final=model_final
        self.orig_t_final=orig_t_final
        self.orig_t_all=orig_t_all



class synth():
    """
    Class object containing synthetic data of the forward calulation.

    Parameters
    --------------------
    *args

       :param sols: Contains the iterative solutions of the inversion (coordinates)
       :type sols: numpy array
       :param res: The observed arrival time - minus the predicted arrival time
       :type res: numpy array
       :param tpred: Contains the predicted arrival times for comparison with origin time
       :type tpred: numpy array

    --------------------
    """

    def __init__(self, sols, res, tpred):
        self.sols=sols
        self.res=res
        self.tpred=tpred


#def eqlocate(x0,y0,z0,ts,la,lo,el,vpin,tol,solvedep=False,nimax=100,verbose=False,kms2deg=[111.19,75.82]):
    #la2km=kms2deg[0]
    #lo2km=kms2deg[1]

    #i=np.argmin(ts)
    ##i = 4
    #t0=ts[i]-np.sqrt(((x0*lo2km-lo[i]*lo2km)**2)+((y0*la2km-la[i]*la2km)**2)+(el[i]-z0)**2)/vpin[i]  # initial guess origin time

    #ni=0
    #sols=[[t0,x0,y0,z0]]
    #ndata = len(ts) # Number of data

    #while 1:
        #ni=ni+1
        #D0=np.zeros(ndata)
        #for i in range(ndata):
            #D0[i] = np.sqrt(((lo[i]-x0)*lo2km)**2+((la[i]-y0)*la2km)**2+(el[i]-z0)**2)
        #G=[]
        #res=[]
        #for i in range(ndata):
            #vp = vpin[i]
            #if(solvedep):
                #G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp),(z0-el[i])/(D0[i]*vp)])
            #else:
                #G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp)])
            #res.append(ts[i]-(D0[i]/vp+t0))
        #G=np.array(G)
        #res=np.array(res)
        ##print(' ni ',ni)
        ##print('G :\n',G[ni-1])
        ##print('d :\n',d[ni-1])
        #m=np.linalg.lstsq(G,res, rcond=-1)[0]
        #t0=t0+m[0]
        #x0=x0+m[1]/lo2km # update longitude solution and convert to degrees
        #y0=y0+m[2]/la2km # update latitude solution and convert to degrees
        #if(solvedep):
            #z0=z0+m[3]
            #dtol = np.sqrt((m[1]**2+m[2]**2+m[3]**2)) # distance moved by hypocentre
        #else:
            #dtol = np.sqrt(m[1]**2+m[2]**2)
        #chisq = np.dot(res.T,res)
        #if(verbose): print('Iteration :',ni,'Chi-sq:',chisq,' Change in origin time',m[0],' change in spatial distance:',dtol)
        #sols.append([t0,x0,y0,z0])
        #if m[0]<tol[0] and dtol<tol[1]:
            #break
        #if(ni==nimax):
            #print(' Maximum number of iterations reached in eqlocate. No convergence')
            #break
    #sols=np.array(sols)
    #return sols, res

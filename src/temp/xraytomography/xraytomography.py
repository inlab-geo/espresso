import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import tqdm
import os

# Andrew Valentine (andrew.valentine@anu.edu.au)
# Malcolm Sambridge (malcolm.sambridge@anu.edu.au)
#
# Research School of Earth Sciences
# The Australian National University
#
# May 2018
#
# X-ray tomography, after Tarantola Ch. 5.


def displayModel(model,paths=None,extent=(0,1,0,1),clim=None,cmap=None,figsize=(8,6)):
    """
    Function to visualise the recovered model.

    Arguments:

    ---------------
    :param model: Contains attenuation values in a 2-dimensional (N_x * N_y) array
    :type model: numpy array
    :param paths: specifies the start and end points rays displayed in the subsequent model plot.
    :type paths: numpy array
    :param extent: Use to limit the boundary of the model region where rays are allowed to travel.
    :type extent: numpy array
    :param clim: Set the color limits of the current image.
    :type clim: numpy array; see matplotlib.pyplot.clim
    :param cmap: Set the color scheme of the current image.
    :type cmap: numpy array
    :param figsize: Set the figure size of the current image.
    :type fixsize: numpy array

    ---------------
    """

    plt.figure(figsize=figsize)
    if cmap is None: cmap = plt.cm.bone_r

    plt.imshow(model.T,origin='lower',extent=extent,cmap=cmap)


    if paths is not None:
        for p in paths:
            plt.plot([p[0],p[2]],[p[1],p[3]],'k')
    if clim is not None: plt.clim(clim)
    plt.colorbar()

    plt.show()


def tracer(model,paths,extent=(0,1,0,1)):
    """
    For every raypath defined in paths: This function calculates the distance of the
    ray in each model cell.
    The distances are combined with the attenuation of each model cell to estimate
    the overall attenuation rate between source and receiver for each ray path.

    Arguments in:
    ---------------
    :param model: Contains attenuation values in a 2-dimensional (N_x * N_y) array
    :type model: numpy array
    :param paths: specifies the start and end points of each ray traveling through the model.
        - paths[i,0] - x-location of source for path i
        - paths[i,1] - y-location of source for path i
        - paths[i,2] - x-location of receiver for path i
        - paths[i,3] - y-location of receiver for path i
    :type paths: numpy array
    :param extent: Use to limit the boundary of the model region where rays are allowed to travel.
    :type extent: numpy array
    ---------------
    Arguments out:
    ---------------
    :param attns: is the attenuation for each path. It is an array of dimension
    :type attns: numpy array
    :param A: The matrix relating model to data. Contains distance information of each raypath in each model cell.
    :type A: numpy array
    ---------------
    """

    try:
        nx,ny = model.shape
    except:
        raise ValueError("Argument 'model' must be a 2-D numpy array")
    try:
        xmin,xmax,ymin,ymax = extent
    except:
        raise ValueError("Argument 'extent' must be a tuple,list or 1-D array with 4 elements (xmin,xmax,ymin,ymax)")
    if type(paths) == type([]):
        paths = np.array(paths)
    try:
        npaths,ncomp = paths.shape
    except:
        raise ValueError("Argument 'paths' must be a list or 2-D array")
    if ncomp!=4: raise ValueError("Each path must be described by four elements (xstart,ystart,xend,yend)")
    if any(paths[:,0]<xmin) or any(paths[:,0]>xmax) or any(paths[:,1]<ymin) or any(paths[:,1]>ymax) \
                        or any(paths[:,2]<xmin) or any(paths[:,2]>xmax) or any(paths[:,3]<ymin) or any(paths[:,3]>ymax):
        raise ValueError("All sources and receivers must be within or on boundary of model region")

    xGridBounds = np.linspace(xmin,xmax,nx+1)
    yGridBounds = np.linspace(ymin,ymax,ny+1)
    A = np.zeros([npaths,nx*ny])
    attns = np.zeros([npaths])
    #print ""
    t = tqdm.tqdm(desc="Evaluating paths",total=npaths)
    for ip,p in enumerate(paths):
        xs,ys,xr,yr = p
        pathLength = np.sqrt((xr-xs)**2 + (yr-ys)**2)
        # Compute lambda for intersection with each grid-line
        lamX = np.array([]) if xr==xs else np.array([(d-xs)/(xr-xs) for d in xGridBounds])
        lamY = np.array([]) if yr==ys else np.array([(d-ys)/(yr-ys) for d in yGridBounds])
        # Discard any intersections that would occur outside the region
        lamX = np.extract(np.logical_and(lamX>=xmin,lamX<=xmax),lamX)
        nlamX = len(lamX)
        lamY = np.extract(np.logical_and(lamY>=ymin,lamY<=ymax),lamY)
        lam = np.concatenate( (lamX,lamY) )
        lamSort = np.argsort(lam,kind='mergesort')
        dx = 1 if xr>xs else -1
        dy = 1 if yr>ys else -1
        #print lam
        try:
            if lam[lamSort[0]]!=0:
                lam = np.concatenate((np.array([0]),lam))
                lamSort = np.concatenate((np.array([0]),lamSort+1))
                nlamX+=1
        except IndexError:
            lam = np.array([0])
            lamSort = np.array([0])
        if lam[lamSort[-1]]!=1:
            lam = np.concatenate((lam,np.array([1])))
            lamSort = np.concatenate((lamSort,np.array([lam.shape[0]-1])))
        #print lam,lamSort
        if xs==xmin:
            ix = 0
        elif xs==xmax:
            ix = nx-1
        else:
            ix = np.searchsorted(xGridBounds,xs,side='right' if xr>xs else 'left')-1
        if ys==ymin:
            iy = 0
        elif ys==ymax:
            iy = ny-1
        else:
            iy = np.searchsorted(yGridBounds,ys,side='right' if yr>ys else 'left')-1
        #print ix,iy
        pathSensitivity = np.zeros_like(model)
        ilam0=2 if lam[lamSort[1]]==0 else 1
        for ilam in range(ilam0,len(lam)):
            dl = (lam[lamSort[ilam]] - lam[lamSort[ilam-1]])*pathLength
            pathSensitivity[ix,iy] = dl
            attns[ip]+=dl*model[ix,iy]
            if lamSort[ilam]>=nlamX:
                iy+=dy
            else:
                ix+=dx

            #print ix,iy,lam[lamSort[ilam]],(xs+lam[lamSort[ilam]]*(xr-xs),ys+lam[lamSort[ilam]]*(yr-ys))
            if lam[lamSort[ilam]]==1.:break

        A[ip,:] = pathSensitivity.flatten()
        t.update(1)
    t.close()

    return attns,A
####################################################################
# Additions for Inversion Test Suite

class Basics():
    """
    Creates a class object containing basic information about the inversion test problem.

    Attributes:
    --------------------

    :param model_size: defines the model size; model is always squared so setting one value is sufficient.
    :type model_size: int
    :param epsSquared: regularisation parameter
    :type epsSquared: float
    :param noise: defines how much noise is added on the noiseless data prior to inversion.
    :type noise: float
    :param subset: defines how much of the data is used for the inversion. 0 means no data used; 1 means all data used.
    :type subset: float
    :param data: contains attenuation rates for each ray traveling through the model
    :type data: numpy array
    :param paths: specifies the start and end points of each ray traveling through the model.
        - paths[i,0] - x-location of source for path i
        - paths[i,1] - y-location of source for path i
        - paths[i,2] - x-location of receiver for path i
        - paths[i,3] - y-location of receiver for path i
    :type paths: numpy array
    -------
    """


    def data_path(filename):
        path_to_current_file = os.path.realpath(__file__)
        current_directory = os.path.split(path_to_current_file)[0]
        data_path = os.path.join(current_directory, filename)
        return data_path


    model_size=50
    epsSquared=0.001
    noise=0
    subset=1
    dataset = np.loadtxt(data_path('xrt_data.dat'))
    data = np.zeros([np.shape(dataset)[0],2])
    data = -np.log(dataset[:,5]) + np.log(dataset[:,2])
    paths = np.zeros([np.shape(dataset)[0],4])
    paths[:,0] = dataset[:,0]
    paths[:,1] = dataset[:,1]
    paths[:,2] = dataset[:,3]
    paths[:,3] = dataset[:,4]

    del dataset


def init_routine(xrt_basics):
    """
    Returns a starting model for the forward calculation.

    If xrt_basics.model is set, it returns that as the starting model. If xrt_basics.model is
    not set, it returns a default starting model containing ones.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class

    -------------
    """
    try:
        start_model=xrt_basics.model
    except:
        start_model = np.ones([xrt_basics.model_size,xrt_basics.model_size])
    return start_model

def forward(xrt_basics, model):
    """
    Returns the attenuation rate along ray paths given a model. Calculates the attenuation
    of each model cell based on the cell's attenuation rate and the length of the ray path
    within that cell.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class
    :param model: Contains attenuation values in a 2-dimensional (N_x * N_y) array
    :type model: numpy array

    :param synthetics: Contains synthetic data of the forward calulation (attenuation rate) and other parameters needed to understand them
    :type synthetics: class
    :param gradient: Empty variable in this inversion test problem.
    :type gradient: list (empty)

    -------------
    """

    synthetics=list()
    data, G = tracer(model,xrt_basics.paths)
    synthetics=synth(data, G)
    gradient=[]
    return synthetics, gradient

class synth():
    """
    Class object containing synthetic data of the forward calulation and other parameters needed to understand them.

    Parameters
    --------------------
    *args

        data : Contains synthetic attenuation rates for every ray path.

        G : Contains the attenuation of each model cell along each ray path.

    --------------------
    """

    def __init__(self, data, G):
        self.G=G
        self.data=data

def solver(xrt_basics, model, synthetics, gradient):
    """
    Performs the inversion. Returns a recovered model that is a
    regularised least squares solution given the data and the starting model.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class
    :param model: Contains attenuation parameters of each model grid cell in a 2-dimensional (N_x * N_y) array
    :type model: numpy array
    :param synthetics: Contains synthetic data (attenutation rate) and the corresponding
    array showing the distance a ray spent in which grid cell.
    :type synthetics: class
    :param gradient: Empty variable in this inversion test problem.
    :type gradient: list (empty)

    -------------
    """
    G=synthetics.G
    data=xrt_basics.data

    if xrt_basics.subset<1:
        ind=np.random.choice(len(data),round(len(data)*xrt_basics.subset))
        data=data[ind]
        G=G[ind,:]

    data=data+np.random.normal(0,xrt_basics.noise*np.max(data),len(data))

    #print(np.shape(G))
    result = np.linalg.inv((G.T).dot(G) + xrt_basics.epsSquared*np.eye(np.shape(G)[1])).dot(G.T).dot(data)
    return result

def plot_model(xrt_basics,result):
    """
    Visualises the recovered model. This is a wrapper for the underlying function displayModel.

    Arguments:
    -------------

    :param xrt_basics: Basic parameters of the inversion test problem
    :type xrt_basics: class
    :param result: Contains the recovered model as attenuation rates in a 2-dimensional (N_x * N_y) array
    :type result: numpy array

    --------------------
    """

    size=int(np.sqrt(len(result)))
    displayModel(result.reshape(size,size))

# -------------------------------------------------------------------------------
# Useful functions that are not used in this inversion test problem
# -------------------------------------------------------------------------------
#def generateExampleDataset(filename):
    #noiseLevels=None #[0.005,0.01,0.015,0.02,0.025]
    #m = pngToModel('csiro_logo.png',1024,1024,1,1)
    #srcs = np.array([[0,0],[0,0.2],[0.,0.4],[0,0.5],[0,0.6],[0.,0.65],[0.,0.7]]+[[0,x] for x in np.linspace(0.71,1.0,30)]+[[x,0] for x in np.linspace(0.3,0.6,30)])
    #recs = generateSurfacePoints(40,surface=[False,True,False,True])
    ##recs = generateSurfacePoints(50,surface=[False,True,False,False])
    ##srcs = generateSurfacePoints(50,surface=[False,False,True,False])
    #paths = buildPaths(srcs,recs)
    #Isrc = np.random.uniform(10,1,size=paths.shape[0])
    #attns,A = forwardSolver(m,paths)
    #Irec = Isrc*np.exp(-attns)
    #if noiseLevels is not None:
        #noise = np.zeros([paths.shape[0]])
        #for i in range(0,paths.shape[0]):
            #noise[i] = np.random.choice(noiseLevels)
            #Irec[i]+=np.random.normal(0,noise[i])
            #if Irec[i]<=0: Irec[i] = 1.e-3


    #fp = open(filename,'w')
    #fp.write("# Src-x Src-y Src-Int Rec-x Rec-y Rec-Int")
    #if noiseLevels is None:
        #fp.write("\n")
    #else:
        #fp.write(" Rec-sig\n")
    #for i in range(0,paths.shape[0]):
        #fp.write("%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f"%(paths[i,0],paths[i,1],Isrc[i],paths[i,2],paths[i,3],Irec[i]))
        #if noiseLevels is None:
            #fp.write("\n")
        #else:
            #fp.write(" %2.4f\n"%noise[i])
    #fp.close()

#def buildPaths(srcs,recs):
    #if type(srcs) is type([]): srcs = np.array(srcs)
    #try:
        #nsrcs,nc = srcs.shape
    #except:
        #raise ValueError("Argument 'srcs' must be a 2-D nummpy array")
    #if nc!=2: raise ValueError("Argument 'srcs' should have shape (N x 2)")
    #if type(recs) is type([]): recs = np.array(recs)
    #try:
        #nrecs,nc = recs.shape
    #except:
        #raise ValueError("Argument 'recs' must be a 2-D nummpy array")
    #if nc!=2: raise ValueError("Argument 'recs' should have shape (N x 2)")
    #npaths = nsrcs*nrecs
    #paths = np.zeros([npaths,4])
    #ip=0
    #for isrc in range(nsrcs):
        #for irec in range(nrecs):
            #paths[ip,0:2] = srcs[isrc,:]
            #paths[ip,2:4] = recs[irec,:]
            #ip+=1
    #return paths

#def generateSurfacePoints(nPerSide,extent=(0,1,0,1),surface=[True,True,True,True],addCorners=True):
    #out = []
    #if surface[0]:
        #out+=[[extent[0],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    #if surface[1]:
        #out+=[[extent[1],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    #if surface[2]:
        #out+=[[x,extent[2]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    #if surface[3]:
        #out+=[[x,extent[3]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    #if addCorners:
        #if surface[0] or surface[2]:
            #out+=[[extent[0],extent[2]]]
        #if surface[0] or surface[3]:
            #out+=[[extent[0],extent[3]]]
        #if surface[1] or surface[2]:
            #out+=[[extent[1],extent[2]]]
        #if surface[1] or surface[3]:
            #out+=[[extent[1],extent[3]]]
    #return np.array(out)

#def pngToModel(pngfile,nx,ny,bg=1.,sc=1.):
    #png = Image.open(pngfile)
    #png.load()

    #model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.ROTATE_270))/255.)
    #return model

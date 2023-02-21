import numpy as np
from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
from cofi_espresso.utils import loadtxt, absolute_path
from PIL import Image
import matplotlib.pyplot as plt
import tqdm



class XrayTomography(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "X-ray tomography",
        "problem_short_description": "",

        "author_names": ["Andrew Valentine"],

        "contact_name": "Andrew Valentine",
        "contact_email": "andrew.valentine@durham.ac.uk",

        "citations": [("Tarantola, A., 2005. *Inverse problem theory and methods for parameter estimation*, Sec.5.6.","")],

        "linked_sites": [],
    }

    def __init__(self, example_number=1):
        super().__init__(example_number)
        if example_number == 1:
            self._paths, self._attns = load_data('data/example1.dat')
            self._desc = "A straightforward X-ray tomography setup with good data coverage (InLab logo)"
            self._ngrid = 50 
            self._start = np.ones((self._ngrid,self._ngrid))
            self._true = pngToModel('data/inlab_logo.png',self._ngrid,self._ngrid,2,0.5)
        elif example_number == 2:
            self._paths, self._attns = load_data('data/example2.dat')
            self._desc = "A straightforward X-ray tomography setup with good data coverage (CSIRO logo)"
            self._ngrid = 50 
            self._start = np.ones((self._ngrid,self._ngrid))
            self._true = pngToModel('data/csiro_logo.png',self._ngrid,self._ngrid)
        elif example_number == 3:
            self._paths, self._attns = load_data('data/example3.dat')
            self._desc = "X-ray tomography with large gaps (CSIRO logo)"
            self._ngrid = 50
            self._start = np.ones((self._ngrid,self._ngrid))
            self._true = pngToModel('data/csiro_logo.png',self._ngrid,self._ngrid)
        else:
            raise InvalidExampleError

    @property
    def description(self):
        return self._desc

    @property
    def model_size(self):
        return self._ngrid**2

    @property
    def data_size(self):
        return self._attns.size

    @property
    def good_model(self):
        return self._true.flatten()

    @property
    def starting_model(self):
        return self._start.flatten()
    
    @property
    def data(self):
        return self._attns.copy()

    @property
    def covariance_matrix(self):                # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional
        
    def forward(self, model, with_jacobian=False):
        n = model.size
        ngrid = int(n**0.5)
        attns,A = tracer(model.reshape((ngrid,ngrid)),self._paths)
        if with_jacobian:
            return attns, A
        else:
            return attns
    
    def jacobian(self, model):
        n = model.size
        ngrid = int(n**0.5)
        attns,A = tracer(model.reshape((ngrid,ngrid)),self._paths)
        return A

    def plot_model(self, model, paths=False, **kwargs):
        m = model.reshape((self._ngrid,self._ngrid))
        fig = plt.figure()
        ax = fig.subplots(1,1)
        im = ax.imshow(m.T,cmap=plt.cm.Blues,extent=(0,1,0,1),origin='lower',**kwargs)
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.colorbar(im,ax=ax,label='Density')
        if paths:
            for p in self._paths:
                ax.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.05)
        return fig
    
    def plot_data(self, data, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data, data2):              # optional
        raise NotImplementedError

    def log_likelihood(self,data,data2):        # optional
        raise NotImplementedError
    
    def log_prior(self, model):                 # optional
        raise NotImplementedError



def tracer(model,paths,extent=(0,1,0,1)):
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
            if lam[lamSort[ilam]]==1.:break

        A[ip,:] = pathSensitivity.flatten()
        t.update(1)
    t.close()

    return attns,A

def load_data(filename):
    data = loadtxt(filename,skiprows=1)
    ndata = data.shape[0]
    paths = np.zeros((ndata,4))
    attns = np.zeros(ndata)
    paths[:,0:2] = data[:,0:2]
    paths[:,2:4] = data[:,3:5]
    attns = -np.log(data[:,5]/data[:,2])
    return paths, attns

def generateExampleDataset(img_filename, out_filename):
    noiseLevels=None #[0.005,0.01,0.015,0.02,0.025]
    noiseLevels=[0.0001]
    # m = pngToModel(img_filename,1024,1024,1,1)
    m = pngToModel(img_filename,1024,1024,2,0.5)
    # srcs = np.array([[0,0],[0,0.2],[0.,0.4],[0,0.5],[0,0.6],[0.,0.65],[0.,0.7]]+[[0,x] for x in np.linspace(0.71,1.0,30)]+[[x,0] for x in np.linspace(0.3,0.6,30)])
    # recs = generateSurfacePoints(40,surface=[False,True,False,True])
    # recs = generateSurfacePoints(50,surface=[False,True,False,False])
    # srcs = generateSurfacePoints(50,surface=[False,False,True,False])
    recs = generateSurfacePoints(30,surface=[True,True,True,True])
    srcs = generateSurfacePoints(20,surface=[True,True,True,True])
    paths = buildPaths(srcs,recs)
    Isrc = np.random.uniform(10,1,size=paths.shape[0])
    attns,A = tracer(m,paths)
    Irec = Isrc*np.exp(-attns)
    if noiseLevels is not None:
        noise = np.zeros([paths.shape[0]])
        for i in range(0,paths.shape[0]):
            noise[i] = np.random.choice(noiseLevels)
            Irec[i]+=np.random.normal(0,noise[i])
            if Irec[i]<=0: Irec[i] = 1.e-3

    fp = open(out_filename,'w')
    fp.write("# Src-x Src-y Src-Int Rec-x Rec-y Rec-Int")
    if noiseLevels is None:
        fp.write("\n")
    else:
        fp.write(" Rec-sig\n")
    for i in range(0,paths.shape[0]):
        fp.write("%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f"%(paths[i,0],paths[i,1],Isrc[i],paths[i,2],paths[i,3],Irec[i]))
        if noiseLevels is None:
            fp.write("\n")
        else:
            fp.write(" %2.4f\n"%noise[i])
    fp.close()

def buildPaths(srcs,recs):
    if type(srcs) is type([]): srcs = np.array(srcs)
    try:
        nsrcs,nc = srcs.shape
    except:
        raise ValueError("Argument 'srcs' must be a 2-D nummpy array")
    if nc!=2: raise ValueError("Argument 'srcs' should have shape (N x 2)")
    if type(recs) is type([]): recs = np.array(recs)
    try:
        nrecs,nc = recs.shape
    except:
        raise ValueError("Argument 'recs' must be a 2-D nummpy array")
    if nc!=2: raise ValueError("Argument 'recs' should have shape (N x 2)")
    npaths = nsrcs*nrecs
    paths = np.zeros([npaths,4])
    ip=0
    for isrc in range(nsrcs):
        for irec in range(nrecs):
            paths[ip,0:2] = srcs[isrc,:]
            paths[ip,2:4] = recs[irec,:]
            ip+=1
    return paths

def generateSurfacePoints(nPerSide,extent=(0,1,0,1),surface=[True,True,True,True],addCorners=True):
    out = []
    if surface[0]:
        out+=[[extent[0],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    if surface[1]:
        out+=[[extent[1],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    if surface[2]:
        out+=[[x,extent[2]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    if surface[3]:
        out+=[[x,extent[3]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    if addCorners:
        if surface[0] or surface[2]:
            out+=[[extent[0],extent[2]]]
        if surface[0] or surface[3]:
            out+=[[extent[0],extent[3]]]
        if surface[1] or surface[2]:
            out+=[[extent[1],extent[2]]]
        if surface[1] or surface[3]:
            out+=[[extent[1],extent[3]]]
    return np.array(out)

def pngToModel(pngfile,nx,ny,bg=1.,sc=1.):
    png = Image.open(absolute_path(pngfile))
    png.load()

    try:
        model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.Transpose.ROTATE_270))/255.)
    except:
        model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.ROTATE_270))/255.)        
    return model
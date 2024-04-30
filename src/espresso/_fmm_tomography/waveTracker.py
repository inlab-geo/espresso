import subprocess
import shutil
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cartopy

from espresso.utils import absolute_path as path, silent_remove

#--------------------------------------------------------------------------------------------

# This library is a python interface to Nick Rawlinson's 2D Fast Marching Fortran package fm2dss.f90
#
# M. Sambridge 
# July 2022
#--------------------------------------------------------------------------------------------

#  Definitions for waveTracker followed by those from Andrew Valentine's rayTracer.py package

# routines to write data files for fmst
from scipy.sparse import csr_matrix
from scipy.interpolate import RectBivariateSpline
import subprocess

class fmmResult(object):
    # Class variable
    
    def __init__(self):
        self.name = 'Travel time field object'
    def setTimes(self,t):
        self.ttimes = t.copy()
    def setPaths(self,p):
        self.paths = p.copy()
    def setFrechet(self,f):
        self.frechet = f.copy()
    def setTfield(self,w,source):
        self.tfield = w.copy()
        self.tfieldsource = source
        
class gridModel(object):
    def __init__(self,velocities,extent=(0,1,0,1),dicex=8,dicey=8):
        self.nx,self.ny = velocities.shape
        self.velocities=velocities
        self.xmin,self.xmax,self.ymin,self.ymax = extent
        self.xx = np.linspace(self.xmin,self.xmax,self.nx+1)
        self.yy = np.linspace(self.ymin,self.ymax,self.ny+1)
        self.dicex = dicex
        self.dicey = dicey
        self.extent = extent
    def getVelocity(self):
        return self.velocities.copy()
    def getSlowness(self):
        return 1./self.velocities # No copy needed as operation must return copy
    def setVelocity(self,v):
        assert self.velocities.shape == v.shape
        self.velocities = v.copy()
    def setSlowness(self,s):
        assert self.velocities.shape == s.shape
        self.velocities = 1./s
        

    def wavefront_tracker(self,recs,srcs,wdir='wavetracker/fmst',
            verbose=False,paths=False,frechet=False,times=True,tfieldsource=-1,
            tfilename="rtravel.out",ffilename="frechet.out",wfilename="travelt.out",rfilename="raypath.out",
            sourcegrid=1,sourcedice=5,sourcegridsize=10,
            earthradius=6371.0,schemeorder=1,nbsize=0.5,degrees=False,velocityderiv=False):

        v = self. velocities # velocity model node values

        recs = recs.reshape(-1, 2)
        srcs = srcs.reshape(-1, 2)
        if(tfieldsource+1 > len(srcs)): # source requested for travel time field does not exist
            print('Error: Travel time field corresponds to source:',tfieldsource,'\n',
                  '      but total number of sources is',len(srcs),
                  '\n       No travel time field will be calculated.\n')   
        
        # fmst expects input spatial co-ordinates in degrees and velocities in kms/s so we adjust (unless degrees=True)

        kms2deg = 180./(earthradius*np.pi)

        #write out input files for Fast Marching wavefront tracker fmst

        write_fm2dss_input(wdir,paths=paths,frechet=frechet,times=times,tfieldsource=tfieldsource+1,
                           dicex=self.dicex,dicey=self.dicey,sourcegrid=sourcegrid,sourcedice=sourcedice,
                           sourcegridsize=sourcegridsize,earthradius=earthradius,schemeorder=schemeorder,nbsize=nbsize,
                           tfilename=tfilename,ffilename=ffilename,wfilename=wfilename,rfilename=rfilename)  # write out control file

        write_rs(recs,srcs,wdir)    # write out sources and receiver files

        noncushion,nodemap = write_gridc(v,self.extent,wdir) # write data for input velocity model file gridc.txc
    
        write_otimes([[True]*len(recs)]*len(srcs),wdir) # write out rays to be calculated
    
        # run fmst wavefront tracker code from command line
        # see if the executable is there, otherwise do preparation for the executable
        out = run_fm2dss(wdir)
        # if out.returncode:      # re-compile if there's an error
        #     compile_fm2dss()
        #     out = run_fm2dss(wdir)
        if out.returncode:      # add permission if there's a further error
            print("Trying to fix now...")
            try:
                import stat
                exe_file = Path(wdir + "/../build/fm2dss.o")
                st = os.stat(exe_file)
                os.chmod(exe_file, st.st_mode | stat.S_IEXEC)
                print("Execute permission given to fm2dss.o.")
            except:
                print("Failed to fix. Check error message above.")
            out = run_fm2dss(wdir)
        if(verbose): print(' Message from fmm2dss:',out.stdout)
        if(out.returncode != 0):
            print(' The process returned with errorcode:',out.returncode)
            print(' stdout: \n',out.stdout)
            print(' stderr: \n',out.stderr)
            return
        
        # collect results
        if(times):
            ttimes = read_fmst_ttimes(wdir+'/'+tfilename)
            if(not degrees): ttimes*= kms2deg # adjust travel times because inputs are not in degrees
    
        if(paths): 
            raypaths = read_fmst_raypaths(wdir+'/'+rfilename)
    
        if(frechet):
            frechetvals = read_fmst_frechet(wdir+'/'+ffilename,noncushion,nodemap)
            if(not degrees): frechetvals*= kms2deg # adjust travel times because inputs are not in degrees
            if(not velocityderiv): 
                x2 = -(v*v).reshape(-1)
                frechetvals = frechetvals.multiply(x2)

        if(tfieldsource>=0):
            tfieldvals = read_fmst_wave(wdir+'/'+wfilename)
            if(not degrees): tfieldvals*= kms2deg # adjust travel times because inputs are not in degrees
        
    #   build class object to return
        result = fmmResult()
    
        if(times): result.setTimes(ttimes)
        if(paths): result.setPaths(raypaths)
        if(frechet): result.setFrechet(frechetvals)
        if(tfieldsource >-1): result.setTfield(tfieldvals,tfieldsource) # set traveltime field and source id
        return result

def write_fm2dss_input(wdir,
    dicex=8,dicey=8,sourcegrid=1,sourcedice=5,sourcegridsize=10,
    earthradius=6371.0,schemeorder=1,nbsize=0.5,times=True,frechet=False,tfieldsource=0,paths=False,
    tfilename="rtravel.out",ffilename="frechet.out",wfilename="travelt.out",rfilename="raypath.out"):

    #dicex,dicey=8,8              # p-cell: dicing of propagation grid cell wrt to a velocity grid cell
    #sourcegrid=1                 # decide to use a finer propagation grid about each source (y=1,n=0)
    #sourcedice                   # s-cell: number of source cells per propagation grid p-cell;
    # ourcegridsize               # radius of source box in p grid cells 
    #earthradius                  # radius used to convert co-lat and long to kms (spherical Earth approx)
    #schemeorder                  # First-order(0) or mixed-order(1)
    #nbsize                       # narrow band size
    #lttimes = 1                  # bool to calculate travel times (y=1,n=0)
    #tfilename = "rtravel.out"    # filename for output travel times
    #lfrechet = 1                 # bool to calculate Frechet derivatives (0=no,1=yes)
    #ffilename                    # filename for output of  Frechet derivatives
    #tfieldsource                    # Id to calculate travel time field (0=no,>0=source id)
    #wfilename                    # filename for output of travel times
    #lpaths                       # Write out raypaths (<0=all,0=no,>0=source id)
    #rfilename                    # filename for output of raypaths
    lpaths = 0
    if(paths): lpaths=-1
    lttimes=0
    if(times): lttimes = 1
    lfrechet=0
    if(frechet): lfrechet = 1
    
    blank = "c"*70+"\n"
    f = open(wdir+"/fm2dss.in", "w+")
    f.write(blank+"c INPUT PARAMETERS\n"+blank)
    l1 = "sources.dat                    c: File containing source positions\n"
    l2 = "receivers.dat                  c: File containing receiver positions\n"
    l3 = "otimes.dat                     c: File containing source-receiver associations\n"
    l4 = "gridc.vtx                      c: File containing velocity grid information\n"
    f.write(l1+l2+l3+l4)
    string = "{}     {}                        c: Grid dicing in latitude and longitude\n".format(dicex,dicey)
    f.write(string)
    string = "{}                              c: Apply source grid refinement? (0=no,1=yes)\n".format(sourcegrid)
    f.write(string)
    string = "{}     {}                       c: Dicing level and extent of refined grid\n".format(sourcedice,sourcegridsize)
    f.write(string)
    string = "{}                         c: Earth radius in km\n".format(earthradius)
    f.write(string)
    string = "{}                              c: Use first-order(0) or mixed-order(1) scheme\n".format(schemeorder)
    f.write(string)
    string = "{}                            c: Narrow band size (0-1) as fraction of nnx*nnz\n".format(nbsize)
    f.write(string)
    f.write(blank+"c OUTPUT FILES\n"+blank)
    string = "{}                              c: find source-receiver traveltimes (0=no,1=yes)\n".format(lttimes)
    f.write(string)
    f.write(tfilename.ljust(30)+"c: Name of file containing source-receiver traveltimes\n")
    string = "{}                              c: Calculate Frechet derivatives (0=no,1=yes)\n".format(lfrechet)
    f.write(string)
    f.write(ffilename.ljust(30)+"c: Name of file containing Frechet derivatives\n")
    string = "{}                              c: Write traveltime field to file? (0=no,>0=source id)\n".format(tfieldsource)
    f.write(string)
    f.write(wfilename.ljust(30)+"c: Name of file containing traveltime field\n")
    string = "{}                             c: Write out raypaths (<0=all,0=no,>0=source id)\n".format(lpaths)
    f.write(string)
    f.write(rfilename.ljust(30)+"c: Name of file containing raypath geometry\n")

    f.close()
    
def write_rs(recs,srcs,wdir): # write sources and receivers in format for fmst
    f = open(wdir+"/receivers.dat", "w+")
    f.write(" {} \n".format(len(recs)))
    for i in range(len(recs)):
        f.write(" {} {} \n".format(recs[i,1],recs[i,0]))
    f.close()
    f = open(wdir+"/sources.dat", "w+")
    f.write(" {} \n".format(len(srcs)))
    for i in range(len(srcs)):
        f.write(" {} {} \n".format(srcs[i,1],srcs[i,0]))
    f.close()

def write_gridc(v,extent,wdir): # write data for input velocity model file gridc.txc
    #
    # here extent[3],extent[2] is N-S range of grid nodes
    #      extent[0],extent[1] is W-E range of grid nodes
    nx,ny = v.shape
    dlat, dlong = (extent[3]-extent[2])/(ny-1),(extent[1]-extent[0])/(nx-1) # grid node spacing in lat and long

    # gridc.vtx requires a single cushion layer of nodes surrounding the velocty model
    # build velocity model with cushion velocities

    noncushion = np.zeros((nx+2,ny+2),dtype=bool) # bool array to identify cushion and non cushion nodes
    noncushion[1:nx+1,1:ny+1] = True
    
    # mapping from cushion indices to non cushion indices
    nodemap = np.zeros((nx+2,ny+2),dtype=int)
    nodemap[1:nx+1,1:ny+1] = np.array(range((nx*ny))).reshape((nx,ny))
    nodemap=nodemap[:,::-1]

    # build velocity nodes
    # additional boundary layer of velocities are duplicates of the nearest actual velocity value.
    vc = np.ones((nx+2,ny+2))
    vc[1:nx+1,1:ny+1] = v
    vc[1:nx+1,0] = v[:,0]   # add velocities in the cushion boundary layer
    vc[1:nx+1,-1] = v[:,-1] # add velocities in the cushion boundary layer
    vc[0,1:ny+1] = v[0,:]   # add velocities in the cushion boundary layer
    vc[-1,1:ny+1] = v[-1,:] # add velocities in the cushion boundary layer
    vc[0,0],vc[0,-1],vc[-1,0],vc[-1,-1] = v[0,0],v[0,-1],v[-1,0],v[-1,-1]
    vc=vc[:,::-1]

    # write out gridc.vtx file
    dummy = 1.0 # uncertainty in velocity grid parameter (not required here)
    f = open(wdir+"/gridc.vtx", "w+")
    f.write(" {} {} \n".format(ny,nx)) # Number of grid nodes in latitude and longitude 
    f.write(" {} {} \n".format(extent[3],extent[0])) # origin of computational grid
    f.write(" {} {} \n\n".format(dlat, dlong)) # grid node spacing in lat and long
 
    for i in range(nx+2):
        for j in range(ny+2):
            f.write(" {} {} \n".format(vc[i,j],dummy))
        f.write("\n")
    f.close()
    
    return noncushion,nodemap.flatten()

def write_otimes(pathsTF,wdir): # write out paths that are True in 2D list
    f = open(wdir+"/otimes.dat", "w+")
    ns = len(pathsTF)
    nr = len(pathsTF[0])
    for i in range(ns):
        for j in range(nr):
            if(pathsTF[i][j]): 
                f.write(" {} \n".format(1))
            else:
                f.write(" {} \n".format(0))
    f.close()   

def read_fmst_raypaths(filename): # read ray paths from output of fmst
    with open(filename, 'r') as f:
        lines = f.readlines()
        columns = lines[0].split()
        nrays = int(columns[0])
        paths = []
        k=1
        for i in range(nrays):
            lat = []
            long = []
            npts = float(lines[k])
            columns = lines[k].split()
            npts = int(columns[0])
            k+=1
            for j in range(npts):
                columns = lines[k].split()
                lat.append(float(columns[0]))
                long.append(float(columns[1]))
                k+=1
            path = np.array([long,lat]).T
            paths.append(path) 
    f.close()
    return paths 

def read_fmst_ttimes(filename): # read travel times from output of fmst
    with open(filename, 'r') as f:
        lines = f.readlines()
        ids = []
        ttimes = []
        for line in lines:
            columns = line.split()
            ids.append(int(columns[0]))
            ttimes.append(float(columns[1]))
    f.close()
    return np.asarray(ttimes)

def read_fmst_frechet(filename,noncushion,nodemap): # read ray paths from output of fmst
    nx,ny = noncushion.shape
    nx,ny=nx-2,ny-2
    mused = noncushion.flatten()
    frechet = []
    # the file frechet.out is written by a Fortran routine with array indices starting from 1 and
    # includes frechet derivatives for cushion entries that we do not want
    ray = 0 # ray index
    with open(filename, 'r') as f:
        lines = f.readlines()
        k=0
        vals = [] # set up csr_matrix components
        cols = []
        rows = []
        while k < len(lines):
            columns = lines[k].split()
            nparams = int(columns[0])
            k+=1
            kk = 0
            while kk < nparams:
                columns = lines[k].split()
                mod_id = int(columns[0])-1 # input indices start from 1 so adjust.
                if(mused[mod_id]): # ignore model entries corresponding to cushion nodes
                    # map node indices including cushion to original node indices without cushion
                    rows.append(ray)
                    cols.append(nodemap[mod_id])
                    vals.append(float(columns[1]))
                kk+=1
                k+=1
            ray+=1
    f.close()
    
    # convert to csr matrix
    frechet = csr_matrix((vals, (rows, cols)), shape=(ray, nx*ny))
    return frechet

def read_fmst_wave(filename):
    with open(filename, 'r') as f:
        # The grid definition used by fm2sds.f90 here is identical to the v grid specified by extent
        # with origin at the minimum colat and long, equivalent top left on 2D image with z down the page and x to right.
        lines = f.readlines()
        columns = lines[0].split()
        goxd,gozd = float(columns[0]),float(columns[1]) # Origin of grid (theta,phi) degress or long,lat
        columns = lines[1].split()
        nnx,nnz = int(columns[0]),int(columns[1]) # Number of nodes of grid in x and z or long,lat
        columns = lines[2].split()
        dnxd,dnzd = float(columns[0]),float(columns[1]) # cell sizes in x and z (theta,phi) degress or long,lat
        tfield = np.zeros((nnz,nnx))
        k = 3
        for i in range(nnz):
            for j in range(nnx):
                columns = lines[k].split()
                tfield[i,j] = float(columns[0])
                k+=1
    f.close()
    tfield = tfield[:,::-1]
    return tfield

def displayModel(model,paths=None,extent=(0,1,0,1),clim=None,cmap=None,use_geographic=False, 
                 figsize=(6,6),title=None,line=1.0,cline='k',alpha=1.0,wfront=None,cwfront='k',
                 diced=True,dicex=8,dicey=8,cbarshrink=0.6,**wkwargs):
    fig = plt.figure(figsize=figsize)
    
    if use_geographic:
        x0,x1,y0,y1 = extent
        xc = (x0 + x1) / 2
        yc = (y0 + y1) / 2
        nx, ny = model.shape
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        yy, xx = np.meshgrid(y, x)
        zz = model
        cartopy_projection = cartopy.crs.Mercator(
            central_longitude=xc, 
            min_latitude=y0, 
            max_latitude=y1, 
            globe=None,
            latitude_true_scale=None, 
            false_easting=0.0, 
            false_northing=0.0, 
            scale_factor=None
        )
        ax = fig.add_subplot(1, 1, 1, projection=cartopy_projection)
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        if cmap is None:
            cmap = plt.colormaps["Greys_r"]
        cm = ax.pcolormesh(xx, yy, zz, cmap=cmap, transform=cartopy.crs.PlateCarree())
        ax.coastlines(resolution='10m', color='black')
        ax.gridlines(color='k', draw_labels=True)
        fig.colorbar(cm, orientation="horizontal")
    else:
        if cmap is None: cmap = plt.cm.RdBu

        # if diced option plot the actual B-spline interpolated velocity used by fmst program
        
        plotmodel = model
        if(diced):
            plotmodel = dicedgrid(model,extent=extent,dicex=dicex,dicey=dicey) 
        
        plt.imshow(plotmodel.T,origin='lower',extent=extent,cmap=cmap)
        
        if(wfront is None): plt.colorbar(shrink=cbarshrink)

    if paths is not None:
        if(isinstance(paths, np.ndarray) and paths.shape[1] == 4): # we have paths from xrt.tracer so adjust
            paths = changepathsformat(paths)

        for p in paths:
            plt.plot(p[:,0],p[:,1],cline,lw=line,alpha=alpha)
    if clim is not None: plt.clim(clim)
    
    if title is not None: plt.title(title)
    
    if(wfront is not None):
        nx,ny = wfront.shape
        X, Y = np.meshgrid(np.linspace(extent[0],extent[1],nx), np.linspace(extent[2],extent[3],ny))
        plt.contour(X, Y, wfront.T, **wkwargs)  # Negative contours default to dashed.
    

    # plt.show()
    return fig

def dicedgrid(v,extent=[0.,1.,0.,1.],dicex=8,dicey=8):    
    nx,ny = v.shape
    x = np.linspace(extent[0], extent[1],nx)
    y = np.linspace(extent[2], extent[3],ny)
    kx,ky=3,3
    if(nx <= 3): kx = nx-1 # reduce order of B-spline if we have too few velocity nodes
    if(ny <= 3): ky = ny-1 
    rect = RectBivariateSpline(x, y, v,kx=kx,ky=ky)
    xx = np.linspace(extent[0], extent[1],dicex*nx)
    yy = np.linspace(extent[2], extent[3],dicey*ny)
    X,Y = np.meshgrid(xx,yy,indexing='ij')
    vinterp = rect.ev(X,Y)
    return vinterp

def changepathsformat(paths):
    p = np.zeros((len(paths),2,2))
    for i in range(len(paths)):
        p[i,0,0] = paths[i,0]
        p[i,0,1] = paths[i,1]
        p[i,1,0] = paths[i,2]
        p[i,1,1] = paths[i,3]
    return p

#--------------------------------------------------------------------------------------------
def norm(x):
    return np.sqrt(x.dot(x))
def normalise(x):
    return x/norm(x)
def pngToModel(pngfile,nx,ny,bg=1.,sc=1.):
    png = Image.open(pngfile)
    png.load()

    model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.ROTATE_270))/255.)
    return model

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

def run_fm2dss(wdir):
    command = "../build/fm2dss.o"
    return subprocess.run(command,stdout=subprocess.PIPE, text=True,shell=True,cwd=wdir)

def compile_fm2dss():
    # https://github.com/inlab-geo/espresso/blob/main/espresso_machine/build_package/validate.py#L170
    build_dir = path("./build")
    res1 = subprocess.call(["cmake", ".."], cwd=build_dir)
    if res1:
        raise ChildProcessError(f"`cmake .` failed in {build_dir}")
    res2 = subprocess.call(["make"], cwd=build_dir)
    if res2:
        raise ChildProcessError(f"`make` failed in {build_dir}")

def clean_fm2dss():
    shutil.rmtree(path("build"))

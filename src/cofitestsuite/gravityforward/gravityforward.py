import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G
import os



class basics():
    """
    Creates a class object containing basic information about the inversion test problem.

    Each attribute in this class can be set prior to the forward calculation to change the output.

    Attributes:
    --------------------

    :param background_density: Set the background density of the default model
    :type background_density: float
    :param anomaly_density: Set the density of the anomaly of the default model
    :type anomaly_density: float
    :param noise: Set the noise level as % of maximum value of calculated gravitational force
    :type noise: float
    :param x_rec: Set the recording locations in x-direction.
    :type x_rec: numpy array
    :param y_rec: Set the recording locations in y-direction.
    :type y_rec: numpy array

    --------------------
    """

    background_density=10
    anomaly_density=2000
    noise=0

    # Receiver locations in x and y direction
    x_rec = np.linspace(-80.0, 80.0, 17)
    y_rec = np.linspace(-80.0, 80.0, 17)


def init_routine(grav_basics):
    """
    Returns a starting model for the forward calculation.

    If grav_basics.model is set, it returns that as the starting model. If grav_basics.model is
    not set, it returns a default starting model containing ones.

    Arguments:
    -------------

    :param grav_basics: Basic parameters of the inversion test problem
    :type grav_basics: basics

    -------------
    """

    try:
        start_model=eql_basics.model
    except:
        start_model = np.load(file=data_path('Model_1d.npy'))

    start_model[start_model > 10] = grav_basics.anomaly_density
    start_model[start_model == 10] = grav_basics.background_density

    # Combine to get all combinations
    tmp=cartesian((grav_basics.x_rec,grav_basics.y_rec))
    z_rec=np.zeros(len(tmp))
    z_rec=z_rec[:, np.newaxis]
    grav_basics.rec_coords_all=np.append(tmp,z_rec,axis=1)
    del tmp

    return start_model


def forward(grav_basics, model):
    """
    Calculates the gravitational force of each recording location based on the input model.

    Arguments:
    -------------

    :param grav_basics: Basic parameters of the inversion test problem
    :type grav_basics: basics
    :param model: Contains density values in a 1-dimensional array
    :type model: numpy array
    :param synthetics: Contains synthetic data of the forward calulation (graviational response).
    :type synthetics: cell
    :param gradient: Empty variable in this inversion test problem.
    :type gradient: list (empty)

    -------------
    """


    # Create array of all nodes for each direction
    x_node_slice=np.linspace(-30,30,13)
    y_node_slice=np.linspace(-30,30,13)
    z_node_slice=np.linspace(0,-60,13)
    z_node_slice=np.flipud(z_node_slice)
    # Change boundary cells to create larger model:
    x_node_slice[0]=x_node_slice[0]-995
    y_node_slice[0]=y_node_slice[0]-995
    x_node_slice[-1]=x_node_slice[-1]+995
    y_node_slice[-1]=y_node_slice[-1]+995

    # Combine the 3 node arrays to get start&finish of each prism edge
    # 2 rows per array: Start and finish of each edge
    coords_p1=cartesian((z_node_slice[0:-1],y_node_slice[0:-1],x_node_slice[0:-1]))
    coords_p2=cartesian((z_node_slice[1:],y_node_slice[1:],x_node_slice[1:]))

    # Bring output in order for x,y,z
    temp1=coords_p1[:,0]
    temp2=coords_p2[:,0]
    temp1=temp1[:, np.newaxis]
    temp2=temp2[:, np.newaxis]
    z_nodes=np.append(temp1, temp2, axis=1)

    temp1=coords_p1[:,1]
    temp2=coords_p2[:,1]
    temp1=temp1[:, np.newaxis]
    temp2=temp2[:, np.newaxis]
    y_nodes=np.append(temp1, temp2, axis=1)

    temp1=coords_p1[:,2]
    temp2=coords_p2[:,2]
    temp1=temp1[:, np.newaxis]
    temp2=temp2[:, np.newaxis]
    x_nodes=np.append(temp1, temp2, axis=1)
    del temp1, temp2

    gx_rec, gy_rec, gz_rec = calculate_gravity(model, x_nodes, y_nodes, z_nodes, grav_basics.rec_coords_all)

    if grav_basics.noise > 0:
        gx_rec=gx_rec+np.random.normal(0,grav_basics.noise*np.max(np.absolute(gx_rec)),len(gx_rec))
        gy_rec=gy_rec+np.random.normal(0,grav_basics.noise*np.max(np.absolute(gy_rec)),len(gy_rec))
        gz_rec=gz_rec+np.random.normal(0,grav_basics.noise*np.max(np.absolute(gz_rec)),len(gz_rec))

    synthetics = synthetic_class(gx_rec, gy_rec, gz_rec)
    gradient=[]
    return synthetics, gradient



def plot_model(grav_basics, model, synthetics):
    """
    Visualises gx, gy, gz and the input model.

    Arguments:
    -------------

    :param grav_basics: Basic parameters of the inversion test problem
    :type grav_basics: basics
    :param model: Contains density values in a 1-dimensional array
    :type model: numpy array
    :param synthetics: Contains synthetic data of the forward calulation (graviational response).
    :type synthetics: cell

    -------------
    """

    lxr=len(grav_basics.x_rec)
    lyr=len(grav_basics.y_rec)

    limx=max(grav_basics.x_rec)+(grav_basics.x_rec[1]-grav_basics.x_rec[0])*0.5
    limy=max(grav_basics.y_rec)+(grav_basics.y_rec[1]-grav_basics.y_rec[0])*0.5

    plt.figure(figsize=(17, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(grav_basics.rec_coords_all[:,1],grav_basics.rec_coords_all[:,0],s=0.3,color='k')
    plt.imshow(np.reshape(synthetics.gz_rec,[lxr,lyr]),extent=[-limy,limy,-limx,limx])
    plt.title('2D view of gz')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.colorbar(label="Gravity [mGal]")

    model2d=model.reshape(12,12,12)
    plt.subplot(2,2, 2)
    plt.title("Model slice at z = 30 m")
    plt.scatter(grav_basics.rec_coords_all[:,1],grav_basics.rec_coords_all[:,0],s=3,color='r')
    plt.imshow(model2d[7][:][:],extent=[-30,30,-30,30])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.colorbar(label="Density [kg/m$^3$]")
    plt.xlim([-30,30])
    plt.ylim([-30,30])

    plt.subplot(2, 2, 3)
    plt.imshow(np.reshape(synthetics.gx_rec,[lxr,lyr]),extent=[-limy,limy,-limx,limx])
    plt.title('2D view of gx')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.colorbar(label="Gravity [mGal]")

    plt.subplot(2, 2, 4)
    plt.imshow(np.reshape(synthetics.gy_rec,[lxr,lyr]),extent=[-limy,limy,-limx,limx])
    plt.title('2D view of gy')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.colorbar(label="Gravity [mGal]")
    plt.show()





#########################################################################

class synthetic_class():
    """
    Class object containing synthetic data of the forward calulation.

    Parameters
    --------------------
    *args

        gx : Gravitational force in x-direction at all recording stations as calculated by the forward model.
        gy : Gravitational force in y-direction at all recording stations as calculated by the forward model.
        gz : Gravitational force in z-direction at all recording stations as calculated by the forward model.

    --------------------
    """

    def __init__(self, gx_rec, gy_rec, gz_rec):
        self.gx_rec=gx_rec
        self.gy_rec=gy_rec
        self.gz_rec=gz_rec

def kernel(ii,jj,kk,dx,dy,dz,dim):
    """
    Calculates the analytical kernel (or jacobian) used in the forward calculation; based on Plouff (1976). Kernel is called by calculate_gravity.

    Returns the kernel for one component of the gravitational force, i.e. x-, y- or z-direction.

    To-do (minor): Rename g to J. Script does not calculate gravity anymore, only the jacobian.

    Arguments:
    --------------
    :param ii: Specifies edge or prisms in x-direction (start or end of edge). Is either 1 or 2.
    :type ii: int
    :param jj: Specifies edge or prisms in y-direction (start or end of edge). Is either 1 or 2.
    :type jj: int
    :param kk: Specifies edge or prisms in z-direction (start or end of edge). Is either 1 or 2.
    :type kk: int
    :param dx: Contains x-coordinates of all nodes. First column contains x-coordinates of the start of every edge in the model. Second column contains the end of every edge in the model.
    :type dx: numpy array
    :param dy: Contains y-coordinates of all nodes. First column contains y-coordinates of the start of every edge in the model. Second column contains the end of every edge in the model.
    :type dy: numpy array
    :param dz: Contains z-coordinates of all nodes. First column contains z-coordinates of the start of every edge in the model. Second column contains the end of every edge in the model.
    :type dz: numpy array
    :param dim: Sets which component of the gravitational force is calculated. Code can also calculate gradiometry components, which is also set here.
    :type dim: string ('gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gxz', 'gyy', 'gyz', 'gzz')

    :param g: Contains the kernel (or jacobian) of the current set of arguments.
    :type g: numpy array

    --------------

    """


    r = (dx[:, ii] ** 2 + dy[:, jj] ** 2 + dz[:, kk]** 2) ** (0.50)

    dz_r = dz[:, kk] + r
    dy_r = dy[:, jj] + r
    dx_r = dx[:, ii] + r

    dxr = dx[:, ii] * r
    dyr = dy[:, jj] * r
    dzr = dz[:, kk] * r

    dydz = dy[:, jj] * dz[:, kk]
    dxdy = dx[:, ii] * dy[:, jj]
    dxdz = dx[:, ii] * dz[:, kk]

    if dim=="gx":
        g = (-1) ** (ii + jj + kk) * (dy[:, jj] * np.log(dz_r) + dz[:, kk]* np.log(dy_r) - dx[:, ii] * np.arctan(dydz / dxr))
    elif dim=="gy":
        g = (-1) ** (ii + jj + kk) * (dx[:, ii] * np.log(dz_r) + dz[:, kk]* np.log(dx_r) - dy[:, jj] * np.arctan(dxdz / dyr))
    elif dim=="gz":
        g = (-1) ** (ii + jj + kk) * (dx[:, ii] * np.log(dy_r) + dy[:, jj] * np.log(dx_r) - dz[:, kk]* np.arctan(dxdy / dzr))
    elif dim=="gxx":
        arg = dy[:, jj] * dz[:, kk] / dxr
        # It said g-= ... - maybe neet to switch.
        g = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dxdz / (r * dy_r)- np.arctan(arg)+ dx[:, ii]* (1.0 / (1 + arg ** 2.0))* dydz/ dxr ** 2.0* (r + dx[:, ii] ** 2.0 / r)))
    elif dim=="gxy":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = ((-1) ** (ii + jj + kk) * (np.log(dz_r)+ dy[:, jj] ** 2.0 / (r * dz_r)+ dz[:, kk] / r- 1.0/ (1 + arg ** 2.0)* (dz[:, kk] / r ** 2)* (r - dy[:, jj] ** 2.0 / r)))
    elif dim=="gxz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = ((-1) ** (ii + jj + kk) * (np.log(dy_r)+ dz[:, kk] ** 2.0 / (r * dy_r)+ dy[:, jj] / r- 1.0/ (1 + arg ** 2.0)* (dy[:, jj] / (r ** 2))* (r - dz[:, kk] ** 2.0 / r)))
    elif dim=="gyy":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dydz / (r * dx_r)- np.arctan(arg)+ dy[:, jj]* (1.0 / (1 + arg ** 2.0))* dxdz/ dyr ** 2.0* (r + dy[:, jj] ** 2.0 / r)))
    elif dim=="gyz":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = ((-1) ** (ii + jj + kk) * (np.log(dx_r)+ dz[:, kk] ** 2.0 / (r * (dx_r))+ dx[:, ii] / r- 1.0/ (1 + arg ** 2.0)* (dx[:, ii] / (r ** 2))* (r - dz[:, kk] ** 2.0 / r)))
    elif dim=="gzz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        gxx = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dxdz / (r * dy_r)- np.arctan(arg)+ dx[:, ii]* (1.0 / (1 + arg ** 2.0))* dydz/ dxr ** 2.0* (r + dx[:, ii] ** 2.0 / r)))
        arg = dx[:, ii] * dz[:, kk] / dyr
        gyy = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dydz / (r * dx_r)- np.arctan(arg)+ dy[:, jj]* (1.0 / (1 + arg ** 2.0))* dxdz/ dyr ** 2.0* (r + dy[:, jj] ** 2.0 / r)))
        g=-gxx-gyy


    return g

def calculate_gravity(model, x_final, y_final, z_final, recvec):
    """
    The analytical kernel used in the forward calculation; based on Plouff (1976). Kernel is called by calculate_gravity.

    To-do (minor): Rename x_final etc into soemthing more intuitive.

    Arguments:
    --------------
    :param model: Contains density values in a 1-dimensional array
    :type model: numpy array
    :param x_final: Contains x-coordinates of the edges of all prisms in the model in a (Nx2) array. Values are in metre.
        - x_final[:,0] - start coordinate of edges, in m
        - x_final[:,1] - end coordinate of edges, in m
    :param y_final: Contains x-coordinates of the edges of all prisms in the model in a (Nx2) array. Values are in metre.
        - y_final[:,0] - start coordinate of edges, in m
        - y_final[:,1] - end coordinate of edges, in m
    :param z_final: Contains x-coordinates of the edges of all prisms in the model in a (Nx2) array. Values are in metre.
        - z_final[:,0] - start coordinate of edges, in m
        - z_final[:,1] - end coordinate of edges, in m
    :param recvec: Contains coordinates of all recording locations in a (Nx3) array.  Values are in metre.
        - recvec[:,0] - x-coordinates of recording locations, in m
        - recvec[:,1] - y-coordinates of recording locations, in m
        - recvec[:,2] - z-coordinates of recording locations, in m
    :type recvec: numpy array

    --------------

    """


    tol=1e-4

    gx_rec=np.zeros(len(recvec))
    gy_rec=np.zeros(len(recvec))
    gz_rec=np.zeros(len(recvec))

    for recno in range(len(recvec)):

        dx=x_final-recvec[recno,0]
        dy=y_final-recvec[recno,1]
        dz=z_final-recvec[recno,2]

        min_x=np.min(np.diff(dx))
        min_y=np.min(np.diff(dy))
        min_z=np.min(np.diff(dz))

        dx[np.abs(dx) / min_x < tol] = tol * min_x
        dy[np.abs(dy) / min_y < tol] = tol * min_y
        dz[np.abs(dz) / min_z < tol] = tol * min_z

        Jx=0
        Jy=0
        Jz=0

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):

                    # gx, gy apppear to work, but need confcirmation.
                    # gz is tested and works
                    Jx+=kernel(ii,jj,kk,dx,dy,dz,"gx")
                    Jy+=kernel(ii,jj,kk,dx,dy,dz,"gy")
                    Jz+=kernel(ii,jj,kk,dx,dy,dz,"gz")

        # Multiply J (Nx1) with the model density (Nx1) element-wise
        gx_rec[recno] = 1e8*G*sum(model*Jx)
        gy_rec[recno] = 1e8*G*sum(model*Jy)
        gz_rec[recno] = 1e8*G*sum(model*Jz)

    return gx_rec, gy_rec, gz_rec

def cartesian(arrays, out=None):
    """
    Creates all combinations between input arrays. Used to create coordinates of all grid cells and recording locations.

    Arguments:
    -----------

    :param arrays: Input arrays as a [n,m] array, where n is the number of grid points in one direction and m is the number of directions.
    :type arrays: numpy array
    :param out: Returns all possible coordinate combinations as a [n^m,m] array.
    :type out: numpy array

    -----------
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def data_path(filename):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, filename)
    return data_path

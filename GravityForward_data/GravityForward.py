import numpy as np
import matplotlib.pyplot as plt
# from discretize.utils import mkvc
from scipy.constants import G




class Basics():
    
    # Some parameters that can be changed
    background_density=10
    anomaly_density=2000
    noise=0
    
    # Receiver locations in x and y direction
    x_rec = np.linspace(-80.0, 80.0, 17)
    y_rec = np.linspace(-80.0, 80.0, 17)


def init_routine(grav_basics):
    try:
        start_model=eql_basics.model
    except:
        start_model = np.load(file='GravityForward_data/Model_1d.npy')
        
    start_model[start_model > 10] = grav_basics.anomaly_density
    start_model[start_model == 10] = grav_basics.background_density
    
    # Combine to get all combinations 
    tmp=cartesian((grav_basics.x_rec,grav_basics.y_rec))
    z_rec=np.zeros(len(tmp))
    z_rec=z_rec[:, np.newaxis]
    grav_basics.rec_coords=np.append(tmp,z_rec,axis=1)
    del tmp

    return start_model


def forward(grav_basics, model):
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
    
    gx_rec, gy_rec, gz_rec = calculate_gravity(model, x_nodes, y_nodes, z_nodes, grav_basics.rec_coords)   
    
    synthetics = syntheticsmaker(gx_rec, gy_rec, gz_rec)
    gradient=[]
    return synthetics, gradient 
    


def plot_model(grav_basics, model, synthetics):
    lxr=len(grav_basics.x_rec)
    lyr=len(grav_basics.y_rec)
    
    limx=max(grav_basics.x_rec)+(grav_basics.x_rec[1]-grav_basics.x_rec[0])*0.5
    limy=max(grav_basics.y_rec)+(grav_basics.y_rec[1]-grav_basics.y_rec[0])*0.5

    plt.figure(figsize=(17, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(np.reshape(synthetics.gz_rec,[lxr,lyr]),extent=[-limy,limy,-limx,limx])
    plt.scatter(grav_basics.rec_coords[:,1],grav_basics.rec_coords[:,0],s=0.3,color='k')
    plt.title('2D view of gz')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.colorbar(label="Gravity [mGal]")

    model2d=model.reshape(12,12,12)
    plt.subplot(2,2, 2)
    plt.title("Model slice at z = 30 m")
    plt.imshow(model2d[7][:][:],extent=[-30,30,-30,30])
    plt.scatter(grav_basics.rec_coords[:,1],grav_basics.rec_coords[:,0],s=3,color='r')
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

class syntheticsmaker():
    def __init__(self, gx_rec, gy_rec, gz_rec):
        self.gx_rec=gx_rec
        self.gy_rec=gy_rec
        self.gz_rec=gz_rec

def kernel(ii,jj,kk,dx,dy,dz,dim):
    
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
        # It said g-= ... - maybe neet to switch vorzeichen?
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

    # Tolerance implementation follows SimPEG, discussed in Nagy et al., 2000
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


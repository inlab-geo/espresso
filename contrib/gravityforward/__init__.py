import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pkgutil

class gravity:
    """
    Description of Gravityclass.
    
    Parameters:
    --------------------
    
    
    
    --------------------
    Functions(?):
    --------------------
    
    
    
    --------------------
    """
    
    def __init__(self, example_number=0):
        self._ieg=example_number
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


        if self._ieg==0:
            
            name =__name__
            
            # within package/mymodule1.py, for example
            
            tmp = pkgutil.get_data(__name__, "data/gravmodel1.txt")
            tmp2=np.frombuffer(tmp, dtype='S3')
            self.m=tmp2.astype(np.float)
            del  tmp, tmp2
            
            #self.m=tmp3.astype(float)
            #del tmp, tmp2, tmp3
            
            self.lmx = 12
            self.lmy = 12
            self.lmz = 12
            self.lrx = 17
            self.lry = 17
            ##del tmp
            
            # Receiver locations in x and y direction
            x_rec = np.linspace(-80.0, 80.0, self.lrx)
            y_rec = np.linspace(-80.0, 80.0, self.lry)
            tmp=cartesian((x_rec,y_rec))
            z_rec=np.zeros(len(tmp))
            z_rec=z_rec[:, np.newaxis]
            self.rec_coords=np.append(tmp,z_rec,axis=1)
            del tmp

            # Create array of all nodes for each direction
            x_node_slice=np.linspace(-30,30,self.lmx+1)
            y_node_slice=np.linspace(-30,30,self.lmy+1)
            z_node_slice=np.linspace(0,-60,self.lmz+1)
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
            self.z_nodes=np.append(temp1, temp2, axis=1)

            temp1=coords_p1[:,1]
            temp2=coords_p2[:,1]
            temp1=temp1[:, np.newaxis]
            temp2=temp2[:, np.newaxis]
            self.y_nodes=np.append(temp1, temp2, axis=1)

            temp1=coords_p1[:,2]
            temp2=coords_p2[:,2]
            temp1=temp1[:, np.newaxis]
            temp2=temp2[:, np.newaxis]
            self.x_nodes=np.append(temp1, temp2, axis=1)
            del temp1, temp2

            
        else:
            print("Error - example number not defined")
    
    
    def get_model(self):
        """
        Description of get_data().

        Parameters
        --------------------
        *args

        data_x:
        data_y:
        data_z:
        m:
        rec_coords:
        x_nodes:
        y_nodes:
        z_nodes:
        lmx:
        lmy:
        lmz:
        lrx:
        lry:

        --------------------
        """
        # def data_path(filename):
        #     path_to_current_file = os.path.realpath(__file__)
        #     current_directory = os.path.split(path_to_current_file)[0]
        #     data_path = os.path.join(current_directory)
        #     data_path=data_path+"/data/"+filename
        #     return data_path
        
        return self.m

    
    def _kernel(self, ii,jj,kk,dx,dy,dz,dim):
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

    def _calculate_gravity(self, model, x_final, y_final, z_final, recvec, jacobian):
    
        from scipy.constants import G as G
        # Tolerance implementation follows Nagy et al., 2000
        tol=1e-4
        #gx_rec=np.zeros(len(recvec))
        #gy_rec=np.zeros(len(recvec))
        gz_rec=np.zeros(len(recvec))
        if jacobian==True:
            #Jx_rec=np.zeros([len(recvec),len(x_final)])
            #Jy_rec=np.zeros([len(recvec),len(x_final)])
            Jz_rec=np.zeros([len(recvec),len(x_final)])
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
                        #Jx+=self._kernel(ii,jj,kk,dx,dy,dz,"gx")
                        #Jy+=self._kernel(ii,jj,kk,dx,dy,dz,"gy")
                        Jz+=self._kernel(ii,jj,kk,dx,dy,dz,"gz")
            # Multiply J (Nx1) with the model density (Nx1) element-wise
            #gx_rec[recno] = 1e5*G*sum(model*Jx)
            #gy_rec[recno] = 1e5*G*sum(model*Jy)
            gz_rec[recno] = 1e5*G*sum(model*Jz)
            if jacobian==True:
                #Jx_rec[recno,:] = Jx
                #Jy_rec[recno,:] = Jy
                Jz_rec[recno,:] = Jz

        if jacobian==False:
            return gz_rec
        else:
            return Jz_rec
        
    def forward(self, m):

        gz_rec = self._calculate_gravity(m, self.x_nodes, self.y_nodes, self.z_nodes, self.rec_coords, False)
        
        #data=self._synthetic_class(gx_rec, gy_rec, gz_rec)
        
        return gz_rec
    
    def gradient(self, m):
        
        Jz= self._calculate_gravity(m, self.x_nodes, self.y_nodes, self.z_nodes, self.rec_coords, True)
        #gradient=self._gradient_class(Jx, Jy, Jz)

        return Jz
    
    def plot_model(self, m, data):
        if self._ieg == 0:
            
            gx_rec=data.gx_rec
            gy_rec=data.gy_rec
            gz_rec=data.gz_rec
            
            lrx=self.lrx
            lry=self.lry
            lmx=self.lmx
            lmy=self.lmy
            lmz=self.lmx

            limx=max(self.rec_coords[:,0])+(self.rec_coords[1,0]-self.rec_coords[0,0])*0.5
            limy=max(self.rec_coords[:,1])+(self.rec_coords[1,1]-self.rec_coords[0,1])*0.5
            
            plt.figure(figsize=(17, 12))
            plt.subplot(2, 2, 1)
            plt.scatter(self.rec_coords[:,1],self.rec_coords[:,0],s=0.3,color='k')
            plt.imshow(np.reshape(gz_rec,[lrx,lry]),extent=[-limy,limy,-limx,limx])
            plt.title('2D view of gz')
            plt.xlabel('y [m]')
            plt.ylabel('x [m]')
            plt.colorbar(label="Gravity [mGal]")

            model2d=m.reshape(lmx,lmy,lmz)
            plt.subplot(2, 2, 2)
            plt.title("Model slice at z = 30 m")
            plt.scatter(self.rec_coords[:,1],self.rec_coords[:,0],s=3,color='r')
            plt.imshow(model2d[7][:][:],extent=[-30,30,-30,30])
            plt.xlabel('y [m]')
            plt.ylabel('x [m]')
            plt.colorbar(label="Density [kg/m$^3$]")
            plt.xlim([-30,30])
            plt.ylim([-30,30])

            plt.subplot(2, 2, 3)
            plt.imshow(np.reshape(gx_rec,[lrx,lry]),extent=[-limy,limy,-limx,limx])
            plt.title('2D view of gx')
            plt.xlabel('y [m]')
            plt.ylabel('x [m]')
            plt.colorbar(label="Gravity [mGal]")

            plt.subplot(2, 2, 4)
            plt.imshow(np.reshape(gy_rec,[lrx,lry]),extent=[-limy,limy,-limx,limx])
            plt.title('2D view of gy')
            plt.xlabel('y [m]')
            plt.ylabel('x [m]')
            plt.colorbar(label="Gravity [mGal]")
            plt.show()

        else:
            print("Error - example number not defined")
    
        
    

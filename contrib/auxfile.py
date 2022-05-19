import numpy as np
class auxclass:
    
        
    @staticmethod
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
            auxclass.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
            #for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        return out

    def node_maker(x_node_slice, y_node_slice, z_node_slice):

        coords_p1=cartesian((z_node_slice[0:-1],y_node_slice[0:-1],x_node_slice[0:-1]))
        coords_p2=cartesian((z_node_slice[1:],y_node_slice[1:],x_node_slice[1:]))

        temp1=coords_p1[:,0]
        temp2=coords_p2[:,0]
        temp1=temp1[:, np.newaxis]
        temp2=temp2[:, np.newaxis]
        z_final=np.append(temp1, temp2, axis=1)

        temp1=coords_p1[:,1]
        temp2=coords_p2[:,1]
        temp1=temp1[:, np.newaxis]
        temp2=temp2[:, np.newaxis]
        y_final=np.append(temp1, temp2, axis=1)

        temp1=coords_p1[:,2]
        temp2=coords_p2[:,2]
        temp1=temp1[:, np.newaxis]
        temp2=temp2[:, np.newaxis]
        x_final=np.append(temp1, temp2, axis=1)


        return x_final, y_final, z_final

    def inject_density(model,x_final, y_final, z_final, x,y,z, value):
        # Just a little fcn that allows you to change the density of a grid cell based on coordinates. 
        # Easier to make specific changes this way.

        x1=x_final[:,0]
        y1=y_final[:,0]
        z1=z_final[:,1]

        bool1=x1==x
        bool2=y1==y
        bool3=z1==z

        ind=np.where(ar(bool1) & ar(bool2) & ar(bool3))
        ind=np.squeeze(ind[0])

        model[ind]=value

        return model

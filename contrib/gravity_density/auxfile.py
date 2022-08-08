# This file contains functions that are vital for the Gravityforward -
# Inversion Test Problem, but are not the focus and therefore stored
# "under the hood" in this auxiliary file.

import numpy as np

class auxclass:
    @staticmethod
    def _cartesian(arrays, out=None):
        """Calculates the cartesian product between arrays and/or single numbers.

        Used to create coordinates of all grid cells and recording locations.

        *args

        :param arrays: Input arrays as a [n,m] array, where n is the number of grid points in one direction and m is the number of directions.
        :type arrays: numpy array
        :param out: Returns all possible coordinate combinations as a [n^m,m] array.
        :type out: numpy array

        """
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
        # m = n / arrays[0].size
        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            auxclass._cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                # for j in xrange(1, arrays[0].size):
                out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def _node_maker(x_node_slice, y_node_slice, z_node_slice):
        """ Creates arrays with coordinates for all nodes of the model.

        Each returned array contains separate columns for the start and end
        coordinates of each node: A model with M grid cells has M+1 nodes, and
        node_maker returns 3 Mx2 arrays. These arrays together contain the start
        and end coordinates of all nodes.

        *args

        :param x_node_slice: x-coordinates of all nodes
        :type x_node_slice: numpy array
        :param : y-coordinates of all nodes
        :type y_node_slice: numpy array
        :param : z-coordinates of all nodes
        :type z_node_slice: numpy array
        :param x_nodes: x-values of the cartesian product of the three input arrays
        :type x_nodes: numpy array
        :param y_nodes: y-values of the cartesian product of the three input arrays
        :type y_nodes: numpy array
        :param z_nodes: z-values of the cartesian product of the three input arrays
        :type z_nodes: numpy array



        """

        coords_p1 = auxclass._cartesian(
            (z_node_slice[0:-1], y_node_slice[0:-1], x_node_slice[0:-1])
        )
        coords_p2 = auxclass._cartesian(
            (z_node_slice[1:], y_node_slice[1:], x_node_slice[1:])
        )

        temp1 = coords_p1[:, 0]
        temp2 = coords_p2[:, 0]
        temp1 = temp1[:, np.newaxis]
        temp2 = temp2[:, np.newaxis]
        z_nodes = np.append(temp1, temp2, axis=1)

        temp1 = coords_p1[:, 1]
        temp2 = coords_p2[:, 1]
        temp1 = temp1[:, np.newaxis]
        temp2 = temp2[:, np.newaxis]
        y_nodes = np.append(temp1, temp2, axis=1)

        temp1 = coords_p1[:, 2]
        temp2 = coords_p2[:, 2]
        temp1 = temp1[:, np.newaxis]
        temp2 = temp2[:, np.newaxis]
        x_nodes = np.append(temp1, temp2, axis=1)

        return x_nodes, y_nodes, z_nodes

    def _inject_density(model, x_nodes, y_nodes, z_nodes, x, y, z, value):
        """ A function to change the density of a single grid cell based on coordinates.

        :param model: The model in a 1-D array containing densities [1xM]
        :type model: numpy array
        :param x_nodes: x-values of the cartesian product of all node coordinates (x, y and z)
        :type x_nodes: numpy array
        :param y_nodes: y-values of the cartesian product of all node coordinates (x, y and z)
        :type y_nodes: numpy array
        :param z_nodes: z-values of the cartesian product of all node coordinates (x, y and z)
        :type z_nodes: numpy array
        :param x: x-coordinate that should be changed to the new density value
        :type x: numpy array
        :param y: y-coordinate that should be changed to the new density value
        :type y: numpy array
        :param z: z-coordinate that should be changed to the new density value
        :type z: numpy array
        :param value: New density value
        :type value: float

        """

        x1 = x_nodes[:, 0]
        y1 = y_nodes[:, 0]
        z1 = z_nodes[:, 1]

        bool1 = x1 == x
        bool2 = y1 == y
        bool3 = z1 == z

        ind = np.where(np.asarray(bool1) & np.asarray(bool2) & np.asarray(bool3))
        ind = np.squeeze(ind[0])

        model[ind] = value

        return model

"""Gravity density library file

Author: Hannes Hollmann, 2022

"""

import pkgutil
from io import StringIO
import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt

from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
from cofi_espresso.utils import loadtxt


class GravityDensity(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Gravity calculation from a density model",
        "problem_short_description": "This example implements a simple gravity" \
                                    "forward problem. The model represents density within" \
                                    "the earth on a 3D Cartesian grid.",

        "author_names": ["Hannes Hollmann"],

        "contact_name": "Hannes Hollmann",
        "contact_email": "hannes.hollmann@anu.edu.au",

        "citations": [],
        "linked_sites": [],
    }

    def __init__(self, example_number=1):
        super().__init__(example_number)

        """you might want to set other useful example specific parameters here
        so that you can access them in the other functions see the following as an 
        example (suggested) usage of `self.params`
        """
        setup_params = _setup(example_number)
        if setup_params:
            _to_expand = [
                "m", 
                "rec_coords", 
                "x_nodes", 
                "y_nodes", 
                "z_nodes", 
                "lmx", 
                "lmy", 
                "lmz", 
                "lrx", 
                "lry"
            ]
            for idx, name in enumerate(_to_expand):
                self.params[name] = setup_params[idx]
        else:
            raise InvalidExampleError

    @property
    def model_size(self):
        return self.lmx*self.lmy*self.lmz

    @property
    def data_size(self):
        return len(self.rec_coords)

    @property
    def starting_model(self):
        return np.zeros_like(self.m)

    @property
    def good_model(self):
        return self.m
    
    @property
    def data(self):
        m = self.m
        x_nodes = self.x_nodes
        y_nodes = self.y_nodes
        z_nodes = self.z_nodes
        rec_coords = self.rec_coords
        gz_rec = _calculate_gravity(m, x_nodes, y_nodes, z_nodes, rec_coords)
        datan=gz_rec+np.random.normal(0,0.005*np.max(np.abs(gz_rec)),len(gz_rec))
        return datan

    @property
    def covariance_matrix(self):
        print("This needs fixing")
        return np.zeros([self.data_size,self.data_size])

    def forward(self, model, with_jacobian=False):
        x_nodes = self.x_nodes
        y_nodes = self.y_nodes
        z_nodes = self.z_nodes
        rec_coords = self.rec_coords
        res = _calculate_gravity(
            model, x_nodes, y_nodes, z_nodes, rec_coords, with_jacobian
        )
        return res
    
    def jacobian(self, model):
        x_nodes = self.x_nodes
        y_nodes = self.y_nodes
        z_nodes = self.z_nodes
        rec_coords = self.rec_coords
        res = _calculate_gravity(model, x_nodes, y_nodes, z_nodes, rec_coords, True)
        return res[1]

    def plot_model(self, model):
        rec_coords = self.rec_coords
        lmx = self.lmx
        lmy = self.lmy
        lmz = self.lmz

        if self.example_number == 1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            model2d = model.reshape(lmx, lmy, lmz)
            ax.set_title("Model slice at z = 30 m")
            ax.scatter(rec_coords[:, 1], rec_coords[:, 0], s=3, color="r")
            img = ax.imshow(model2d[7][:][:], extent=[-30, 30, -30, 30])
            ax.set_xlabel("y [m]")
            ax.set_ylabel("x [m]")
            plt.colorbar(img, label="Density [kg/m$^3$]")
            ax.set_xlim([-30, 30])
            ax.set_ylim([-30, 30])
            return fig
        elif self.example_number == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(rec_coords[:, 0], rec_coords[:, 1], s=0.3, color="k")
            img = ax.imshow(np.reshape(model, [lmz, lmx]))
            ax.set_title("2D view of the model")
            ax.set_xlabel("y [m]")
            ax.set_ylabel("z [m]")
            plt.colorbar(img, label="Density [kg/m^3]")
            return fig
        else:
            raise NotImplementedError               # optional
    
    def plot_data(self, data):
        rec_coords = self.rec_coords
        lrx = self.lrx
        lry = self.lry
        gz_rec = data
        limx = (
            max(rec_coords[:, 0])
            + (rec_coords[1, 0] - rec_coords[0, 0]) * 0.5
        )
        limy = (
            max(rec_coords[:, 1])
            + (rec_coords[1, 1] - rec_coords[0, 1]) * 0.5
        )
        if self.example_number == 1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(rec_coords[:, 1], rec_coords[:, 0], s=0.3, color="k")
            img = ax.imshow(
                np.reshape(gz_rec, [lrx, lry]), extent=[-limy, limy, -limx, limx]
            )
            ax.set_title("2D view of gz")
            ax.set_xlabel("y [m]")
            ax.set_ylabel("x [m]")
            plt.colorbar(img, label="Gravity [mGal]")
            return fig
        elif self.example_number == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(rec_coords[:, 0], data)
            ax.set_title("gz")
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Gravity [mGal]")
            ax.grid()
            return fig
        else:
            raise NotImplementedError               # optional


def _kernel(ii, jj, kk, dx, dy, dz, dim):
    r"""Calculates parts of the Jacobian/design matrix/geometry
    using the approach described by Plouff et al, 1976.

    Parameters

    *args

    :param ii: Specifies either start (1) or end (2) of edge in x-direction
    :type ii: int
    :param jj: Specifies either start (1) or end (2) of edge in y-direction
    :type jj: int
    :param kk: Specifies either start (1) or end (2) of edge in z-direction
    :type kk: int
    :param dx: x-coordinates of all edges in the model (start and end in separate
        columns); [Nx2]
    :type dx: numpy array
    :param dy: y-coordinates of all edges in the model (start and end in separate
        columns); [Nx2]
    :type dy: numpy array
    :param dz: z-coordinates of all edges in the model (start and end in separate
        columns); [Nx2]
    :type dz: numpy array
    :param dim: Specifies which part of the gravitational force is calculated.
        For Espresso, this is hard-coded to "gz".
        Possible options are:
        - "gx": x-component of the gravitational force
        - "gy": y-component of the gravitational force
        - "gz": z-component of the gravitational force
        - "gxx": Derivative of the x-component, in x-direction
        - "gyy": Derivative of the x-component, in y-direction
        - "gxz": Derivative of the x-component, in z-direction
        - "gyy": Derivative of the y-component, in y-direction
        - "gyz": Derivative of the y-component, in z-direction
        - "gzz": Derivative of the z-component, in z-direction
    :type dim: string
    :param g: Vertical component of the gravitational force at the recording
        locations
    :type g: numpy array

    """

    r = (dx[:, ii] ** 2 + dy[:, jj] ** 2 + dz[:, kk] ** 2) ** (0.50)
    dz_r = dz[:, kk] + r
    dy_r = dy[:, jj] + r
    dx_r = dx[:, ii] + r
    dxr = dx[:, ii] * r
    dyr = dy[:, jj] * r
    dzr = dz[:, kk] * r
    dydz = dy[:, jj] * dz[:, kk]
    dxdy = dx[:, ii] * dy[:, jj]
    dxdz = dx[:, ii] * dz[:, kk]
    if dim == "gx":
        g = (-1) ** (ii + jj + kk) * (
            dy[:, jj] * np.log(dz_r)
            + dz[:, kk] * np.log(dy_r)
            - dx[:, ii] * np.arctan(dydz / dxr)
        )
    elif dim == "gy":
        g = (-1) ** (ii + jj + kk) * (
            dx[:, ii] * np.log(dz_r)
            + dz[:, kk] * np.log(dx_r)
            - dy[:, jj] * np.arctan(dxdz / dyr)
        )
    elif dim == "gz":
        g = (-1) ** (ii + jj + kk) * (
            dx[:, ii] * np.log(dy_r)
            + dy[:, jj] * np.log(dx_r)
            - dz[:, kk] * np.arctan(dxdy / dzr)
        )
    elif dim == "gxx":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = (-1) ** (ii + jj + kk) * (
            dxdy / (r * dz_r)
            + dxdz / (r * dy_r)
            - np.arctan(arg)
            + dx[:, ii]
            * (1.0 / (1 + arg**2.0))
            * dydz
            / dxr**2.0
            * (r + dx[:, ii] ** 2.0 / r)
        )
    elif dim == "gxy":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = (-1) ** (ii + jj + kk) * (
            np.log(dz_r)
            + dy[:, jj] ** 2.0 / (r * dz_r)
            + dz[:, kk] / r
            - 1.0
            / (1 + arg**2.0)
            * (dz[:, kk] / r**2)
            * (r - dy[:, jj] ** 2.0 / r)
        )
    elif dim == "gxz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = (-1) ** (ii + jj + kk) * (
            np.log(dy_r)
            + dz[:, kk] ** 2.0 / (r * dy_r)
            + dy[:, jj] / r
            - 1.0
            / (1 + arg**2.0)
            * (dy[:, jj] / (r**2))
            * (r - dz[:, kk] ** 2.0 / r)
        )
    elif dim == "gyy":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = (-1) ** (ii + jj + kk) * (
            dxdy / (r * dz_r)
            + dydz / (r * dx_r)
            - np.arctan(arg)
            + dy[:, jj]
            * (1.0 / (1 + arg**2.0))
            * dxdz
            / dyr**2.0
            * (r + dy[:, jj] ** 2.0 / r)
        )
    elif dim == "gyz":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = (-1) ** (ii + jj + kk) * (
            np.log(dx_r)
            + dz[:, kk] ** 2.0 / (r * (dx_r))
            + dx[:, ii] / r
            - 1.0
            / (1 + arg**2.0)
            * (dx[:, ii] / (r**2))
            * (r - dz[:, kk] ** 2.0 / r)
        )
    elif dim == "gzz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        gxx = (-1) ** (ii + jj + kk) * (
            dxdy / (r * dz_r)
            + dxdz / (r * dy_r)
            - np.arctan(arg)
            + dx[:, ii]
            * (1.0 / (1 + arg**2.0))
            * dydz
            / dxr**2.0
            * (r + dx[:, ii] ** 2.0 / r)
        )
        arg = dx[:, ii] * dz[:, kk] / dyr
        gyy = (-1) ** (ii + jj + kk) * (
            dxdy / (r * dz_r)
            + dydz / (r * dx_r)
            - np.arctan(arg)
            + dy[:, jj]
            * (1.0 / (1 + arg**2.0))
            * dxdz
            / dyr**2.0
            * (r + dy[:, jj] ** 2.0 / r)
        )
        g = -gxx - gyy
    return g

def _calculate_gravity(model, x_nodes, y_nodes, z_nodes, rec_coords, with_jacobian=False):
    # Tolerance implementation follows Nagy et al., 2000
        tol = 1e-4
        # gx_rec=np.zeros(len(rec_coords))
        # gy_rec=np.zeros(len(rec_coords))
        gz_rec = np.zeros(len(rec_coords))
        if with_jacobian:
            # Jx_rec=np.zeros([len(rec_coords),len(x_nodes)])
            # Jy_rec=np.zeros([len(rec_coords),len(x_nodes)])
            Jz_rec = np.zeros([len(rec_coords), len(x_nodes)])
        for recno in range(len(rec_coords)):
            dx = x_nodes - rec_coords[recno, 0]
            dy = y_nodes - rec_coords[recno, 1]
            dz = z_nodes - rec_coords[recno, 2]
            min_x = np.min(np.diff(dx))
            min_y = np.min(np.diff(dy))
            min_z = np.min(np.diff(dz))
            dx[np.abs(dx) / min_x < tol] = tol * min_x
            dy[np.abs(dy) / min_y < tol] = tol * min_y
            dz[np.abs(dz) / min_z < tol] = tol * min_z
            Jx = 0
            Jy = 0
            Jz = 0
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        # Jx+=_kernel(ii,jj,kk,dx,dy,dz,"gx")
                        # Jy+=_kernel(ii,jj,kk,dx,dy,dz,"gy")
                        Jz += _kernel(ii, jj, kk, dx, dy, dz, "gz")
            # Multiply J (Nx1) with the model density (Nx1) element-wise

            # Result is multiplied by 1e5 to convert from m/s^2 to mGal
            # gx_rec[recno] = 1e5*G*sum(model*Jx)
            # gy_rec[recno] = 1e5*G*sum(model*Jy)
            gz_rec[recno] = 1e5 * G * sum(model * Jz)
            if with_jacobian:
                # Jx_rec[recno,:] = Jx
                # Jy_rec[recno,:] = Jy
                Jz_rec[recno, :] = Jz

        if with_jacobian:
            return gz_rec, Jz_rec
        else:
            return gz_rec

def _setup(num):
    # tmp = pkgutil.get_data(__name__, "data/gravmodel1.txt")
    # tmp2 = tmp.decode("utf-8")
    # m = np.loadtxt(StringIO(tmp2))
    m = loadtxt("data/gravmodel1.txt")
    # del tmp, tmp2

    if num == 1:
        lmx = 12
        lmy = 12
        lmz = 12
        lrx = 17
        lry = 17

        # Receiver locations in x and y direction
        x_rec = np.linspace(-80.0, 80.0, lrx)
        y_rec = np.linspace(-80.0, 80.0, lry)
        tmp = auxclass._cartesian((x_rec, y_rec))
        z_rec = np.zeros(len(tmp))
        z_rec = z_rec[:, np.newaxis]
        rec_coords = np.append(tmp, z_rec, axis=1)
        del tmp

        # Create array of all nodes for each direction
        x_node_slice = np.linspace(-30, 30, lmx + 1)
        y_node_slice = np.linspace(-30, 30, lmy + 1)
        z_node_slice = np.linspace(0, -60, lmz + 1)
        z_node_slice = np.flipud(z_node_slice)
        # Change boundary cells to create larger model:
        x_node_slice[0] = x_node_slice[0] - 995
        y_node_slice[0] = y_node_slice[0] - 995
        x_node_slice[-1] = x_node_slice[-1] + 995
        y_node_slice[-1] = y_node_slice[-1] + 995

        # Combine the 3 node arrays to get start&finish of each prism edge
        # 2 rows per array: Start and finish of each edge
        coords_p1 = auxclass._cartesian(
            (z_node_slice[0:-1], y_node_slice[0:-1], x_node_slice[0:-1])
        )
        coords_p2 = auxclass._cartesian(
            (z_node_slice[1:], y_node_slice[1:], x_node_slice[1:])
        )

        # Bring output in order for x,y,z
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
        del temp1, temp2
        return m, rec_coords, x_nodes, y_nodes, z_nodes, lmx, lmy, lmz, lrx, lry

    elif num == 2:

        # Define number of model cells and recording locations
        lmx = 9
        lmy = 1
        lmz = 3
        lrx = 9
        lry = 1

        ## Create node coordinates:
        x_node_slice = np.linspace(-45, 45, lmx + 1)
        y_node_slice = np.linspace(-1000, 1000, lmy + 1)
        z_node_slice = np.linspace(-30, 0, lmz + 1)
        x_nodes, y_nodes, z_nodes = auxclass._node_maker(
            x_node_slice, y_node_slice, z_node_slice
        )

        ## Define recording coordinates
        x_rec = np.linspace(-40, 40, lrx)
        y_rec = np.zeros(np.shape(x_rec))
        x_rec = x_rec[:, np.newaxis]
        y_rec = y_rec[:, np.newaxis]
        tmp = np.append(x_rec, y_rec, axis=1)
        z_rec = np.zeros(len(tmp))
        z_rec = z_rec[:, np.newaxis]
        rec_coords = np.append(tmp, z_rec, axis=1)
        del tmp

        ## Create model and insert density anomaly
        # Create model
        m = np.zeros((lmx) * (lmy) * (lmz))
        # Change specific cells to higher densities (anomaly)
        anomaly = 1000
        m = auxclass._inject_density(
            m, x_nodes, y_nodes, z_nodes, -5, -1000, 0, anomaly
        )
        m = auxclass._inject_density(
            m, x_nodes, y_nodes, z_nodes, -15, -1000, -10, anomaly
        )
        m = auxclass._inject_density(
            m, x_nodes, y_nodes, z_nodes, -5, -1000, -10, anomaly
        )
        m = auxclass._inject_density(
            m, x_nodes, y_nodes, z_nodes, 5, -1000, -10, anomaly
        )
        m = auxclass._inject_density(
            m, x_nodes, y_nodes, z_nodes, -5, -1000, -20, anomaly
        )
        return m, rec_coords, x_nodes, y_nodes, z_nodes, lmx, lmy, lmz, lrx, lry

    else:
        return False


class auxclass:
    """This class contains functions that are vital for the GravityForward - Espresso
    but are not the focus and therefore stored "under the hood" in this auxiliary class
    """
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

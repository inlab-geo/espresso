import numpy as np
import matplotlib.pyplot as plt
# import re
import pkgutil
from io import StringIO
import importlib

# Import class auxclass containing functions like kernel
auxclass = getattr(
    importlib.import_module(".gravityforward.auxfile", package="inversiontestproblems"),
    "auxclass",
)


class gravityforward(auxclass):
    r""" Returns the vertical component of the gravitational force using a 3D density model.

    This Inversion Test Problem explores the gravitational response of a
    three-dimensional model containing densities at specified receiver locations.
    This problem returns the z-component of the gravitational response, but the
    underlying code can return all three gravity and 6 gradiometry components,
    if needed.

    To calculate the gravitational response, the mass of each model cell has to
    be estimated. This Inversion Test Problem uses an analytical approach to
    calculate the mass, based on Plouff et al., 1976.

    :param example_number: Specify to choose between different model set-ups.
        See documentation for more information - currently valid options:
        - 0: A symmetric model with a high density cube located centrally.
        - 1: The pseud-3D 'cross' example from Last et al., 1983 (Figure 2).
    :type example_number: int
    :param m: The model in a 1-D array containing densities [1xM]
    :type m: numpy array
    :param rec_coords: Array containing coonrdinates of recording stations [Nx3]
    :type rec_coords: numpy array
    :param x_nodes: X-coordinates of all nodes in model [2xM]
    :type x_nodes: numpy array
    :param y_nodes: Y-coordinates of all nodes in model [2xM]
    :type y_nodes: numpy array
    :param z_nodes: Z-coordinates of all nodes in model [2xM]
    :type z_nodes: numpy array
    :param lmx: Number of cells in model; x-direction
    :type lmx: numpy array
    :param lmy: Number of cells in model; y-direction
    :type lmy: numpy array
    :param lmz: Number of cells in model; z-direction
    :type lmz: numpy array
    :param lrx: Number of recording stations; x-direction
    :type lrx: numpy array
    :param lry: Number of recording stations; y-direction
    :type lry: numpy array

    """

    def __init__(self, example_number=0):
        self._ieg = example_number

        if self._ieg == 0:

            name = __name__

            tmp = pkgutil.get_data(__name__, "data/gravmodel1.txt")
            tmp2 = tmp.decode("utf-8")
            self.m = np.loadtxt(StringIO(tmp2))
            del tmp, tmp2

            self.lmx = 12
            self.lmy = 12
            self.lmz = 12
            self.lrx = 17
            self.lry = 17

            # Receiver locations in x and y direction
            x_rec = np.linspace(-80.0, 80.0, self.lrx)
            y_rec = np.linspace(-80.0, 80.0, self.lry)
            tmp = auxclass._cartesian((x_rec, y_rec))
            z_rec = np.zeros(len(tmp))
            z_rec = z_rec[:, np.newaxis]
            self.rec_coords = np.append(tmp, z_rec, axis=1)
            del tmp

            # Create array of all nodes for each direction
            x_node_slice = np.linspace(-30, 30, self.lmx + 1)
            y_node_slice = np.linspace(-30, 30, self.lmy + 1)
            z_node_slice = np.linspace(0, -60, self.lmz + 1)
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
            self.z_nodes = np.append(temp1, temp2, axis=1)

            temp1 = coords_p1[:, 1]
            temp2 = coords_p2[:, 1]
            temp1 = temp1[:, np.newaxis]
            temp2 = temp2[:, np.newaxis]
            self.y_nodes = np.append(temp1, temp2, axis=1)

            temp1 = coords_p1[:, 2]
            temp2 = coords_p2[:, 2]
            temp1 = temp1[:, np.newaxis]
            temp2 = temp2[:, np.newaxis]
            self.x_nodes = np.append(temp1, temp2, axis=1)
            del temp1, temp2

        elif self._ieg == 1:

            # Define number of model cells and recording locations
            self.lmx = 9
            self.lmy = 1
            self.lmz = 3
            self.lrx = 9
            self.lry = 1

            ## Create node coordinates:
            x_node_slice = np.linspace(-45, 45, self.lmx + 1)
            y_node_slice = np.linspace(-1000, 1000, self.lmy + 1)
            z_node_slice = np.linspace(-30, 0, self.lmz + 1)
            self.x_nodes, self.y_nodes, self.z_nodes = auxclass._node_maker(
                x_node_slice, y_node_slice, z_node_slice
            )

            ## Define recording coordinates
            x_rec = np.linspace(-40, 40, self.lrx)
            y_rec = np.zeros(np.shape(x_rec))
            x_rec = x_rec[:, np.newaxis]
            y_rec = y_rec[:, np.newaxis]
            tmp = np.append(x_rec, y_rec, axis=1)
            z_rec = np.zeros(len(tmp))
            z_rec = z_rec[:, np.newaxis]
            self.rec_coords = np.append(tmp, z_rec, axis=1)
            del tmp

            ## Create model and insert density anomaly
            # Create model
            m = np.zeros((self.lmx) * (self.lmy) * (self.lmz))
            # Change specific cells to higher densities (anomaly)
            anomaly = 1000
            m = auxclass._inject_density(
                m, self.x_nodes, self.y_nodes, self.z_nodes, -5, -1000, 0, anomaly
            )
            m = auxclass._inject_density(
                m, self.x_nodes, self.y_nodes, self.z_nodes, -15, -1000, -10, anomaly
            )
            m = auxclass._inject_density(
                m, self.x_nodes, self.y_nodes, self.z_nodes, -5, -1000, -10, anomaly
            )
            m = auxclass._inject_density(
                m, self.x_nodes, self.y_nodes, self.z_nodes, 5, -1000, -10, anomaly
            )
            self.m = auxclass._inject_density(
                m, self.x_nodes, self.y_nodes, self.z_nodes, -5, -1000, -20, anomaly
            )

        else:

            raise ValueError("The chosen example-number does not match any examples for this Inversion Test Problem.")

    def get_model(self):
        r""" Returns the starting model for the forward calculation.

        This is some random additional text. Does it show up somewhere? If not, why?
        And also, can I click on the get model link now?

        Parameters
        ----------
        *args

        m: The model in a 1-D array containing densities


        """

        return self.m

    def get_data(self, m):
        r"""Returns synthetic measurements of gravitational force with added gaussian noise;

        Parameters
        ----------
        *args

        datan: Measurements of gravitational force, with added
            gaussian noise.


        """

        gz_rec = self._calculate_gravity(
            m, self.x_nodes, self.y_nodes, self.z_nodes, self.rec_coords, False
        )

        self.datan=gz_rec+np.random.normal(0,0.005*np.max(gz_rec),len(gz_rec))

        return self.datan

    def _kernel(self, ii, jj, kk, dx, dy, dz, dim):
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
            For Inversion Test Problems, this is hard-coded to "gz".
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

    def _calculate_gravity(self, model, x_nodes, y_nodes, z_nodes, rec_coords, jacobian):
        """ Calculates the gravitational force for each recording location based using the input model.

        *args
        :param model: The model in a 1-D array containing densities; [1xM]
        :type model: numpy array
        :param rec_coords: Array containing coonrdinates of recording stations; [Nx3]
        :type rec_coords: numpy array
        :param x_nodes: X-coordinates of all nodes in model; [2xM]
        :type x_nodes: numpy array
        :param y_nodes: Y-coordinates of all nodes in model; [2xM]
        :type y_nodes: numpy array
        :param z_nodes: Z-coordinates of all nodes in model; [2xM]
        :type z_nodes: numpy array
        :param jacobian: Specifies whether to return gravity (False) or Jacobian(True)
        :type jacobian: bool
        :param gz_rec: Vertical component of the gravitational force at the recording
            locations; [Nx1]
        :type gz_rec: numpy array

        """

        from scipy.constants import G as G

        # Tolerance implementation follows Nagy et al., 2000
        tol = 1e-4
        # gx_rec=np.zeros(len(rec_coords))
        # gy_rec=np.zeros(len(rec_coords))
        gz_rec = np.zeros(len(rec_coords))
        if jacobian == True:
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
                        # Jx+=self._kernel(ii,jj,kk,dx,dy,dz,"gx")
                        # Jy+=self._kernel(ii,jj,kk,dx,dy,dz,"gy")
                        Jz += self._kernel(ii, jj, kk, dx, dy, dz, "gz")
            # Multiply J (Nx1) with the model density (Nx1) element-wise

            # Result is multiplied by 1e5 to convert from m/s^2 to mGal
            # gx_rec[recno] = 1e5*G*sum(model*Jx)
            # gy_rec[recno] = 1e5*G*sum(model*Jy)
            gz_rec[recno] = 1e5 * G * sum(model * Jz)
            if jacobian == True:
                # Jx_rec[recno,:] = Jx
                # Jy_rec[recno,:] = Jy
                Jz_rec[recno, :] = Jz

        if jacobian == False:
            return gz_rec
        else:
            return Jz_rec

    def forward(self, m):
        r"""Calculates the gravitational force of each recording location.

        *args
        :param m: The model in a 1-D array containing densities; [1xM]
        :type m: numpy array
        :param gz_rec: Vertical component of the gravitational force at the recording
            locations; [Nx1]
        :type gz_rec: numpy array

        """
        gz_rec = self._calculate_gravity(
            m, self.x_nodes, self.y_nodes, self.z_nodes, self.rec_coords, False
        )

        # data=self._synthetic_class(gx_rec, gy_rec, gz_rec)

        return gz_rec

    def gradient(self, m):
        r"""Returns the Jacobian / design matrix / problem geometry.

        Returns the Jacobian given the model and recording locations in a [NxM]
        array, with N being the number of recording locations and M being the
        number of model cells.


        :param m: The model in a 1-D array containing densities; [1xM]
        :type m: numpy array
        :param Jz: The jacobian / design matrix / problem geometry; [NxM]
        :type Jz: numpy array


        """



        Jz = self._calculate_gravity(
            m, self.x_nodes, self.y_nodes, self.z_nodes, self.rec_coords, True
        )
        # gradient=self._gradient_class(Jx, Jy, Jz)

        return Jz

    def plot_model(self, m, data):
        r"""Visualisation of the input model and the vertical gravity component.

        *args

        :param m: The model in a 1-D array containing densities [1xM]
        :type m: numpy array
        :param example_number: Specify to choose between different model set-ups.
            See documentation for more information - currently valid options:
            - 0: A symmetric model with a high density cube located centrally.
            - 1: The pseud-3D 'cross' example from Last et al., 1983 (Figure 2).

        """

        if self._ieg == 0:

            # gx_rec=data.gx_rec
            # gy_rec=data.gy_rec
            gz_rec = data

            lrx = self.lrx
            lry = self.lry
            lmx = self.lmx
            lmy = self.lmy
            lmz = self.lmx

            limx = (
                max(self.rec_coords[:, 0])
                + (self.rec_coords[1, 0] - self.rec_coords[0, 0]) * 0.5
            )
            limy = (
                max(self.rec_coords[:, 1])
                + (self.rec_coords[1, 1] - self.rec_coords[0, 1]) * 0.5
            )

            plt.figure(figsize=(17, 12))
            plt.subplot(1, 2, 1)
            plt.scatter(self.rec_coords[:, 1], self.rec_coords[:, 0], s=0.3, color="k")
            plt.imshow(
                np.reshape(gz_rec, [lrx, lry]), extent=[-limy, limy, -limx, limx]
            )
            plt.title("2D view of gz")
            plt.xlabel("y [m]")
            plt.ylabel("x [m]")
            plt.colorbar(label="Gravity [mGal]")

            model2d = m.reshape(lmx, lmy, lmz)
            plt.subplot(1, 2, 2)
            plt.title("Model slice at z = 30 m")
            plt.scatter(self.rec_coords[:, 1], self.rec_coords[:, 0], s=3, color="r")
            plt.imshow(model2d[7][:][:], extent=[-30, 30, -30, 30])
            plt.xlabel("y [m]")
            plt.ylabel("x [m]")
            plt.colorbar(label="Density [kg/m$^3$]")
            plt.xlim([-30, 30])
            plt.ylim([-30, 30])

            plt.show()
        elif self._ieg == 1:

            gz_rec = data

            lrx = self.lrx
            lry = self.lry
            lmx = self.lmx
            lmy = self.lmy
            lmz = self.lmx

            limx = (
                max(self.rec_coords[:, 0])
                + (self.rec_coords[1, 0] - self.rec_coords[0, 0]) * 0.5
            )
            limy = (
                max(self.rec_coords[:, 1])
                + (self.rec_coords[1, 1] - self.rec_coords[0, 1]) * 0.5
            )

            # model2d=model.reshape(3,1,9)
            # plt.figure(figsize=(17, 8))
            # plt.subplot(2, 1, 1)
            # plt.title("Model slice at y=0")
            # plt.imshow(model2d[:,0,:])
            # plt.colorbar(label="density [kg/m^3]")

            plt.figure(figsize=(17, 8))
            plt.subplot(1, 2, 1)
            plt.scatter(self.rec_coords[:, 0], self.rec_coords[:, 1], s=0.3, color="k")
            plt.imshow(np.reshape(m, [self.lmz, self.lmx]))
            plt.title("2D view of the model")
            plt.xlabel("y [m]")
            plt.ylabel("z [m]")
            plt.colorbar(label="Density [kg/m^3]")

            plt.subplot(1, 2, 2)
            plt.plot(self.rec_coords[:, 0], data)
            plt.title("gz")
            plt.xlabel("Distance [m]")
            plt.ylabel("Gravity [mGal]")
            plt.grid()

            plt.show()


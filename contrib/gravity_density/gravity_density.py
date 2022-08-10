"""Gravity density library file

Author: Hannes Hollmann, 2022

"""

import pkgutil
from io import StringIO
import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt

from .lib import auxclass


_params = {"example_number": 0}

def set_example_number(num):
    _params["example_number"] = num
    setup_params = _setup(_params["example_number"])
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
        _params[name] = setup_params[idx]

def suggested_model():
    return _params["m"]

def data():
    m = _params["m"]
    x_nodes = _params["x_nodes"]
    y_nodes = _params["y_nodes"]
    z_nodes = _params["z_nodes"]
    rec_coords = _params["rec_coords"]
    gz_rec = _calculate_gravity(m, x_nodes, y_nodes, z_nodes, rec_coords)
    datan=gz_rec+np.random.normal(0,0.005*np.max(np.abs(gz_rec)),len(gz_rec))
    return datan

def forward(model, with_jacobian=False):
    r"""Calculates the gravitational force of each recording location.

    *args
    :param m: The model in a 1-D array containing densities; [1xM]
    :type m: numpy array
    :param gz_rec: Vertical component of the gravitational force at the recording
        locations; [Nx1]
    :type gz_rec: numpy array

    """
    x_nodes = _params["x_nodes"]
    y_nodes = _params["y_nodes"]
    z_nodes = _params["z_nodes"]
    rec_coords = _params["rec_coords"]
    res = _calculate_gravity(
        model, x_nodes, y_nodes, z_nodes, rec_coords, with_jacobian
    )
    return res

def jacobian(model):
    x_nodes = _params["x_nodes"]
    y_nodes = _params["y_nodes"]
    z_nodes = _params["z_nodes"]
    rec_coords = _params["rec_coords"]
    jac = _calculate_gravity(model, x_nodes, y_nodes, z_nodes, rec_coords)
    return jac

def plot_model(model):
    rec_coords = _params["rec_coords"]
    lmx = _params["lmx"]
    lmy = _params["lmy"]
    lmz = _params["lmz"]

    if _params["example_number"] == 0:
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
    elif _params["example_number"] == 1:
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

def plot_data(data):
    rec_coords = _params["rec_coords"]
    lrx = _params["lrx"]
    lry = _params["lry"]
    gz_rec = data
    limx = (
        max(rec_coords[:, 0])
        + (rec_coords[1, 0] - rec_coords[0, 0]) * 0.5
    )
    limy = (
        max(rec_coords[:, 1])
        + (rec_coords[1, 1] - rec_coords[0, 1]) * 0.5
    )
    if _params["example_number"] == 0:
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
    elif _params["example_number"] == 1:
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
    tmp = pkgutil.get_data(__name__, "data/gravmodel1.txt")
    tmp2 = tmp.decode("utf-8")
    m = np.loadtxt(StringIO(tmp2))
    del tmp, tmp2

    if num == 0:
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

    elif num == 1:

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
        raise ValueError("The chosen example-number does not match any examples for this Inversion Test Problem.")


set_example_number(0)

This Inversion Test Problem explores the gravitational acceleration of a three-dimensional (example 1) 
and pseude-2D (example 2) density model onto specified receiver locations. In this example, 
only the z-component of the gravityational force 
is calculated. The underlying code itself is capable of calculating all three gravity components 
and six gradiometry components and could be modified quickly if there is the need. 

The gravitational acceleration is calculated using Newton's law of universal gravitation: 

$$
    g (r) =- G \frac{ m} {r^2} 
$$

With G being the gravitational constant, r is the distance of the mass
to the receiver and m is the overall mass of the model, which depends on the density $\rho$ and the volume V:

$$
    m = \int_V {\rho(r) dV}
$$

Here, we solve volume integral for the vertical component of $g$ analytically, using the approach by Plouff et al., 1976:

$$
g_z(M,N)=G \rho \sum_{i=1}^2 \sum_{j=1}^2 \sum_{k=1}^2  (-1)^{i+j+k} [tan^{-1} \frac{a_ib_j}{z_k R_{ijk}} - a_i ln(R_{ijk} + b_j) - b_j ln(R_{ijk} + a_i)]
$$

with $R_{ijk}=\sqrt{a_i^2 + b_j^2 + z_k^2}$ and $a_i, b_j, z_k$ being the distances from receiver N to the 
nodes of the current prism M (i.e. grid cell) in x, y, and z directions. It is assumed that $\rho=const.$ within each grid cell. 
For more information, please see the original paper: 

Plouff, D., 1976. *Gravity and magnetic fields of polygonal prisms and application to magnetic terrain corrections.* **Geophysics**, 41(4), pp.727-741

For further reading, see also Nagy et al., 2000:

Nagy, D., Papp, G. and Benedek, J., 2000. *The gravitational potential and its derivatives for the prism.* **Journal of Geodesy**, 74(7), pp.552-560

**Example details:**

 1. **Model:** Density values on a regularly spaced, rectangular grid. Example-model one is a 3D cube of low density (10 $kgm^{-3}$) containing a centrally located high-density cube (1000 $kgm^{-3}$). Example-model two repeats Figure 2 of Last and Kubik, 1983, which means a pseudo-2D model containing zero-density background cells and centrally high-density cells in the shape of a cross (1000 $kgm^{-3}$).

    Last, B.J. and Kubik, K., 1983. *Compact gravity inversion.* **Geophysics**, 48(6), pp.713-721
    
 2. **Returned data:** Gravitational acceleration (vertical component).
    
 3. **Forward:**  The volume integral is solved analytically following the above described approach by Plouff et al., 1976.

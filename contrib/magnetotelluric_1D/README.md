# Magnetotelluric 1D

## Theoretical background

The magnetotelluric (MT) method is a passive electromagnetic method that uses the fluctuations of the natural electromagnetic field to determine the distribution of the electrical conductivity in the subsurface. When the electric and magnetic fields are measured simultaneously on the surface, their complex ratio (the impedance) can be used to describe the penetration of the EM fields in the Earth, which is dependent on their frequencies and the electrical conductivity of the subsurface. 

The impedance $Z$ is a 4x4 tensor which relates electric ( $E$ ) and magnetic ( $H$ ) fields in the x and y directions in Cartesian coordinates (x, y):

$$
\begin{pmatrix}
E_{x}\\ 
E_{y}
\end{pmatrix} = 
\begin{pmatrix}
Z_{xx} & Z_{xy}\\
Z_{yx} & Z_{yy}  
\end{pmatrix} \cdot
\begin{pmatrix}
H_{x}\\
H_{y} 
\end{pmatrix}
$$


In these examples, we assume that the Earth is 1-D, that there is no lateral variations of the electrical conductivity in the x and y directions, only in the z direction. In that case the impedance tensor simplifies to:

$$
Z = 
\begin{pmatrix}
 0  & Z_{1D}\\
-Z_{1D} & 0
\end{pmatrix} 
$$

Therefore, the response of a 1D layered Earth is $Z_{1D}$, and is defined as a function of frequency. In 1-D, determining $Z$ given the electrical conductivity values of the Earth (the forward model) is done using a recursive approach (Wait, 1954; Pedersen and Hermance, 1986). 

Because of its practicality to visually interpret and analyse the data, it is common to represent the complex impedance tensor by its magnitude (the apparent resistivity) and its phase, defined by:

$$
\rho_{app} = \frac{1}{5f} |Z|^2
$$

$$
\Phi = tan^{-1} \frac{\Im(Z)}{\Re(Z)}
$$

with $f$ the frequency defined in Hz, the apparent resistivity $\rho_{app}$ defined in $\Omega m$ and the phase in degrees. 

Details regarding the MT method can be found in Simpson and Bahr (2005) and Chave and Jones (2012).

As mentioned earlier, the penetration of the EM fields depends on the frequency and the electrical conductivity of the Earth. Lower frequencies will penetrate deeper into the Earth, and conductive material will attenuate faster the EM fields. Therefore, depending on the frequency range available (and the composition of the Earth), the MT method can be used to map the distribution of the electrical conductivity into the Earth from tens of meters to hundreds of kilometres. Programs such as AusLAMP (for example Robertson et al., 2016) aims at imaging the Australia lithosphere and upper mantle using long period MT data, while Audio MT (AMT) and Broad Band MT (BBMT) data are used to image the upper crust (for example Simpson et al., 2021 or Jiang et al., 2022). 

### References

*Chave, A. D., & Jones, A. G. (Eds.). (2012). The magnetotelluric method: Theory and practice. Cambridge University Press.*

*Jiang, W., Duan, J., Doublier, M., Clark, A., Schofield, A., Brodie, R. C., & Goodwin, J. (2022). Application of multiscale magnetotelluric data to mineral exploration: An example from the east Tennant region, Northern Australia. Geophysical Journal International, 229(3), 1628-1645.*

*Pedersen, J., & Hermance, J. F. (1986). Least squares inversion of one-dimensional magnetotelluric data: An assessment of procedures employed by Brown University. Surveys in Geophysics, 8(2), 187-231.*

*Robertson, K., Heinson, G., & Thiel, S. (2016). Lithospheric reworking at the Proterozoic–Phanerozoic transition of Australia imaged using AusLAMP Magnetotelluric data. Earth and Planetary Science Letters, 452, 27-35.*

*Simpson, F., & Bahr, K. (2005). Practical magnetotellurics. Cambridge University Press.*

*Simpson, J., Brown, D.D., Soeffky, P., Kyi, D., Mann, S., Khrapov, A., Duan, J., and Greenwood, M., (2021). Cloncurry Extension Magnetotelluric Survey. GSQ Technical Notes
2021/03.*

*Wait, J. R. (1954). On the relation between telluric currents and the earth's magnetic field. Geophysics, 19(2), 281-289.*

## Examples

Three examples of inversion of MT data are presented here:
- __Example 01__: inversion of a synthetic 3 layers Earth model using a smooth regularization. 
- __Synthetic AMT data__: inversion of a synthetic 5 layers Earth model using a smooth regularization. 
- __Field AMT/BBMT data__: inversion of field AMT/BBMT data from the Coompana Province, South Australia, using a smooth regularization.


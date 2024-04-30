# Aquifer constant rate discharge pumping test example

<!-- Please write anything you'd like to explain about the forward problem here -->

This Inversion Test Problem demonstrates a range of solutions for the 
time-drawdown response at an observation well resulting from the application of
a constant rate discharge pumping test at an adjacent production well. Each 
solution describes the response for a unique aquifer type. These are:

1. Confined aquifer solution (Theis, 1935)

2. Leaky aquifer solution featuring no aquitard storage (Hantush and Jacob, 1955)

3. Leaky aquifer solution featuring aquitard storage (Hantush, 1960)

4. Fractured rock aquifer solution (Barker, 1988)

These solutions assume full aquifer penetration by both the production and 
observation wells. All solutions are calculated in the Laplace domain and are
inverted to the time domain numerically using the De Hoog et al. (1982) 
algorithm.

import cofi
import cofi_espresso
import numpy as np
import matplotlib.pyplot as plt


xrt = cofi_espresso.XrayTomography()
xrt_problem = cofi.BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))

my_options = cofi.InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

lamdas = np.logspace(-6, 4, 100)
points = np.zeros((len(lamdas), 2))

for i, lamda in enumerate(lamdas):
    reg = cofi.utils.QuadraticReg(lamda, xrt.model_size)
    xrt_problem.set_regularization(1, 2, lamda*reg.matrix)
    m = cofi.Inversion(xrt_problem, my_options).run().model
    points[i,0] = np.linalg.norm(xrt.forward(m)-xrt.data) # ||Gm-d||
    points[i,1] = np.linalg.norm(m) # ||m||
    print(lamda, points[i,:])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(points[:,0], points[:,1])
fig.savefig("test")
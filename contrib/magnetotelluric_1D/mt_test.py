from cofi_espresso import Magnetotelluric1D
import matplotlib.pyplot as plt

problem = Magnetotelluric1D(example_number=1)

model = problem.good_model
data = problem.data
Cm = problem.covariance_matrix

fig_true_model = problem.plot_model(model, title='True model')
fig_data = problem.plot_data(data, Cm=Cm)
plt.show()


# to plot instead the starting model and its response:
starting_model = problem.starting_model
true_resp = problem.forward(model, with_jacobian=False)
starting_model_resp = problem.forward(starting_model)

fig_starting_model = problem.plot_model(starting_model, depths=problem._dpstart, title='Starting model')
fig_data = problem.plot_data(data, starting_model_resp, Cm=Cm)
plt.show()

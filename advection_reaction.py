import numpy as np
from mpi4py import MPI
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square

from ufl import (
	grad,
	dx,
	dS,
	ds,
	dot,
	as_vector,
	TrialFunction,
	TestFunction,
	FacetNormal,
	jump, avg, sqrt,
	conditional, lt
)

"""
Reference:
	Gürkan, Stabilized Cut Discontinuous Galerkin Methods
	for Advection-Reaction Problems, 2020
"""

def plot_2D_spacial(x_coords, y_coords, cells, values):

	fig = plt.figure(figsize = (4, 4))
	ax = fig.add_subplot(111, projection = "3d")
	
	surf = ax.plot_trisurf(x_coords, y_coords, values, triangles = cells, cmap = "bone")
	surf.set_facecolor((0, 0, 1, 0.2))
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal")
	fig.tight_layout()	
	plt.show()

def solve(velocity, source):
	
	domain = create_unit_square(MPI.COMM_WORLD, 50, 50)

	V = functionspace(domain, ("DG", 1))
	# For interpolation
	IV = functionspace(domain, ("CG", 1))
	# For the velocity field
	W = functionspace(domain, ("CG", 1, (2, )))

	f_abs = lambda u: sqrt(u * u) # absolute value function
	nv = FacetNormal(domain) # normal vector

	# velocity field
	theta = Function(W)
	theta.interpolate(velocity)

	# right-side function
	f = Function(V)
	f.interpolate(source)

	u, v = TrialFunction(V), TestFunction(V)

	flux = dot(theta, nv)
	cond_flux = conditional(lt(flux, 0.), flux, 0.)

	a = (u + dot(theta, grad(u)))*v*dx
	a += 0.5*f_abs(dot(nv("+"), theta))*jump(u)*jump(v)*dS
	a -= dot(nv("+"), theta)*jump(u)*avg(v)*dS
	a -= u*cond_flux*v*ds

	L = f*v*dx

	opts = {"ksp_type": "preonly", "pc_type": "lu"}
	problem = LinearProblem(
		a, L,
		petsc_options = opts
	)

	uh = problem.solve()

	uh_ = Function(IV)
	uh_.interpolate(uh)

	print("Interpolated solution:\n", uh_.x.array[:20])

	plot_2D_spacial(
		domain.geometry.x[:, 0],
		domain.geometry.x[:, 1],
		domain.topology.connectivity(2, 0).array.reshape((-1, 3)),
		uh_.x.array
	)

def solve2(velocity, source, inflow):
	
	domain = create_unit_square(MPI.COMM_WORLD, 50, 50)

	V = functionspace(domain, ("DG", 1))
	# For interpolation
	IV = functionspace(domain, ("CG", 1))
	# For the velocity field
	W = functionspace(domain, ("CG", 1, (2, )))

	f_abs = lambda u: sqrt(u * u) # absolute value function
	nv = FacetNormal(domain) # normal vector

	# velocity field
	theta = Function(W)
	theta.interpolate(velocity)

	# right-side function
	f = Function(V)
	f.interpolate(source)

	g = Function(V)
	g.interpolate(inflow)

	u, v = TrialFunction(V), TestFunction(V)

	flux = dot(theta, nv)
	cond_flux = conditional(lt(flux, 0.), flux, 0.)

	a = (u + dot(theta, grad(u)))*v*dx
	a += 0.5*f_abs(dot(nv("+"), theta))*jump(u)*jump(v)*dS
	a -= dot(nv("+"), theta)*jump(u)*avg(v)*dS
	a -= u*cond_flux*v*ds

	L = f*v*dx - g*cond_flux*v*ds

	opts = {"ksp_type": "preonly", "pc_type": "lu"}
	problem = LinearProblem(
		a, L,
		petsc_options = opts
	)

	uh = problem.solve()

	uh_ = Function(IV)
	uh_.interpolate(uh)

	print("Interpolated solution:\n", uh_.x.array[:20])

	plot_2D_spacial(
		domain.geometry.x[:, 0],
		domain.geometry.x[:, 1],
		domain.topology.connectivity(2, 0).array.reshape((-1, 3)),
		uh_.x.array
	)
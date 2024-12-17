import sys
import time

import gmsh
import numpy as np
from numpy.linalg import norm
from mpi4py import MPI
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from dolfinx.io import gmshio
from dolfinx import default_scalar_type
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.fem import (
	form,
	Constant,
	dirichletbc,
	functionspace,
	assemble_scalar,
	locate_dofs_topological
)
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner
)


def plot_u(domain, u):
	x1, x2 = domain.geometry.x[:, 0], domain.geometry.x[:, 1]
	cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3)) 
	fig = ff.create_trisurf(
		x=x1, y=x2, z=u.x.array,
		simplices=cells,
		title='Approximate solution',
		show_colorbar=True, plot_edges=True, colormap='Viridis')
	fig.show()

def create_domain_2d(vertices, size_mesh, markers = [], plot = False):
	gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0)
	
	K = len(vertices)
	gmsh.model.geo.addPoint(vertices[0][0], vertices[0][1], 0., 0.1, tag=1)
	for k in range(1, K):
		gmsh.model.geo.addPoint(vertices[k][0], vertices[k][1], 0., 0.1,tag=k+1)
		gmsh.model.geo.addLine(k, k+1, tag=k)
	gmsh.model.geo.addLine(K, 1, tag = K)
	gmsh.model.geo.addCurveLoop([k for k in range(1, K+1)], tag=K+1)
	tag_surface = 1
	gmsh.model.geo.addPlaneSurface([K+1], tag=tag_surface)
	# line tags : [1, 2, ..., K]
	gmsh.model.geo.synchronize()
	
	tag_domain, tag_boundary = 100*K, 200*K
	
	# add name to the domain
	gmsh.model.addPhysicalGroup(2, [1], tag_domain)
	gmsh.model.setPhysicalName(2, tag_domain, "domain")
	
	# add mark to subsets of the boundary
	for fcs, mkr in markers:
		gmsh.model.addPhysicalGroup(1, fcs, tag=mkr)
	
	gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_mesh)
	gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mesh)
	
	gmsh.model.mesh.generate(2)
	num_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])
	
	if plot : gmsh.fltk.run() # Plot the mesh
	
	domain, cell_tags, facet_tags = \
		gmshio.model_to_mesh(model = gmsh.model,
							comm = MPI.COMM_WORLD,
							rank = 0,
							gdim = 2)
	
	gmsh.clear()
	gmsh.finalize()
	
	return domain, num_triangles, facet_tags

def get_val(formula):
	return assemble_scalar(form(formula))

def const(domain, value):
	return Constant(domain, default_scalar_type(value))

def solve_pde(domain, ft, bot, top, xmax, u0):
	
	def periodic_boundary(x):
		return np.isclose(x[0], xmax)
	
	def periodic_relation(x):
		out_x = np.zeros_like(x)
		out_x[0] = xmax - x[0]
		out_x[1] = x[1]
		out_x[2] = x[2]
		return out_x
	
	V = functionspace(domain, ("CG", 1))
	bot_dofs = locate_dofs_topological(V, 1, ft.indices[ft.values==bot])
	top_dofs = locate_dofs_topological(V, 1, ft.indices[ft.values==top])
	bcs = [dirichletbc(default_scalar_type(u0), bot_dofs, V), \
			dirichletbc(default_scalar_type(0.), top_dofs, V)]
	mpc = MultiPointConstraint(V)
	mpc.create_periodic_constraint_geometrical(V, \
		periodic_boundary, periodic_relation, bcs)
	mpc.finalize()
	u = TrialFunction(V)
	v = TestFunction(V)
	a = inner(grad(u), grad(v))*dx
	L = const(domain, 0.)*v*dx
	petsc_options = {"ksp_type": "preonly", "pc_type": "lu", \
		"pc_factor_mat_solver_type": "mumps"}
	problem = LinearProblem(a, L, mpc, bcs=bcs, petsc_options=petsc_options) 
	uh = problem.solve()
	return uh



n = 100 # n points, n-1 paths, n>1
x = np.linspace(0., 1, n)
y = np.sin(2*x*np.pi) + 2.
top_curve = [[xi, yi] for xi, yi in zip(x, y)]
vts = [[1,0.], [0.,0.]] + top_curve
markers = [([1], 1), ([2], 2), ([i+3 for i in range(n-1)], 3), ([n-1+3], 4)]
domain, num_tri, ft = create_domain_2d(vts, 0.1, markers, plot = True)
uh = solve_pde(domain, ft, 1, 3, 1., 1.)
plot_u(domain, uh)

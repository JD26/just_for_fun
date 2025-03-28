from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, Function, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from dolfinx.mesh import locate_entities, exterior_facet_indices
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, dx, grad, inner
from mpi4py import MPI
import gmsh
import numpy as np
import pyvista
import plotly.figure_factory as ff

def create_domain_2d(vertices, size_mesh, curve = [], plot = False):
	gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0)
	
	K = len(vertices)
	gmsh.model.geo.addPoint(vertices[0][0], vertices[0][1], 0., 0.1, tag=1)
	for k in range(1, K):
		gmsh.model.geo.addPoint(vertices[k][0], vertices[k][1], 0., 0.1, tag=k+1)
		gmsh.model.geo.addLine(k, k+1, tag=k)
	gmsh.model.geo.addLine(K, 1, tag = K)
	gmsh.model.geo.addCurveLoop([k for k in range(1, K+1)], tag=K+1)
	tag_surface = 1
	gmsh.model.geo.addPlaneSurface([K+1], tag=tag_surface)

	L = len(curve)
	if L > 0 :
		gmsh.model.geo.addPoint(curve[0][0], curve[0][1], 0., 0.1, tag=2*K+1)
		for l in range(1, L):
			gmsh.model.geo.addPoint(curve[l][0], curve[l][1], 0., 0.1, tag=2*K+l+1)
			gmsh.model.geo.addLine(2*K+l, 2*K+l+1, tag=2*K+l)
		gmsh.model.geo.addLine(2*K+L, 2*K+1, tag = 2*K+L)
	
	gmsh.model.geo.synchronize()
	
	if L > 0 : 
		gmsh.model.mesh.embed(1, [2*K+l for l in range(1,L+1)], 2, tag_surface)

	tag_domain, tag_boundary = 100*K, 200*K

	# add name to the domain
	gmsh.model.addPhysicalGroup(2, [1], tag_domain)
	gmsh.model.setPhysicalName(2, tag_domain, "domain")
	
	# add name to the boundary
	surfaces = gmsh.model.occ.getEntities(1)
	gmsh.model.addPhysicalGroup(1, [s[1] for s in surfaces], tag_boundary)
	gmsh.model.setPhysicalName(1, tag_boundary, "boundary")

	gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_mesh)
	gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mesh)
	
	gmsh.model.mesh.generate(2)
	num_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])
	
	if plot : gmsh.fltk.run() # Plot the mesh
	
	domain, cell_tags, facet_tags = \
		gmshio.model_to_mesh(model = gmsh.model,
							comm = MPI.COMM_WORLD,
							rank = 0, # For parallel processes
							gdim = 2)
	
	gmsh.clear()
	gmsh.finalize()
	
	return domain, num_triangles

def boundary_all(V, domain, dim):
	domain.topology.create_connectivity(dim-1, dim)
	boundary_facets = exterior_facet_indices(domain.topology)
	bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, dim-1, boundary_facets), V)
	return bc

def bean_curve(t):
	"""
	Implicit equation: 4y^2-4y+4x^2y+2x^2+x^4=3
	""" 
	return np.cos(t), np.sin(t)+ (np.sin(t))**2/2.

def alternative_bean_curve(t):
	r = np.cos(t)**3+np.sin(t)**3
	return r*np.cos(t)/2., r*np.sin(t)/2.

def alternative_bean_curve_equation(x, y):
	"""
	Implicit equation : 2*(x^2 + y^2)^2 = x^3 + y^3 
	"""
	return 2*(x**2 + y**2)**2 - x**3 - y**3

K = 100
size_mesh = 0.1
t = np.linspace(0, 2.*np.pi, K, endpoint=False)
xt, yt = bean_curve(t)
Vertices = [[xt[i], yt[i]] for i in range(K)]
L = 30
t = np.linspace(0, np.pi, L, endpoint=False)
xc, yc = alternative_bean_curve(t)
Curve = [[xc[i], yc[i]] for i in range(L)]
mesh, _ = create_domain_2d(Vertices, size_mesh, Curve, plot = True)

Q = functionspace(mesh, ("DG", 0))

def sub_domain(x):
	return alternative_bean_curve_equation(x[0], x[1]) <= 0.0001

sigma_max = 4
sigma_min = 1
sigma = Function(Q)
cells_0 = locate_entities(mesh, mesh.topology.dim, sub_domain)
sigma.x.array[:] = sigma_max
sigma.x.array[cells_0] = np.full_like(cells_0, sigma_min, dtype=default_scalar_type)

V = functionspace(mesh, ("Lagrange", 1))
u, v = TrialFunction(V), TestFunction(V)
a = inner(sigma * grad(u), grad(v)) * dx
L = Constant(mesh, default_scalar_type(1)) * v * dx

problem = LinearProblem(a, L, bcs=[boundary_all(V, mesh, 2)])
uh = problem.solve()

tdim = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local
marker = sigma_max*np.ones(num_cells_local, dtype=np.int32)
marker[cells_0] = sigma_min
mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

p = pyvista.Plotter(off_screen=True)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Coefficient"] = marker
grid.set_active_scalars("Coefficient")
p.add_mesh(grid, show_edges=True)
p.screenshot("predefined_subdomain_sigma.png")

p2 = pyvista.Plotter(off_screen=True)
grid_uh = pyvista.UnstructuredGrid(*vtk_mesh(V))
grid_uh.point_data["ApproxSolution"] = uh.x.array.real
grid_uh.set_active_scalars("ApproxSolution")
p2.add_mesh(grid_uh, show_edges=True)
p2.screenshot("predefined_subdomain_uh.png")

x1, x2 = mesh.geometry.x[:, 0], mesh.geometry.x[:, 1]
cells = mesh.topology.connectivity(2, 0).array.reshape((-1, 3)) 
fig = ff.create_trisurf(
    x=x1, y=x2, z=uh.x.array,
    simplices=cells,
    title='Approximate solution uh',
    show_colorbar=True, plot_edges=True, colormap='Viridis'
)

fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
fig.show()

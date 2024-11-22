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

def create_domain_2d(K, V, size_mesh, plot = False):
	gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0)
	gmsh.model.geo.addPoint(V[0][0], V[0][1], 0., 0.1, tag=1)
	for k in range(1, K):
		gmsh.model.geo.addPoint(V[k][0], V[k][1], 0., 0.1, tag=k+1)
		gmsh.model.geo.addLine(k, k+1, tag=k)
	gmsh.model.geo.addLine(K, 1, tag = K)
	gmsh.model.geo.addCurveLoop([k for k in range(1, K+1)], tag=K+1)
	gmsh.model.geo.addPlaneSurface([K+1], tag=1)
	gmsh.model.geo.synchronize()
	
	# add name to the polyhedral, tag = 2K
	gmsh.model.addPhysicalGroup(2, [1], 2*K)
	gmsh.model.setPhysicalName(2, 2*K, "polyhedral")
	
	# add name to the boundary
	surfaces = gmsh.model.occ.getEntities(1)
	gmsh.model.addPhysicalGroup(1, [s[1] for s in surfaces], 3*K)
	gmsh.model.setPhysicalName(1, 3*K, "boundary")
	
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

def parametric_curve(t):
     return np.cos(t), np.sin(t)+ (np.sin(t))**2/2.

K = 200
size_mesh = 0.085
t = np.linspace(0, 2.*np.pi, K, endpoint=False)
xt, yt = parametric_curve(t)
vertices = [[xt[i], yt[i]] for i in range(K)]
mesh, _ = create_domain_2d(K, vertices, size_mesh, plot = True)

Q = functionspace(mesh, ("DG", 0))

def sub_domain(x):
    return x[0]*x[0]+x[1]*x[1] <= 0.2

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
p.screenshot("dirichlet_subdomain_coeff.png")

p2 = pyvista.Plotter(off_screen=True)
grid_uh = pyvista.UnstructuredGrid(*vtk_mesh(V))
grid_uh.point_data["ApproxSolution"] = uh.x.array.real
grid_uh.set_active_scalars("ApproxSolution")
p2.add_mesh(grid_uh, show_edges=True)
p2.screenshot("dirichlet_subdomain_approx_sol.png")

x1, x2 = mesh.geometry.x[:, 0], mesh.geometry.x[:, 1]
cells = mesh.topology.connectivity(2, 0).array.reshape((-1, 3)) 
fig = ff.create_trisurf(
    x=x1, y=x2, z=uh.x.array,
    simplices=cells,
    title='Approximate solution',
    show_colorbar=True, plot_edges=True, colormap='Viridis'
)

fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
fig.show()
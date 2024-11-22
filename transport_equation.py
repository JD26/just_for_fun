
from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import numpy as np
import gmsh
from dolfinx.io import gmshio
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

def circle(t):
	# The domain is a ball  
    return np.cos(t), np.sin(t)

K = 200
size_mesh = 0.1
xt, yt = circle(np.linspace(0, 2.*np.pi, K, endpoint=False))
Vertices = [[xt[i], yt[i]] for i in range(K)]
domain, _ = create_domain_2d(Vertices, size_mesh, [], plot = True)


t = 0  # Start time
T = 1.  # End time
num_steps = 40 # Number of time steps
dt = (T - t) / num_steps  # Time step size

# parameters of the solution
alpha = 3.
beta = 2.

V = fem.functionspace(domain, ("Lagrange", 2))

# Defining the exact solution
class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1. + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

u_exact = exact_solution(alpha, beta, t)

u_n = fem.Function(V)
u_n.interpolate(u_exact)

x = ufl.SpatialCoordinate(domain)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
vxy = ufl.as_vector([-x[1], x[0]])
f = 2.*alpha*x[0]*x[1] + beta - 2.*x[0]*x[1]
F = (u + dt * ufl.dot(vxy, ufl.grad(u)))* v * ufl.dx - (u_n + dt * f) * v * ufl.dx
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = assemble_matrix(a)
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for n in range(num_steps):
    u_exact.t += dt
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)
    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array


# Compute L2 error
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute valuerror at vertices
V1 = fem.functionspace(domain, ("Lagrange", 1))
u_ex = fem.Function(V1)
u_ex.interpolate(u_exact)
uh1 = fem.Function(V1)
uh1.interpolate(uh)
error_max = domain.comm.allreduce(np.max(np.abs(uh1.x.array - u_ex.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")

x1, x2 = domain.geometry.x[:, 0], domain.geometry.x[:, 1]
cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3)) 
fig = ff.create_trisurf(
    x=x1, y=x2, z=uh1.x.array,
    simplices=cells,
    title='Approximate solution',
    show_colorbar=True, plot_edges=True, colormap='Viridis'
)

fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
fig.show()

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors
import dolfinx.fem.petsc
import numpy as np
from scifem import create_real_functionspace, assemble_scalar
import ufl
import plotly.figure_factory as ff

M = 100
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M, dolfinx.mesh.CellType.triangle, dtype=np.float64)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

def u_exact(x):
    return ufl.sin(ufl.pi*x[0])+ufl.sin(ufl.pi*x[1])

def sigma(x):
    return ufl.exp(-x[0]**2-x[1]**2)

#0.3 * x[1] ** 2 + ufl.sin(2 * ufl.pi * x[0])

x = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)
f = sigma(x)*ufl.dot(ufl.grad(u_exact(x)), n)
S = -ufl.div(sigma(x)*ufl.grad(u_exact(x)))
h = assemble_scalar(u_exact(x) * ufl.ds)

R = create_real_functionspace(domain)

W = ufl.MixedFunctionSpace(V, R)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

zero = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0))

a = dolfinx.fem.form([[sigma(x)*ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(p, v) * ufl.ds], \
                      [ufl.inner(u, q) * ufl.ds, None]])
L = dolfinx.fem.form([ufl.inner(S, v) * ufl.dx + ufl.inner(f, v) * ufl.ds, ufl.inner(zero, q) * ufl.ds])

A = dolfinx.fem.petsc.assemble_matrix_block(a)
A.assemble()
b = dolfinx.fem.petsc.assemble_vector_block(L, a, bcs=[])

maps = [(Wi.dofmap.index_map, Wi.dofmap.index_map_bs) for Wi in W.ufl_sub_spaces()]

b_local = get_local_vectors(b, maps)
b_local[1][:] = h
scatter_local_vectors(
        b,
        b_local,
        maps,
    )
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
xh = dolfinx.fem.petsc.create_vector_block(L)
ksp.solve(b, xh)
xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

uh = dolfinx.fem.Function(V, name="u")
x_local = get_local_vectors(xh, maps)
uh.x.array[: len(x_local[0])] = x_local[0]
uh.x.scatter_forward()
diff = uh - u_exact(x)
error = dolfinx.fem.form(ufl.inner(diff, diff) * ufl.dx)
print(f"L2 error: {np.sqrt(assemble_scalar(error)):.2e}")

x1, x2 = domain.geometry.x[:, 0], domain.geometry.x[:, 1]
cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3)) 
fig = ff.create_trisurf(
    x=x1, y=x2, z=uh.x.array,
    simplices=cells,
    title='Approximate solution',
    show_colorbar=True, plot_edges=True, colormap='Viridis'
)

fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
fig.show()
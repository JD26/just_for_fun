
import gmsh
import numpy as np
from mpi4py import MPI
import matplotlib.tri as tri
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
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner
)

from abc import ABC, abstractmethod

def plot_u(domain, u):
	x1, x2 = domain.geometry.x[:, 0], domain.geometry.x[:, 1]
	cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3)) 
	fig = ff.create_trisurf(
		x=x1, y=x2, z=u.x.array,
		simplices=cells,
		title='Approximate solution',
		show_colorbar=True, plot_edges=True, colormap='Viridis')
	fig.show()

def plot_uV2(domain, u):
    # Extraer las coordenadas de los nodos
    x1, x2 = domain.geometry.x[:, 0], domain.geometry.x[:, 1]
    
    # Extraer las conectividades de las celdas (triángulos)
    cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3))
    
    # Crear el objeto de triangulación
    triang = tri.Triangulation(x1, x2, cells)
    
    # Extraer los valores de la solución
    z = u.x.array
    
    # Crear la figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Graficar el campo de la solución con sombreado
    tpc = ax.tripcolor(triang, z, shading='flat', cmap='viridis')
    
    # Agregar una barra de color
    cbar = fig.colorbar(tpc, ax=ax, orientation='vertical')
    cbar.set_label('Solution value')
    
    # Configurar el título y los ejes
    ax.set_title('Approximate solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    # Mostrar la gráfica
    plt.show()

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

class Linear():
	def __init__(self, x, h):
		self.x = x
		self._h = h
	
	def _basis_func(self, xi):
		h = self._h
		def bf(x):
			return np.maximum(1 - np.abs(x - xi)/h, 0)
		return bf

	def basis(self):
		return [self._basis_func(xi) for xi in self.x]


class BasisFunction(ABC):
	def __init__(self, x, sigma):
		self.x = x
		self._sigma = sigma

	@abstractmethod
	def _basis_func(self, xi):
		"""Abstract method for the basis function."""
		pass
	
	@abstractmethod
	def _der_basis_func(self, xi):
		"""Abstract method for the derivative of the basis function."""
		pass

	def basis(self):
		"""Generates all basis functions."""
		return [self._basis_func(xi) for xi in self.x]

	def dbasis(self):
		"""Generates derivatives of all basis functions."""
		return [self._der_basis_func(xi) for xi in self.x]

class Wendland(BasisFunction):
	def __init__(self, x, sigma):
		super().__init__(x, sigma)

	def _basis_func(self, xi):
		sigma = self._sigma
		def bf(x):
			r = np.abs(x - xi)/sigma
			return (1. + 4.*r)*np.maximum(1 - r, 0)**4
		return bf

	def _der_basis_func(self, xi):
		sigma = self._sigma
		def dbf(x):
			r = np.abs(x - xi)/sigma
			return -20.*r*np.maximum(1 - r, 0)**3/sigma*np.sign(x - xi)
		return dbf

class Exponencial(BasisFunction):
	def __init__(self, x, sigma):
		super().__init__(x, sigma)

	def _basis_func(self, xi):
		sigma = self._sigma
		def bf(x):
			return np.exp(-((x - xi)/sigma)**2)
		return bf

	def _der_basis_func(self, xi):
		sigma = self._sigma
		def dbf(x):
			return 2.*(xi - x)*np.exp(-((x - xi)/sigma)**2)/sigma**2
		return dbf
	
class Interpolator:
	"""
	This class build a interpolating periodic
	function. The get_interpolator method
	returns the interpolating function.
	basis : all basis functions
	alpha, beta : boundary coefficients
	"""
	def __init__(self, basis, alpha, beta):
		self.ca = alpha
		self.cb = beta
		self.b = basis
	
	def get_interpolator(self, c):
		def interpolator(x):
			mtx = [p * self.b[0](x) + v(x) + q * self.b[-1](x)
					for p, v, q in zip(self.ca, self.b[1:-1], self.cb)]
			return np.dot(c, np.array(mtx))
		return interpolator

def coef_boundary_basis(b, db, xini, xend):
	"""
	b : basis functions
	db : derivatives of basis functions
	xini, xend : boundary nodes
	This function returns coefficients
	alpha and beta such that
	c[0] = np.inner(alpha, c[1:-1])
	c[-1] = np.inner(beta, c[1:-1])
	"""
	bx = np.array([bi(xend)-bi(xini) for bi in b])
	dbx = np.array([dbi(xend)-dbi(xini) for dbi in db])
	dtm = bx[0]*dbx[-1]-bx[-1]*dbx[0]
	alpha = (bx[-1]*dbx[1:-1] - dbx[-1]*bx[1:-1])/dtm 
	beta = (dbx[0]*bx[1:-1] - bx[0]*dbx[1:-1])/dtm
	return alpha, beta

def coef_periodic_basis(xin, yin, b, p, q):
	"""
	xin, yin : interior points
	b : all basis functions
	p, q: boundary coefficients
	This function returns coefficients
	for the peridic interpolation function.
	"""
	m = xin.shape[0]
	A = np.zeros((m, m))
	for i in range(m):
		for j in range(m):
			A[i, j] = p[j]*b[0](xin[i])+b[1+j](xin[i])+q[j]*b[-1](xin[i])
	return np.linalg.solve(A, yin)

def test():
	f = lambda x: 0.1*np.sin(2*x*np.pi) + 1.
	m = 6
	x = np.linspace(0., 1., m) # control points
	y = f(x) # interpolating values
	h = (1.-0.)/(m-1.) # step size

	"""
	# linear interpolation
	li = Linear(x, h)
	b = li.basis()
	cab = np.zeros(len(x)-2)
	cab[0], cab[-1] = .5, .5
	ca, cb = cab.copy(), cab.copy()
	c = y[1:-1]
	"""

	#"""
	# Radial interpolation
	ra = Wendland(x, 10.*h)
	#ra = Exponencial(x, 3.*h)
	b = ra.basis()
	db = ra.dbasis()
	ca, cb = coef_boundary_basis(b, db, x[0], x[-1])
	c = coef_periodic_basis(x[1:-1], y[1:-1], b, ca, cb)
	print("> alpha coefficients:", ca)
	print("> beta coefficients:", cb)
	print("> periodic coefficients:", c)

	# Test for check periodicity
	cini = np.inner(ca, c) 
	cend = np.inner(cb, c)
	bini = np.array([bi(x[0]) for bi in b])
	bend = np.array([bi(x[-1]) for bi in b])
	dbini = np.array([dbi(x[0]) for dbi in db])
	dbend = np.array([dbi(x[-1]) for dbi in db])
	ct = np.concatenate(([cini], c, [cend]))
	print("> Values at the boundary:")
	print(np.inner(ct, bini), np.inner(ct, bend))
	print(np.inner(ct, dbini), np.inner(ct, dbend))
	#"""

	Ip = Interpolator(b, ca, cb)
	fb = Ip.get_interpolator(c)

	ni = 6 # points in each subinterval
	n = m+(m-1)*ni # mesh top boundary points
	xs = np.linspace(0., 1., n)
	ys = fb(xs)

	xx = np.linspace(0, 1, 100)
	plt.plot(xx, f(xx), "-k") # original function
	plt.plot(xs, ys, "-ro") # mesh points
	plt.plot(x[1:-1], y[1:-1], "bo") # interpolating points
	plt.show()

	top_curve = [[xi, yi] for xi, yi in zip(xs, ys)]
	vts = [[1,0.], [0.,0.]] + top_curve
	markers = [([1], 1), ([2], 2), ([i+3 for i in range(n-1)], 3), ([n-1+3], 4)]
	domain, num_tri, ft = create_domain_2d(vts, 0.1, markers, plot = True)
	uh = solve_pde(domain, ft, 1, 3, 1., 1.)
	plot_uV2(domain, uh)

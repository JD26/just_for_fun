from dolfinx.fem import functionspace, assemble_scalar, form
from dolfinx.io import gmshio
from ufl import ds, SpatialCoordinate, Measure
from mpi4py import MPI
import gmsh
import numpy as np
import plotly.figure_factory as ff

def create_domain_2d(vertices, size_mesh, curve = [], markers = [], plot = False):
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
    # line tags : [1, 2, ..., K]

	L = len(curve)
	if L > 0 :
		gmsh.model.geo.addPoint(curve[0][0], curve[0][1], 0., 0.1, tag=2*K+1)
		for l in range(1, L):
			gmsh.model.geo.addPoint(curve[l][0], curve[l][1], 0., 0.1, tag=2*K+l+1)
			gmsh.model.geo.addLine(2*K+l, 2*K+l+1, tag=2*K+l)
		gmsh.model.geo.addLine(2*K+L, 2*K+1, tag = 2*K+L)
	
	gmsh.model.geo.synchronize()
	
	if L > 0 :
		# add curve to the surface
		gmsh.model.mesh.embed(1, [2*K+l for l in range(1,L+1)], 2, tag_surface)

	tag_domain, tag_boundary = 100*K, 200*K

	# add name to the domain
	gmsh.model.addPhysicalGroup(2, [1], tag_domain)
	gmsh.model.setPhysicalName(2, tag_domain, "domain")
	
    # mark the boundary
	if len(markers)>0:
		# add mark to subsets of the boundary
		for fcs, mkr in markers:
			gmsh.model.addPhysicalGroup(1, fcs, tag=mkr)
	else :
		# add mark to the boundary
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
							rank = 0,
							gdim = 2)
	
	gmsh.clear()
	gmsh.finalize()
	
	return domain, num_triangles, facet_tags

def bean_curve(t):
	"""
	Implicit equation: 4y^2-4y+4x^2y+2x^2+x^4=3
	""" 
	return np.cos(t), np.sin(t)+ (np.sin(t))**2/2.

K = 200
size_mesh = 0.1
t = np.linspace(0, 2.*np.pi, K, endpoint=False)
xt, yt = bean_curve(t)
vertices = [[xt[i], yt[i]] for i in range(K)]
L = 30
t = np.linspace(0, np.pi, L, endpoint=False)
curve = []
# 200 <-> 2 pi
# 25 <->  1/4 pi
# 100 <-> pi
# [101, 125] <-> [pi, pi+pi/4]
markers = [([i for i in range(1, 25+1)], 1), ([i for i in range(101, 125 + 1)], 2)]
domain, _, facet_mrks = create_domain_2d(vertices, size_mesh, curve, markers, plot = True)

ds_1 = ds(domain=domain, subdomain_data=facet_mrks, subdomain_id=1)
ds_2 = ds(domain=domain, subdomain_data=facet_mrks, subdomain_id=2)

ds = Measure("ds", domain=domain, subdomain_data=facet_mrks)

V = functionspace(domain, ("CG", 1))

# Definir función f(x, y)
x = SpatialCoordinate(domain)
f = x[0]+x[1]
print(assemble_scalar(form(f*ds_1)), assemble_scalar(form(f*ds(1))))
f = 2*x[0]**2 + x[0]*x[1]
print(assemble_scalar(form(f*ds_2)), assemble_scalar(form(f*ds(2))))
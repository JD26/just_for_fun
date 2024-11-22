# Just for fun
## dirichlet_subdomain
We solve the Dirichlet problem$$\nabla \cdot (\sigma \nabla u)  = 1 \quad x\in\Omega,\quad u = 0 \quad x\in\partial\Omega$$in the two-dimensional domain $\Omega$ determined by the parametric curve$$t \mapsto (\cos t, \sin t+ \frac{\sin^2 t}{2})$$for $t\in[0,2\pi)$. We use `Gmsh` to generate the mesh. The coefficient $\sigma$ takes the value $\sigma=1$ on the subdomain $\Omega_0 \subset \Omega$, and $\sigma = 4$ on $\Omega \setminus \Omega_0$. The subdomain $\Omega_0$ is a ball of radius $\sqrt{0.2}$ and center $(0,0)$. 
## predefined_subdomain
We solve the same problem that in `dirichlet_subdomain` but with a subdomain $\Omega_0$ defined when $\Omega$ is generated. More precisely, the implicit equation$$2(x^2 + y^2)^2 = x^3 + y^3$$defines the boundary of the subdomain $\Omega_0$. This approach yields a more exact approximate solution.
## transport_equation
We solve the transport equation$$\phi_t (x,y,t)+\theta(x,y)\cdot\nabla\phi(x,y,t)=f(x,y)$$for $t\in[0,T]$ and $x\in\Omega$, with $\Omega$ being a two-dimensional ball. The vector field $\theta$ is defined by $\theta(x,y)=(-y,x)$. Thus, $\theta \cdot n =0$ on $\partial\Omega$. The _Backward Euler_ method is used to approximate the partial derivative with respect to $t$. The function$$\phi(x,y,t)=1 + x^2 + \alpha y^2 + \beta t$$is used to verify the numerical method.
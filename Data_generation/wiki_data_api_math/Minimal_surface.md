# Minimal surface

In mathematics, a minimal surface is a surface that locally minimizes its area. This is equivalent to having zero mean curvature (see definitions below).

The term "minimal surface" is used because these surfaces originally arose as surfaces that minimized total surface area subject to some constraint. Physical models of area-minimizing minimal surfaces can be made by dipping a wire frame into a soap solution, forming a soap film, which is a minimal surface whose boundary is the wire frame. However, the term is used for more general surfaces that may self-intersect or do not have constraints. For a given constraint there may also exist several minimal surfaces with different areas (for example, see minimal surface of revolution): the standard definitions only relate to a local optimum, not a global optimum.

Minimal surfaces can be defined in several equivalent ways in

. The fact that they are equivalent serves to demonstrate how minimal surface theory lies at the crossroads of several mathematical disciplines, especially differential geometry, calculus of variations, potential theory, complex analysis and mathematical physics.

is minimal if and only if every point p ∈ M has a neighbourhood, bounded by a simple closed curve, which has the least area among all surfaces having the same boundary.

This property is local: there might exist regions in a minimal surface, together with other surfaces of smaller area which have the same boundary. This property establishes a connection with soap films; a soap film deformed to have a wire frame as boundary will minimize area.

is minimal if and only if it is a critical point of the area functional for all compactly supported variations.

This definition makes minimal surfaces a 2-dimensional analogue to geodesics, which are analogously defined as critical points of the length functional.

is minimal if and only if its mean curvature is equal to zero at all points.

A direct implication of this definition is that every point on the surface is a saddle point with equal and opposite principal curvatures. Additionally, this makes minimal surfaces into the static solutions of mean curvature flow. By the Young–Laplace equation, the mean curvature of a soap film is proportional to the difference in pressure between the sides. If the soap film does not enclose a region, then this will make its mean curvature zero. By contrast, a spherical soap bubble encloses a region which has a different pressure from the exterior region, and as such does not have zero mean curvature.

is a real valued function, is minimal if and only if

{\displaystyle (1+u_{x}^{2})u_{yy}-2u_{x}u_{y}u_{xy}+(1+u_{y}^{2})u_{xx}=0}

The partial differential equation in this definition was originally found in 1762 by Lagrange, and Jean Baptiste Meusnier discovered in 1776 that it implied a vanishing mean curvature. This equation gives an asymmetric definition in the sense that the position on the

. Not all surfaces are conveniently represented this way. An alternative definition based on the more general representation

{\displaystyle \mathbf {x} :\mathbb {R} ^{2}\to \mathbb {R} ^{3},(u,v)\mapsto (x,y,z)}

{\displaystyle {\frac {\partial }{\partial u}}{\frac {{\frac {\partial \mathbf {x} }{\partial v}}{\boldsymbol {\times }}({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}})}{\sqrt {({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}}){\boldsymbol {\cdot }}({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}})}}}={\frac {\partial }{\partial v}}{\frac {{\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}})}{\sqrt {({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}}){\boldsymbol {\cdot }}({\frac {\partial \mathbf {x} }{\partial u}}{\boldsymbol {\times }}{\frac {\partial \mathbf {x} }{\partial v}})}}}}

is minimal if and only if it is a critical point of the Dirichlet energy for all compactly supported variations, or equivalently if any point

has a neighbourhood with least energy relative to its boundary.

This definition ties minimal surfaces to harmonic functions and potential theory.

{\displaystyle X=(x_{1},x_{2},x_{3}):M\rightarrow \mathbb {R} ^{3}}

is an isometric immersion of a Riemann surface into 3-space, then

A direct implication of this definition and the maximum principle for harmonic functions is that there are no compact complete minimal surfaces in

is minimal if and only if its stereographically projected Gauss map

{\displaystyle g:M\rightarrow \mathbb {C} \cup {\infty }}

is meromorphic with respect to the underlying Riemann surface structure, and

This definition uses that the mean curvature is half of the trace of the shape operator, which is linked to the derivatives of the Gauss map. If the projected Gauss map obeys the Cauchy–Riemann equations then either the trace vanishes or every point of M is umbilic, in which case it is a piece of a sphere.

The local least area and variational definitions allow extending minimal surfaces to other Riemannian manifolds than

Minimal surface theory originates with Lagrange who in 1762 considered the variational problem of finding the surface

of least area stretched across a given closed contour. He derived the Euler–Lagrange equation for the solution

{\displaystyle {\frac {d}{dx}}\left({\frac {z_{x}}{\sqrt {1+z_{x}^{2}+z_{y}^{2}}}}\right)+{\frac {d}{dy}}\left({\frac {z_{y}}{\sqrt {1+z_{x}^{2}+z_{y}^{2}}}}\right)=0}

He did not succeed in finding any solution beyond the plane. In 1776 Jean Baptiste Marie Meusnier discovered that the helicoid and catenoid satisfy the equation and that the differential expression corresponds to twice the mean curvature of the surface, concluding that surfaces with zero mean curvature are area-minimizing.

{\displaystyle \left(1+z_{x}^{2}\right)z_{yy}-2z_{x}z_{y}z_{xy}+\left(1+z_{y}^{2}\right)z_{xx}=0}

Gaspard Monge and Legendre in 1795 derived representation formulas for the solution surfaces. While these were successfully used by Heinrich Scherk in 1830 to derive his surfaces, they were generally regarded as practically unusable. Catalan proved in 1842/43 that the helicoid is the only ruled minimal surface.

Progress had been fairly slow until the middle of the century when the Björling problem was solved using complex methods. The "first golden age" of minimal surfaces began. Schwarz found the solution of the Plateau problem for a regular quadrilateral in 1865 and for a general quadrilateral in 1867 (allowing the construction of his periodic surface families) using complex methods. Weierstrass and Enneper developed more useful representation formulas, firmly linking minimal surfaces to complex analysis and harmonic functions. Other important contributions came from Beltrami, Bonnet, Darboux, Lie, Riemann, Serret and Weingarten.

Between 1925 and 1950 minimal surface theory revived, now mainly aimed at nonparametric minimal surfaces. The complete solution of the Plateau problem by Jesse Douglas and Tibor Radó was a major milestone. Bernstein's problem and Robert Osserman's work on complete minimal surfaces of finite total curvature were also important.

Another revival began in the 1980s. One cause was the discovery in 1982 by Celso Costa of a surface that disproved the conjecture that the plane, the catenoid, and the helicoid are the only complete embedded minimal surfaces in

of finite topological type. This not only stimulated new work on using the old parametric methods, but also demonstrated the importance of computer graphics to visualise the studied surfaces and numerical methods to solve the "period problem" (when using the conjugate surface method to determine surface patches that can be assembled into a larger symmetric surface, certain parameters need to be numerically matched to produce an embedded surface). Another cause was the verification by H. Karcher that the triply periodic minimal surfaces originally described empirically by Alan Schoen in 1970 actually exist. This has led to a rich menagerie of surface families and methods of deriving new surfaces from old, for example by adding handles or distorting them.

Currently the theory of minimal surfaces has diversified to minimal submanifolds in other ambient geometries, becoming relevant to mathematical physics (e.g. the positive mass conjecture, the Penrose conjecture) and three-manifold geometry (e.g. the Smith conjecture, the Poincaré conjecture, the Thurston Geometrization Conjecture).

catenoids: minimal surfaces made by rotating a catenary once around its directrix

helicoids: A surface swept out by a line rotating with uniform velocity around an axis perpendicular to the line and simultaneously moving along the axis with uniform velocity

Schwarz minimal surfaces: triply periodic surfaces that fill

Riemann's minimal surface: A posthumously described periodic surface

the Henneberg surface: the first non-orientable minimal surface

the Gyroid: One of Schoen's surfaces from 1970, a triply periodic surface of particular interest for liquid crystal structure

the Saddle tower family: generalisations of Scherk's second surface

Costa's minimal surface: Famous conjecture disproof. Described in 1982 by Celso Costa and later visualized by Jim Hoffman. Jim Hoffman, David Hoffman and William Meeks III then extended the definition to produce a family of surfaces with different rotational symmetries.

the Chen–Gackstatter surface family, adding handles to the Enneper surface.

Generalisations and links to other areas of mathematics

Minimal surfaces can be defined in other manifolds than

, such as hyperbolic space, higher-dimensional spaces or Riemannian manifolds.

The definition of minimal surfaces can be generalized/extended to cover constant-mean-curvature surfaces: surfaces with a constant mean curvature, which need not equal zero.

In discrete differential geometry discrete minimal surfaces are studied: simplicial complexes of triangles that minimize their area under small perturbations of their vertex positions. Such discretizations are often used to approximate minimal surfaces numerically, even if no closed form expressions are known.

Brownian motion on a minimal surface leads to probabilistic proofs of several theorems on minimal surfaces.

The curvature lines of an isothermal surface form an isothermal net.

Minimal surfaces have become an area of intense scientific study, especially in the areas of molecular engineering and materials science, due to their anticipated applications in self-assembly of complex materials. The endoplasmic reticulum, an important structure in cell biology, is proposed to be under evolutionary pressure to conform to a nontrivial minimal surface.

In the fields of general relativity and Lorentzian geometry, certain extensions and modifications of the notion of minimal surface, known as apparent horizons, are significant. In contrast to the event horizon, they represent a curvature-based approach to understanding black hole boundaries.

Structures with minimal surfaces can be used as tents.

Minimal surfaces are part of the generative design toolbox used by modern designers. In architecture there has been much interest in tensile structures, which are closely related to minimal surfaces. Notable examples can be seen in the work of Frei Otto, Shigeru Ban, and Zaha Hadid. The design of the Munich Olympic Stadium by Frei Otto was inspired by soap surfaces. Another notable example, also by Frei Otto, is the German Pavilion at Expo 67 in Montreal, Canada.

In the art world, minimal surfaces have been extensively explored in the sculpture of Robert Engman (1927–2018), Robert Longhurst (1949– ), and Charles O. Perry (1929–2011), among others.
# Geometric measure theory

In mathematics, geometric measure theory (GMT) is the study of geometric properties of sets (typically in Euclidean space) through measure theory. It allows mathematicians to extend tools from differential geometry to a much larger class of surfaces that are not necessarily smooth.

Geometric measure theory was born out of the desire to solve Plateau's problem (named after Joseph Plateau) which asks if for every smooth closed curve in

there exists a surface of least area among all surfaces whose boundary equals the given curve. Such surfaces mimic soap films.

The problem had remained open since it was posed in 1760 by Lagrange. It was solved independently in the 1930s by Jesse Douglas and Tibor Radó under certain topological restrictions. In 1960 Herbert Federer and Wendell Fleming used the theory of currents with which they were able to solve the orientable Plateau's problem analytically without topological restrictions, thus sparking geometric measure theory. Later Jean Taylor after Fred Almgren proved Plateau's laws for the kind of singularities that can occur in these more general soap films and soap bubbles clusters.

The following objects are central in geometric measure theory:

Rectifiable sets (or Radon measures), which are sets with the least possible regularity required to admit approximate tangent spaces.

Characterization of rectifiability through existence of approximate tangents, densities, projections, etc.

Orthogonal projections, Kakeya sets, Besicovitch sets

Rectifiability and uniform rectifiability of (subsets of) metric spaces, e.g. SubRiemannian manifolds, Carnot groups, Heisenberg groups, etc.

Connections to singular integrals, Fourier transform, Frostman measures, harmonic measures, etc

Currents, a generalization of the concept of oriented manifolds, possibly with boundary.

Flat chains, an alternative generalization of the concept of manifolds, possibly with boundary.

Caccioppoli sets (also known as sets of locally finite perimeter), a generalization of the concept of manifolds on which the divergence theorem applies.

Plateau type minimization problems from calculus of variations

The following theorems and concepts are also central:

The area formula, which generalizes the concept of change of variables in integration.

The coarea formula, which generalizes and adapts Fubini's theorem to geometric measure theory.

The isoperimetric inequality, which states that the smallest possible circumference for a given area is that of a round circle.

Flat convergence, which generalizes the concept of manifold convergence.

The Brunn–Minkowski inequality for the n-dimensional volumes of convex bodies K and L,

{\displaystyle \mathrm {vol} {\big (}(1-\lambda )K+\lambda L{\big )}^{1/n}\geq (1-\lambda )\mathrm {vol} (K)^{1/n}+\lambda \,\mathrm {vol} (L)^{1/n},}

can be proved on a single page and quickly yields the classical isoperimetric inequality. The Brunn–Minkowski inequality also leads to Anderson's theorem in statistics.  The proof of the Brunn–Minkowski inequality predates modern measure theory; the development of measure theory and Lebesgue integration allowed connections to be made between geometry and analysis, to the extent that in an integral form of the Brunn–Minkowski inequality known as the Prékopa–Leindler inequality the geometry seems almost entirely absent.

Consider the Gauss map, which sends each point on a smooth surface to the unit vector orthogonal to its local tangent plane. This can be generalized. Let

containing the origin, and enclosing a convex region, then each point on

to be the set of orthogonal directions to all local supporting planes to

{\displaystyle G_{K}:\mathbb {S} ^{n-1}\to \mathbb {S} ^{n-1}}

for any Borel set. This is the Gaussian curvature measure associated with

This then induces the inverse problem: Given a Borel measure on the sphere, is it a Gaussian curvature measure? Alexandrov showed the following:

{\displaystyle {\begin{cases}v\left(\mathbb {S} ^{n-1}\right)={\mathcal {H}}^{n-1}\left(\mathbb {S} ^{n-1}\right)\\v\left(\mathbb {S} ^{n-1}\backslash \omega \right)>{\mathcal {H}}^{n-1}\left(\omega ^{*}\right),\;\;\forall \omega \subseteq \mathbb {S} ^{n-1}{\text{ compact and convex. }}\end{cases}}}

as the weak limit of a sequence of point-mass measures. Each is the Gaussian curvature measure of a polytope, and the sequence of polytopes converge to

has a bounded probability density function (in the sense of a Radon–Nikodym derivative):

{\displaystyle {\frac {d\mu _{K}}{d\mu }}(y)=\rho (y),\quad \rho :\mathbb {S} ^{n-1}\to [0,c]}

is continuously differentiable (that is, its tangent planes are unique and varies continuously over its surface).

. This has an intuitive picture in as follows: Consider an open set

{\displaystyle {\frac {1}{4}}(\delta _{(a,b)/{\sqrt {a^{2}+b^{2}}}}+\delta _{(-a,b)/{\sqrt {a^{2}+b^{2}}}}+\delta _{(a,-b)/{\sqrt {a^{2}+b^{2}}}}+\delta _{(-a,-b)/{\sqrt {a^{2}+b^{2}}}})}

{\displaystyle 2b\delta _{(1,0)}+2b\delta _{(-1,0)}+2a\delta _{(0,1)}+2a\delta _{(0,-1)}}

{\displaystyle \mu _{u}(E):=\mu (\cap _{x\in \Omega }\partial u(x))}

. This has applications in the study of weak solutions to the Monge–Ampère equation. It is a nonnegative, locally finite, Borel measure.
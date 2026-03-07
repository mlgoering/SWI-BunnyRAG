# Harmonic measure

In mathematics, especially potential theory, harmonic measure is a concept related to the theory of harmonic functions that arises from the solution of the classical Dirichlet problem.  In probability theory, the harmonic measure of a subset of the boundary of a bounded domain in Euclidean space

is the probability that a Brownian motion started inside a domain hits that subset of the boundary. More generally, harmonic measure of an Itō diffusion X describes the distribution of X as it hits the boundary of D. In the complex plane, harmonic measure can be used to estimate the modulus of an analytic function inside a domain D given bounds on the modulus on the boundary of the domain; a special case of this principle is Hadamard's three-circle theorem. On simply connected planar domains, there is a close connection between harmonic measure and  the theory of conformal maps.

The term harmonic measure was introduced by Rolf Nevanlinna in 1928 for planar domains, although Nevanlinna notes the idea appeared implicitly in earlier work by Johansson, F. Riesz, M. Riesz, Carleman, Ostrowski and Julia (original order cited). The connection between harmonic measure and Brownian motion was first identified by Kakutani ten years later in 1944.

Let D be a bounded, open domain in n-dimensional Euclidean space Rn, n ≥ 2, and let ∂D denote the boundary of D.  Any continuous function f : ∂D → R determines a unique harmonic function Hf that solves the Dirichlet problem

{\displaystyle {\begin{cases}-\Delta H_{f}(x)=0,&x\in D;\\H_{f}(x)=f(x),&x\in \partial D.\end{cases}}}

If a point x ∈ D is fixed, by the Riesz–Markov–Kakutani representation theorem and the maximum principle Hf(x) determines a probability measure ω(x, D) on ∂D by

{\displaystyle H_{f}(x)=\int _{\partial D}f(y)\,\mathrm {d} \omega (x,D)(y).}

The measure ω(x, D) is called the harmonic measure (of the domain D with pole at x).

For any Borel subset E of ∂D, the harmonic measure ω(x, D)(E) is equal to the value at x of the solution to the Dirichlet problem with boundary data equal to the indicator function of E.

For fixed D and E ⊆ ∂D, ω(x, D)(E) is a harmonic function of x ∈ D and

{\displaystyle 1-\omega (x,D)(E)=\omega (x,D)(\partial D\setminus E);}

Hence, for each x and D, ω(x, D) is a probability measure on ∂D.

If ω(x, D)(E) = 0 at even a single point x of D, then

is identically zero, in which case E is said to be a set of harmonic measure zero. This is a consequence of Harnack's inequality.

Since explicit formulas for harmonic measure are not typically available, we are interested in determining conditions which guarantee a set has harmonic measure zero.

is a simply connected planar domain bounded by a rectifiable curve (i.e. if

), then harmonic measure is mutually absolutely continuous with respect to arc length: for all

. Moreover, harmonic measure on D is mutually singular with respect to t-dimensional Hausdorff measure for all t > 1.

is a bounded Lipschitz domain, then harmonic measure and (n − 1)-dimensional Hausdorff measure are mutually absolutely continuous: for all

{\displaystyle \mathbb {D} =\{X\in \mathbb {R} ^{2}:|X|<1\}}

with pole at the origin is length measure on the unit circle normalized to be a probability, i.e.

{\displaystyle \omega (0,\mathbb {D} )(E)=|E|/2\pi }

{\displaystyle \omega (X,\mathbb {D} )(E)=\int _{E}{\frac {1-|X|^{2}}{|X-Q|^{2}}}{\frac {dH^{1}(Q)}{2\pi }}}

denotes length measure on the unit circle. The Radon–Nikodym derivative

{\displaystyle \mathbb {B} ^{n}=\{X\in \mathbb {R} ^{n}:|X|<1\}}

is the n-dimensional unit ball, then harmonic measure with pole at

{\displaystyle \omega (X,\mathbb {B} ^{n})(E)=\int _{E}{\frac {1-|X|^{2}}{|X-Q|^{n}}}{\frac {dH^{n-1}(Q)}{\sigma _{n-1}}}}

denotes surface measure ((n − 1)-dimensional Hausdorff measure) on the unit sphere

is a simply connected planar domain bounded by a Jordan curve and X

is the unique Riemann map which sends the origin to X, i.e.

is the domain bounded by the Koch snowflake, then there exists a subset

Consider an Rn-valued Itō diffusion X starting at some point x in the interior of a domain D, with law Px.  Suppose that one wishes to know the distribution of the points at which X exits D.  For example, canonical Brownian motion B on the real line starting at 0 exits the interval (−1, +1) at −1 with probability ⁠1/2⁠ and at +1 with probability ⁠1/2⁠, so Bτ(−1, +1) is uniformly distributed on the set {−1, +1}.

In general, if G is compactly embedded within Rn, then the harmonic measure (or hitting distribution) of X on the boundary ∂G of G is the measure μGx defined by

{\displaystyle \mu _{G}^{x}(F)=\mathbf {P} ^{x}{\big [}X_{\tau _{G}}\in F{\big ]}}

Returning to the earlier example of Brownian motion, one can show that if B is a Brownian motion in Rn starting at x ∈ Rn and D ⊂ Rn is an open ball centred on x, then the harmonic measure of B on ∂D is invariant under all rotations of D about x and coincides with the normalized surface measure on ∂D

Garnett, John B.; Marshall, Donald E. (2005). Harmonic Measure. Cambridge: Cambridge University Press. ISBN 978-0-521-47018-6.

Øksendal, Bernt K. (2003). Stochastic Differential Equations: An Introduction with Applications (Sixth ed.). Berlin: Springer. ISBN 3-540-04758-1. MR 2001996 (See Sections 7, 8 and 9)

Capogna, Luca; Kenig, Carlos E.; Lanzani, Loredana (2005). Harmonic Measure: Geometric and Analytic Points of View. University Lecture Series. Vol. ULECT/35. American Mathematical Society. p. 155. ISBN 978-0-8218-2728-4.
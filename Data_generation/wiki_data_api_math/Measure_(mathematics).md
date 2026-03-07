# Measure (mathematics)

In mathematics, the concept of a measure is a generalization and formalization of geometrical measures (length, area, volume) and other common notions, such as magnitude, mass, and probability of events. These seemingly distinct concepts have many similarities and can often be treated together in a single mathematical context. Measures are foundational in probability theory, integration theory, and can be generalized to assume negative values, as with electrical charge. Far-reaching generalizations (such as spectral measures and  positive operator-valued measures) of measure are widely used in quantum physics and physics in general.

The intuition behind this concept dates back to Ancient Greece, when Archimedes tried to calculate the area of a circle. But it was not until the late 19th and early 20th centuries that measure theory became a branch of mathematics. The foundations of modern measure theory were laid in the works of Émile Borel, Henri Lebesgue, Nikolai Luzin, Johann Radon, Constantin Carathéodory, and Maurice Fréchet, among others. According to Thomas W. Hawkins Jr., "It was primarily through the theory of multiple integrals and, in particular the work of Camille Jordan that the importance of the notion of measurability was first recognized."

to the extended real number line, that is, the real number line together with new (so-called infinite) values

, respectively greater and lower than all other (so-called finite) elements, is called a measure if the following conditions hold:

Countable additivity (or σ-additivity): For all countable collections

{\displaystyle \mu {\left(\bigcup _{k=1}^{\infty }E_{k}\right)}=\sum _{k=1}^{\infty }\mu (E_{k})}

{\displaystyle \mu (E)=\mu (E\cup \varnothing )=\mu (E)+\mu (\varnothing ),}

is called a measure space. A probability measure is a measure with total measure one – that is,

A probability space is a measure space with a probability measure.

For measure spaces that are also topological spaces various compatibility conditions can be placed for the measure and the topology. Most measures met in practice in analysis (and in many cases also in probability theory) are Radon measures (usually defined on Hausdorff spaces). When working with locally compact Hausdorff spaces, Radon measures have an alternative, equivalent definition in terms of linear functionals on the locally convex topological vector space of continuous functions with compact support. This approach is taken by Bourbaki (2004) and a number of other sources. For more details, see the article on Radon measures.

is a complete translation-invariant measure on a σ-algebra containing the intervals in

; and every other measure with these properties extends the Lebesgue measure.

The arc length of an interval on the unit circle in the Euclidean plane extends to a measure on the

-algebra they generate. It can be called angle measure since the arc length of an interval equals the angle it supports. This measure is invariant under rotations preserving the circle. Similarly, hyperbolic angle measure is invariant under squeeze mapping.

The Haar measure for a locally compact topological group. For example,

is such a group and its Haar measure is the Lebesgue measure; for the unit circle (seen as a subgroup of the multiplicative group of

) its Haar measure is the angle measure. For a discrete group the counting measure is a Haar measure.

{\displaystyle {\sqrt {\left|\det g\right|}}d^{n}x}

The Hausdorff measure is a generalization of the Lebesgue measure to sets with non-integer dimension, in particular, fractal sets.

Every probability space gives rise to a measure which takes the value 1 on the whole space (and therefore takes all its values in the unit interval [0, 1]). Such a measure is called a probability measure or distribution. See the list of probability distributions for instances.

The Dirac measure δa (cf. Dirac delta function) is given by δa(S) = χS(a), where χS is the indicator function of

Other 'named' measures used in various theories include: Borel measure, Jordan measure, ergodic measure, Gaussian measure, Baire measure, Radon measure, Young measure, and Loeb measure.

In physics an example of a measure is spatial distribution of mass (see for example, gravity potential), or another non-negative extensive property, conserved (see conservation law for a list of these) or not. Negative values lead to signed measures, see "generalizations" below.

Liouville measure, known also as the natural volume form on a symplectic manifold, is useful in classical statistical and Hamiltonian mechanics.

Gibbs measure is widely used in statistical mechanics, often under the name canonical ensemble.

{\displaystyle \mu \left(\bigcup _{i=1}^{\infty }E_{i}\right)\leq \sum _{i=1}^{\infty }\mu (E_{i}).}

are measurable sets that are increasing (meaning that

{\displaystyle E_{1}\subseteq E_{2}\subseteq E_{3}\subseteq \ldots }

{\displaystyle \mu \left(\bigcup _{i=1}^{\infty }E_{i}\right)~=~\lim _{i\to \infty }\mu (E_{i})=\sup _{i\geq 1}\mu (E_{i}).}

are measurable sets that are decreasing (meaning that

{\displaystyle E_{1}\supseteq E_{2}\supseteq E_{3}\supseteq \ldots }

{\displaystyle \mu \left(\bigcap _{i=1}^{\infty }E_{i}\right)=\lim _{i\to \infty }\mu (E_{i})=\inf _{i\geq 1}\mu (E_{i}).}

This property is false without the assumption that at least one of the

{\displaystyle E_{n}=[n,\infty )\subseteq \mathbb {R} ,}

which all have infinite Lebesgue measure, but the intersection is empty.

A subset of a null set is called a negligible set. A negligible set need not be measurable, but every measurable negligible set is automatically a null set. A measure is called complete if every negligible set is measurable.

A measure can be extended to a complete one by considering the σ-algebra of subsets

which differ by a negligible set from a measurable set

{\displaystyle \mu \{x\in X:f(x)\geq t\}=\mu \{x\in X:f(x)>t\}}

This property is used in connection with Lebesgue integral.

Measures are required to be countably additive. However, the condition can be strengthened as follows.

{\displaystyle \sum _{i\in I}r_{i}=\sup \left\lbrace \sum _{i\in J}r_{i}:|J|<\infty ,J\subseteq I\right\rbrace .}

to be the supremum of all the sums of finitely many of them.

{\displaystyle \bigcup _{\alpha \in \lambda }X_{\alpha }\in \Sigma }

{\displaystyle \mu \left(\bigcup _{\alpha \in \lambda }X_{\alpha }\right)=\sum _{\alpha \in \lambda }\mu \left(X_{\alpha }\right).}

The second condition is equivalent to the statement that the ideal of null sets is

). Nonzero finite measures are analogous to probability measures in the sense that any finite measure

can be decomposed into a countable union of measurable sets of finite measure. Analogously, a set in a measure space is said to have a σ-finite measure if it is a countable union of sets with finite measure.

For example, the real numbers with the standard Lebesgue measure are σ-finite but not finite. Consider the closed intervals

there are countably many such intervals, each has measure 1, and their union is the entire real line. Alternatively, consider the real numbers with the counting measure, which assigns to each finite set of reals the number of points in the set. This measure space is not σ-finite, because every set with finite measure contains only finitely many points, and it would take uncountably many such sets to cover the entire real line. The σ-finite measure spaces have some very convenient properties; σ-finiteness can be compared in this respect to the Lindelöf property of topological spaces. They can be also thought of as a vague generalization of the idea that a measure space may have 'uncountable measure'.

{\displaystyle A\in \mu ^{\text{pre}}\{+\infty \},}

{\displaystyle {\cal {P}}(A)\cap \mu ^{\text{pre}}(\mathbb {R} _{>0})\neq \emptyset .}

Semifinite measures generalize sigma-finite measures, in such a way that some big theorems of measure theory that hold for sigma-finite but not arbitrary measures can be extended with little modification to hold for semifinite measures. (To-do: add examples of such theorems; cf. the talk page.)

is countable). (Thus, counting measure, on the power set

gives an example of a semifinite measure that is not sigma-finite.)

The zero measure is sigma-finite and thus semifinite. In addition, the zero measure is clearly less than or equal to

It can be shown there is a greatest measure with these two properties:

defined in the above theorem. We give some nice, explicit formulas, which some authors may take as definition, for the semifinite part:

{\displaystyle \mu _{\text{sf}}=(\sup\{\mu (B):B\in {\cal {P}}(A)\cap \mu ^{\text{pre}}(\mathbb {R} _{\geq 0})\})_{A\in {\cal {A}}}.}

{\displaystyle \mu _{\text{sf}}=(\sup\{\mu (A\cap B):B\in \mu ^{\text{pre}}(\mathbb {R} _{\geq 0})\})_{A\in {\cal {A}}}\}.}

{\displaystyle \mu _{\text{sf}}=\mu |_{\mu ^{\text{pre}}(\mathbb {R} _{>0})}\cup \{A\in {\cal {A}}:\sup\{\mu (B):B\in {\cal {P}}(A)\}=+\infty \}\times \{+\infty \}\cup \{A\in {\cal {A}}:\sup\{\mu (B):B\in {\cal {P}}(A)\}<+\infty \}\times \{0\}.}

measure that is not the zero measure is not semifinite. (Here, we say

{\displaystyle (\forall A\in {\cal {A}})(\mu (A)\in \{0,+\infty \}).}

{\displaystyle \mu =(\sum _{x\in A}f(x))_{A\in {\cal {A}}}.}

{\displaystyle \mu =\{(\emptyset ,0)\}\cup ({\cal {A}}\setminus \{\emptyset \})\times \{+\infty \}.}

{\displaystyle \mu =\{(\emptyset ,0),(X,+\infty )\}.}

{\displaystyle {\cal {C}}=\{A\in {\cal {A}}:A{\text{ is countable}}\}}

{\displaystyle \mu ={\cal {C}}\times \{0\}\cup ({\cal {A}}\setminus {\cal {C}})\times \{+\infty \}.}

Measures that are not semifinite are very wild when restricted to certain sets. Every measure is, in a sense, semifinite once its

defined in the above theorem. Here is an explicit formula for

{\displaystyle \mu _{0-\infty }=(\sup\{\mu (B)-\mu _{\text{sf}}(B):B\in {\cal {P}}(A)\cap \mu _{\text{sf}}^{\text{pre}}(\mathbb {R} _{\geq 0})\})_{A\in {\cal {A}}}.}

{\displaystyle T:L_{\mathbb {F} }^{\infty }(\mu )\to \left(L_{\mathbb {F} }^{1}(\mu )\right)^{*}:g\mapsto T_{g}=\left(\int fgd\mu \right)_{f\in L_{\mathbb {F} }^{1}(\mu )}.}

is injective. (This result has import in the study of the dual space of

is the measure defined in Theorem 39.1 in Berberian '65.)

Localizable measures are a special case of semifinite measures and a generalization of sigma-finite measures.

{\displaystyle T:L_{\mathbb {F} }^{\infty }(\mu )\to \left(L_{\mathbb {F} }^{1}(\mu )\right)^{*}:g\mapsto T_{g}=\left(\int fgd\mu \right)_{f\in L_{\mathbb {F} }^{1}(\mu )}.}

A measure is said to be s-finite if it is a countable sum of finite measures. S-finite measures are more general than sigma-finite ones and have applications in the theory of stochastic processes.

If the axiom of choice is assumed to be true, it can be proved that not all subsets of Euclidean space are Lebesgue measurable; examples of such sets include the Vitali set, and the non-measurable sets postulated by the Hausdorff paradox and the Banach–Tarski paradox.

For certain purposes, it is useful to have a "measure" whose values are not restricted to the non-negative reals or infinity. For instance, a countably additive set function with values in the (signed) real numbers is called a signed measure, while such a function with values in the complex numbers is called a complex measure. Observe, however, that complex measure is necessarily of finite variation, hence complex measures include finite signed measures but not, for example, the Lebesgue measure.

Measures that take values in Banach spaces have been studied extensively. A measure that takes values in the set of self-adjoint projections on a Hilbert space is called a projection-valued measure; these are used in functional analysis for the spectral theorem. When it is necessary to distinguish the usual measures which take non-negative values from generalizations, the term positive measure is used. Positive measures are closed under conical combination but not general linear combination, while signed measures are the linear closure of positive measures. More generally see measure theory in topological vector spaces.

Another generalization is the finitely additive measure, also known as a content. This is the same as a measure except that instead of requiring countable additivity we require only finite additivity. Historically, this definition was used first. It turns out that in general, finitely additive measures are connected with notions such as Banach limits, the dual of

and the Stone–Čech compactification. All these are linked in one way or another to the axiom of choice.  Contents remain useful in certain technical problems in geometric measure theory; this is the theory of Banach measures.

A charge is a generalization in both directions: it is a finitely additive, signed measure. (Cf. ba space for information about bounded charges, where we say a charge is bounded to mean its range its a bounded subset of R.)
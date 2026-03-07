# Strongly regular graph

In graph theory, a strongly regular graph  (SRG) is a regular graph G = (V, E) with v vertices and degree k such that for some given integers

every two adjacent vertices have λ common neighbours, and

every two non-adjacent vertices have μ common neighbours.

Such a strongly regular graph is denoted by srg(v, k, λ, μ). Its complement graph is also strongly regular: it is an srg(v, v − k − 1, v − 2 − 2k + μ, v − 2k + λ).

If a graph G is strongly regular with μ > 0, then G is distance-regular with diameter 2. Likewise, if G is strongly regular with λ = 1, then it is locally linear.

A strongly regular graph is denoted as an srg(v, k, λ, μ) in the literature. By convention, graphs which satisfy the definition trivially are excluded from detailed studies and lists of strongly regular graphs. These include the disjoint union of one or more equal-sized complete graphs, and their complements, the complete multipartite graphs with equal-sized independent sets.

Andries Brouwer and Hendrik van Maldeghem (see #References) use an alternate but fully equivalent definition of a strongly regular graph based on spectral graph theory: a strongly regular graph is a finite regular graph that has exactly three eigenvalues, only one of which is equal to the degree k, of multiplicity 1. This automatically rules out fully connected graphs (which have only two distinct eigenvalues, not three) and disconnected graphs (for which the multiplicity of the degree k is equal to the number of different connected components, which would therefore exceed one). Much of the literature, including Brouwer, refers to the larger eigenvalue as r (with multiplicity f) and the smaller one as s (with multiplicity g).

Strongly regular graphs were introduced by R.C. Bose in 1963. They built upon earlier work in the 1950s in the then-new field of spectral graph theory.

The Shrikhande graph is an srg(16, 6, 2, 2) which is not a distance-transitive graph.

The n × n square rook's graph, i.e., the line graph of a balanced complete bipartite graph Kn,n, is an srg(n2, 2n − 2, n − 2, 2).  The parameters for n = 4 coincide with those of the Shrikhande graph, but the two graphs are not isomorphic. (The vertex neighborhood for the Shrikhande graph is a hexagon, while that for the rook graph is two triangles.)

{\textstyle \operatorname {srg} \left({\binom {n}{2}},2(n-2),n-2,4\right)}

The three Chang graphs are srg(28, 12, 6, 4), the same as the line graph of K8, but these four graphs are not isomorphic.

Every generalized quadrangle of order (s, t) gives an srg((s + 1)(st + 1), s(t + 1), s − 1, t + 1) as its line graph. For example, GQ(2, 4) gives srg(27, 10, 1, 5) as its line graph.

The Schläfli graph is an srg(27, 16, 10, 8) and is the complement of the aforementioned line graph on GQ(2, 4).

The Hoffman–Singleton graph is an srg(50, 7, 0, 1).

The M22 graph aka the Mesner graph is an srg(77, 16, 0, 4).

The Local McLaughlin graph is an srg(162, 56, 10, 24).

The Berlekamp–van Lint–Seidel graph is an srg(243, 22, 1, 2).

The Paley graph of order q is an srg(q, (q − 1)/2, (q − 5)/4, (q − 1)/4).  The smallest Paley graph, with q = 5, is the 5-cycle (above).

Self-complementary arc-transitive graphs are strongly regular.

A strongly regular graph is called primitive if both the graph and its complement are connected. All the above graphs are primitive, as otherwise μ = 0 or v + λ = 2k.

Conway's 99-graph problem asks for the construction of an srg(99, 14, 1, 2). It is unknown whether a graph with these parameters exists, and John Horton Conway offered a $1000 prize for the solution to this problem.

The strongly regular graphs with λ = 0 are triangle free. Apart from the complete graphs on fewer than 3 vertices and all regular complete bipartite graphs, the seven listed earlier (pentagon, Petersen, Clebsch, Hoffman-Singleton, Gewirtz, Mesner-M22, and Higman-Sims) are the only known ones.

is a geodetic graph, a graph in which every two vertices have a unique shortest path. The only known strongly regular graphs with

is 0, therefore triangle-free as well. These are called the Moore graphs and are explored below in more detail. Other combinations of parameters such as (400, 21, 2, 1) have not yet been ruled out. Despite ongoing research on the properties that a strongly regular graph with

would have, it is not known whether any more exist or even whether their number is finite. Only the elementary result is known, that

The four parameters in an srg(v, k, λ, μ) are not independent: In order for an srg(v, k, λ, μ) to exist, the parameters must obey the following relation:

The above relation is derived through a counting argument as follows:

Imagine the vertices of the graph to lie in three levels. Pick any vertex as the root, in Level 0. Then its k neighbors lie in Level 1, and all other vertices lie in Level 2.

Vertices in Level 1 are directly connected to the root, hence they must have λ other neighbors in common with the root, and these common neighbors must also be in Level 1. Since each vertex has degree k, there are

edges remaining for each Level 1 node to connect to vertices in Level 2. Therefore, there are

Vertices in Level 2 are not directly connected to the root, hence they must have μ common neighbors with the root, and these common neighbors must all be in Level 1. There are

vertices in Level 2, and each is connected to μ vertices in Level 1. Therefore the number of edges between Level 1 and Level 2 is

Equating the two expressions for the edges between Level 1 and Level 2, the relation follows.

This relation is a necessary condition for the existence of a strongly regular graph, but not a sufficient condition. For instance, the quadruple (21,10,4,5) obeys this relation, but there does not exist a strongly regular graph with these parameters.

Let I denote the identity matrix and let J denote the matrix of ones, both matrices of order v.  The adjacency matrix A of a strongly regular graph satisfies two equations.

which is a restatement of the regularity requirement. This shows that k is an eigenvalue of the adjacency matrix with the all-ones eigenvector.

which expresses strong regularity. The ij-th element of the left hand side gives the number of two-step paths from i to j. The first term of the right hand side gives the number of two-step paths from i back to i, namely k edges out and back in. The second term gives the number of two-step paths when i and j are directly connected. The third term gives the corresponding value when i and j are not connected. Since the three cases are mutually exclusive and collectively exhaustive, the simple additive equality follows.

Conversely, a graph whose adjacency matrix satisfies both of the above conditions and which is not a complete or null graph is a strongly regular graph.

Since the adjacency matrix A is symmetric, it follows that its eigenvectors are orthogonal. We already observed one eigenvector above which is made of all ones, corresponding to the eigenvalue k. Therefore the other eigenvectors x must all satisfy

where J is the all-ones matrix as before. Take the previously established equation:

{\displaystyle A^{2}x=kIx+\lambda {A}x+\mu (J-I-A)x}

Call the corresponding eigenvalue p (not to be confused with

{\displaystyle {\frac {1}{2}}\left[(\lambda -\mu )\pm {\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}\,\right]}

. There are thus exactly three eigenvalues for a strongly regular matrix.

Conversely, a connected regular graph with only three eigenvalues is strongly regular.

Following the terminology in much of the strongly regular graph literature, the larger eigenvalue is called r with multiplicity f and the smaller one is called s with multiplicity g.

Since the sum of all the eigenvalues is the trace of the adjacency matrix, which is zero in this case, the respective multiplicities f and g can be calculated:

{\displaystyle r={\frac {1}{2}}\left[(\lambda -\mu )+{\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}\,\right]}

{\displaystyle f={\frac {1}{2}}\left[(v-1)-{\frac {2k+(v-1)(\lambda -\mu )}{\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}}\right]}

{\displaystyle s={\frac {1}{2}}\left[(\lambda -\mu )-{\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}\,\right]}

{\displaystyle g={\frac {1}{2}}\left[(v-1)+{\frac {2k+(v-1)(\lambda -\mu )}{\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}}\right]}

As the multiplicities must be integers, their expressions provide further constraints on the values of v, k, μ, and λ.

have integer eigenvalues with unequal multiplicities.

are called conference graphs because of their connection with symmetric conference matrices. Their parameters reduce to

{\displaystyle \operatorname {srg} \left(v,{\frac {1}{2}}(v-1),{\frac {1}{4}}(v-5),{\frac {1}{4}}(v-1)\right).}

. Further, in this case, v must equal the sum of two squares, related to the Bruck–Ryser–Chowla theorem.

Further properties of the eigenvalues and their multiplicities are:

Given an srg(v, k, λ, μ) with eigenvalues r and s, its complement srg(v, v − k − 1, v − 2 − 2k + μ, v − 2k + λ) has eigenvalues -1-s and -1-r.

{\displaystyle (v-k-1)^{2}(k^{2}+r^{3})\geq (r+1)^{3}k^{2}}

{\displaystyle (v-k-1)^{2}(k^{2}+s^{3})\geq (s+1)^{3}k^{2}}

If any of the above conditions are violated for a set of parameters, then there exists no strongly regular graph for those parameters. Brouwer has compiled such lists of existence or non-existence here with reasons for non-existence if any. For example, there exists no srg(28,9,0,4) because that violates one of the Krein conditions and one of the absolute bound conditions.

As noted above, the multiplicities of the eigenvalues are given by

{\displaystyle M_{\pm }={\frac {1}{2}}\left[(v-1)\pm {\frac {2k+(v-1)(\lambda -\mu )}{\sqrt {(\lambda -\mu )^{2}+4(k-\mu )}}}\right]}

In 1960, Alan Hoffman and Robert Singleton examined those expressions when applied on Moore graphs, which are strongly regular graphs that have λ = 0 and μ = 1. Such graphs are free of triangles (otherwise λ would exceed zero) and quadrilaterals (otherwise μ would exceed 1), hence they have a girth (smallest cycle length) of 5. Substituting the values of λ and μ in the equation

{\displaystyle M_{\pm }={\frac {1}{2}}\left[k^{2}\pm {\frac {2k-k^{2}}{\sqrt {4k-3}}}\right]}

For the multiplicities to be integers, the quantity

k = 0 and v = 1 yields a trivial graph with one vertex and no edges, and

{\displaystyle {\begin{aligned}M_{\pm }&={\frac {1}{2}}\left[\left({\frac {t^{2}+3}{4}}\right)^{2}\pm {\frac {{\frac {t^{2}+3}{2}}-\left({\frac {t^{2}+3}{4}}\right)^{2}}{t}}\right]\\32M_{\pm }&=(t^{2}+3)^{2}\pm {\frac {8(t^{2}+3)-(t^{2}+3)^{2}}{t}}\\&=t^{4}+6t^{2}+9\pm {\frac {-t^{4}+2t^{2}+15}{t}}\\&=t^{4}+6t^{2}+9\pm \left(-t^{3}+2t+{\frac {15}{t}}\right)\end{aligned}}}

must be an integer, therefore t is a factor of 15, namely

k = 1 and v = 2 yields a trivial graph of two vertices joined by an edge,

k = 7 and v = 50 yields the Hoffman–Singleton graph, discovered by Hoffman and Singleton in the course of this analysis, and

k = 57 and v = 3250 famously predicts a graph that has neither been discovered since 1960, nor has its existence been disproven.

The Hoffman-Singleton theorem states that there are no girth-5 Moore graphs except the ones listed above.
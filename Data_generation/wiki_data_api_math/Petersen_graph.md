# Petersen graph

In the mathematical field of graph theory, the Petersen graph is an undirected graph with 10 vertices and 15 edges. It is a small graph that serves as a useful example and counterexample for many problems in graph theory. The Petersen graph is named after Julius Petersen, who in 1898 constructed it to be the smallest bridgeless cubic graph with no three-edge-coloring.

Although the graph is generally credited to Petersen, it had in fact first appeared 12 years earlier, in a paper by A. B. Kempe (1886). Kempe observed that its vertices can represent the ten lines of the Desargues configuration, and its edges represent pairs of lines that do not meet at one of the ten points of the configuration.

Donald Knuth states that the Petersen graph is "a remarkable configuration that serves as a counterexample to many optimistic predictions about what might be true for graphs in general."

The Petersen graph also makes an appearance in tropical geometry. The cone over the Petersen graph is naturally identified with the moduli space of five-pointed rational tropical curves.

The Petersen graph is the complement of the line graph of K5. It is also the Kneser graph KG5,2; this means that it has one vertex for each 2-element subset of a 5-element set, and two vertices are connected by an edge if and only if the corresponding 2-element subsets are disjoint from each other. As a Kneser graph of the form KG2n−1,n−1 it is an example of an odd graph.

Geometrically, the Petersen graph is the graph formed by the vertices and edges of the hemi-dodecahedron, that is, a dodecahedron with opposite points, lines and faces identified together.

The Petersen graph is nonplanar. Any nonplanar graph has as minors either the complete graph K5, or the complete bipartite graph K3,3, but the Petersen graph has both as minors. The K5 minor can be formed by contracting the edges of a perfect matching, for instance the five short edges in the first picture. The K3,3 minor can be formed by deleting one vertex (for instance the central vertex of the 3-symmetric drawing) and contracting an edge incident to each neighbor of the deleted vertex.

The most common and symmetric plane drawing of the Petersen graph, as a pentagram within a pentagon, has five crossings. However, this is not the best drawing for minimizing crossings; there exists another drawing (shown in the figure) with only two crossings. Because it is nonplanar, it has at least one crossing in any drawing, and if a crossing edge is removed from any drawing it remains nonplanar and has another crossing; therefore, its crossing number is exactly 2. Each edge in this drawing is crossed at most once, so the Petersen graph is 1-planar. On a torus the Petersen graph can be drawn without edge crossings; it therefore has orientable genus 1.

The Petersen graph can also be drawn (with crossings) in the plane in such a way that all the edges have equal length. That is, it is a unit distance graph.

The simplest non-orientable surface on which the Petersen graph can be embedded without crossings is the projective plane. This is the embedding given by the hemi-dodecahedron construction of the Petersen graph (shown in the figure). The projective plane embedding can also be formed from the standard pentagonal drawing of the Petersen graph by placing a cross-cap within the five-point star at the center of the drawing, and routing the star edges through this cross-cap; the resulting drawing has six pentagonal faces. This construction forms a regular map and shows that the Petersen graph has non-orientable genus 1.

The Petersen graph is strongly regular (with signature srg(10,3,0,1)). It is also symmetric, meaning that it is edge transitive and vertex transitive. More strongly, it is 3-arc-transitive: every directed three-edge path in the Petersen graph can be transformed into every other such path by a symmetry of the graph.

It is one of only 13 cubic distance-regular graphs.

The automorphism group of the Petersen graph is the symmetric group S5; the action of S5 on the Petersen graph follows from its construction as a Kneser graph. The Petersen graph is a core: every homomorphism of the Petersen graph to itself is an automorphism. As shown in the figures, the drawings of the Petersen graph may exhibit five-way or three-way symmetry, but it is not possible to draw the Petersen graph in the plane in such a way that the drawing exhibits the full symmetry group of the graph.

Despite its high degree of symmetry, the Petersen graph is not a Cayley graph. It is the smallest vertex-transitive graph that is not a Cayley graph.

The Petersen graph has a Hamiltonian path but no Hamiltonian cycle. It is the smallest bridgeless cubic graph with no Hamiltonian cycle. It is hypohamiltonian, meaning that although it has no Hamiltonian cycle, deleting any vertex makes it Hamiltonian, and is the smallest hypohamiltonian graph.

As a finite connected vertex-transitive graph that does not have a Hamiltonian cycle, the Petersen graph is a counterexample to a variant of the Lovász conjecture, but the canonical formulation of the conjecture asks for a Hamiltonian path and is verified by the Petersen graph.

Only five connected vertex-transitive graphs with no Hamiltonian cycles are known: the complete graph K2, the Petersen graph, the Coxeter graph and two graphs derived from the Petersen and Coxeter graphs by replacing each vertex with a triangle. If G is a 2-connected, r-regular graph with at most 3r + 1 vertices, then G is Hamiltonian or G is the Petersen graph.

To see that the Petersen graph has no Hamiltonian cycle, consider the edges in the cut disconnecting the inner 5-cycle from the outer one. If there is a Hamiltonian cycle C, it must contain an even number of these edges. If it contains only two of them, their end-vertices must be adjacent in the two 5-cycles, which is not possible. Hence, it contains exactly four of them. Assume that the top edge of the cut is not contained in C (all the other cases are the same by symmetry). Of the five edges in the outer cycle, the two top edges must be in C, the two side edges must not be in C, and hence the bottom edge must be in C. The top two edges in the inner cycle must be in C, but this completes a non-spanning cycle, which cannot be part of a Hamiltonian cycle. Alternatively, we can also describe the ten-vertex 3-regular graphs that do have a Hamiltonian cycle and show that none of them is the Petersen graph, by finding a cycle in each of them that is shorter than any cycle in the Petersen graph. Any ten-vertex Hamiltonian 3-regular graph consists of a ten-vertex cycle C plus five chords. If any chord connects two vertices at distance two or three along C from each other, the graph has a 3-cycle or 4-cycle, and therefore cannot be the Petersen graph. If two chords connect opposite vertices of C to vertices at distance four along C, there is again a 4-cycle. The only remaining case is a Möbius ladder formed by connecting each pair of opposite vertices by a chord, which again has a 4-cycle. Since the Petersen graph has girth five, it cannot be formed in this way and has no Hamiltonian cycle.

The Petersen graph has chromatic number 3, meaning that its vertices can be colored with three colors — but not with two — such that no edge connects vertices of the same color. It has a list coloring with 3 colors, by Brooks's theorem for list colorings.

The Petersen graph has chromatic index 4; coloring the edges requires four colors. As a connected bridgeless cubic graph with chromatic index four, the Petersen graph is a snark. It is the smallest possible snark, and was the only known snark from 1898 until 1946. The snark theorem, a result conjectured by W. T. Tutte and announced in 2001 by Robertson, Sanders, Seymour, and Thomas, states that every snark has the Petersen graph as a minor.

Additionally, the graph has fractional chromatic index 3, proving that the difference between the chromatic index and fractional chromatic index can be as large as 1. The long-standing Goldberg-Seymour Conjecture proposes that this is the largest gap possible.

The Thue number (a variant of the chromatic index) of the Petersen graph is 5.

The Petersen graph requires at least three colors in any (possibly improper) coloring that breaks all of its symmetries; that is, its distinguishing number is three. Except for the complete graphs, it is the only Kneser graph whose distinguishing number is not two.

is 3-connected and hence 3-edge-connected and bridgeless. See the glossary.

has independence number 4 and is 3-partite. See the glossary.

is cubic, has domination number 3, and has a perfect matching and a 2-factor.

is the smallest cubic graph of girth 5. (It is the unique (3,5)-cage. In fact, since it has only 10 vertices, it is the unique (3,5)-Moore graph.)

every cubic bridgeless graph without Petersen minor has a cycle double cover.

is the smallest cubic graph with Colin de Verdière graph invariant μ = 5.

has radius 2 and diameter 2. It is the largest cubic graph with diameter 2.

has 2000 spanning trees, the most of any 10-vertex cubic graph.

has chromatic polynomial t(t − 1)(t − 2)(t7 − 12t6 + 67t5 − 230t4 + 529t3 − 814t2 + 775t − 352).

has characteristic polynomial (t − 1)5(t + 2)4(t − 3), making it an integral graph—a graph whose spectrum consists entirely of integers.

An Eulerian subgraph of a graph G is a subgraph consisting of a subset of the edges of G, touching every vertex of G an even number of times. These subgraphs are the elements of the cycle space of G and are sometimes called cycles. If G and H are any two graphs, a function from the edges of G to the edges of H is defined to be cycle-continuous if the pre-image of every cycle of H is a cycle of G. A conjecture of Jaeger asserts that every bridgeless graph has a cycle-continuous mapping to the Petersen graph. Jaeger showed this conjecture implies the 5-cycle-double-cover conjecture and the Berge-Fulkerson conjecture."

The generalized Petersen graph G(n,k) is formed by connecting the vertices of a regular n-gon to the corresponding vertices of a star polygon with Schläfli symbol {n/k}. For instance, in this notation, the Petersen graph is G(5,2): it can be formed by connecting corresponding vertices of a pentagon and five-point star, and the edges in the star connect every second vertex. The generalized Petersen graphs also include the n-prism G(n,1), the Dürer graph G(6,2), the Möbius–Kantor graph G(8,3), the dodecahedron G(10,2), the Desargues graph G(10,3), and the Nauru graph G(12,5).

The Petersen family consists of the seven graphs that can be formed from the Petersen graph by zero or more applications of Δ-Y or Y-Δ transforms. The complete graph K6 is also in the Petersen family. These graphs form the forbidden minors for linklessly embeddable graphs, graphs that can be embedded into three-dimensional space in such a way that no two cycles in the graph are linked.

The Clebsch graph contains many copies of the Petersen graph as induced subgraphs: for each vertex v of the Clebsch graph, the ten non-neighbors of v induce a copy of the Petersen graph.
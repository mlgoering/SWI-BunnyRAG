# Glossary of graph theory

This is a glossary of graph theory. Graph theory is the study of graphs, systems of nodes or vertices connected in pairs by lines or edges.

G[S] is the induced subgraph of a graph G for vertex subset S.

The prime symbol is often used to modify notation for graph invariants so that it applies to the line graph instead of the given graph. For instance, α(G) is the independence number of a graph; α′(G) is the matching number of the graph, which equals the independence number of its line graph. Similarly, χ(G) is the chromatic number of a graph; χ ′(G) is the chromatic index of the graph, which equals the chromatic number of its line graph.

The achromatic number of a graph is the maximum number of colors in a complete coloring.

1.  A graph is acyclic if it has no cycles. An undirected acyclic graph is the same thing as a forest. An acyclic directed graph, which is a digraph without directed cycles, is often called a directed acyclic graph, especially in computer science.

2.  An acyclic coloring of an undirected graph is a proper coloring in which every two color classes induce a forest.

The adjacency matrix of a graph is a matrix whose rows and columns are both indexed by vertices of the graph, with a one in the cell for row i and column j when vertices i and j are adjacent, and a zero otherwise.

1.  The relation between two vertices that are both endpoints of the same edge.

2.  The relation between two distinct edges that share an end vertex.

For a graph G, α(G) (using the Greek letter alpha) is its independence number (see independent), and α′(G) is its matching number (see matching).

In a graph with a matching, an alternating path is a path whose edges alternate between matched and unmatched edges. An alternating cycle is, similarly, a cycle whose edges alternate between matched and unmatched edges. An augmenting path is an alternating path that starts and ends at unsaturated vertices. A larger matching can be found as the symmetric difference of the matching and the augmenting path; a matching is maximum if and only if it has no augmenting path.

In a directed acyclic graph, a subset S of vertices that are pairwise incomparable, i.e., for any

in S, there is no directed path from x to y or from y to x. Inspired by the notion of antichains in partially ordered sets.

Synonym for non-edge, a pair of non-adjacent vertices.

A three-vertex independent set, the complement of a triangle.

1.  An apex graph is a graph in which one vertex can be removed, leaving a planar subgraph. The removed vertex is called the apex. A k-apex graph is a graph that can be made planar by the removal of k vertices.

2.  Synonym for universal vertex, a vertex adjacent to all other vertices.

An ordered pair of vertices, such as an edge in a directed graph. An arrow (x, y) has a tail x, a head y, and a direction from x to y; y is said to be the direct successor to x and x the direct predecessor to y. The arrow (y, x) is the inverted arrow of the arrow (x, y).

A vertex in a connected graph whose removal would disconnect the graph. More generally, a vertex whose removal increases the number of components.

A k-ary tree is a rooted tree in which every internal vertex has no more than k children. A 1-ary tree is just a path. A 2-ary tree is also called a binary tree, although that term more properly refers to 2-ary trees in which the children of each node are distinguished as being left or right children (with at most one of each type).  A k-ary tree is said to be complete if every internal vertex has exactly k children.

A special type of alternating path; see alternating.

A graph automorphism is a symmetry of a graph, an isomorphism from the graph to itself.

One of the sets of vertices in a tree decomposition.

A bipartite or multipartite graph is balanced if each two subsets of its vertex partition have sizes within one of each other.

A ball (also known as a neighborhood ball or distance ball) is the set of all vertices that are at most distance r from a vertex. More formally, for a given vertex v and radius r, the ball B(v,r) consists of all vertices whose shortest path distance to v is less than or equal to r.

The bandwidth of a graph G is the minimum, over all orderings of vertices of G, of the length of the longest edge (the number of steps in the ordering between its two endpoints). It is also one less than the size of the maximum clique in a proper interval completion of G, chosen to minimize the clique size.

Synonym for complete bipartite graph or complete bipartite subgraph; see complete.

Usually a synonym for 2-vertex-connected, but sometimes includes K2 though it is not 2-connected. See connected; for biconnected components, see component.

The smallest possible ratio of the number of neighbors of a proper subset of vertices to the size of the subset.

A bipartite graph is a graph whose vertices can be divided into two disjoint sets such that the vertices in one set are not connected to each other, but may be connected to vertices in the other set. Put another way, a bipartite graph is a graph with no odd cycles; equivalently, it is a graph that may be properly colored with two colors. Bipartite graphs are often written G = (U,V,E) where U and V are the subsets of vertices of each color. However, unless the graph is connected, it may not have a unique 2-coloring.

A biregular graph is a bipartite graph in which there are only two different vertex degrees, one for each set of the vertex bipartition.

1.  A block of a graph G is a maximal subgraph which is either an isolated vertex, a bridge edge, or a 2-connected subgraph. If a block is 2-connected, every pair of vertices in it belong to a common cycle. Every edge of a graph belongs in exactly one block.

2.  The block graph of a graph G is another graph whose vertices are the blocks of G, with an edge connecting two vertices when the corresponding blocks share an articulation point; that is, it is the intersection graph of the blocks of G.  The block graph of any graph is a forest.

3.  The block-cut (or block-cutpoint) graph of a graph G is a bipartite graph where one partite set consists of the cut-vertices of G, and the other has a vertex

of G. When G is connected, its block-cutpoint graph is a tree.

4.  A block graph (also called a clique tree if connected, and sometimes erroneously called a Husimi tree) is a graph all of whose blocks are complete graphs. A forest is a block graph; so in particular the block graph of any graph is a block graph, and every block graph may be constructed as the block graph of a graph.

A minimal cut-set: a set of edges whose removal disconnects the graph, for which no proper subset has the same property.

1.  A book, book graph, or triangular book is a complete tripartite graph K1,1,n; a collection of n triangles joined at a shared edge.

2.  Another type of graph, also called a book, or a quadrilateral book, is a collection of 4-cycles joined at a shared edge; the Cartesian product of a star with an edge.

3.  A book embedding is an embedding of a graph onto a topological book, a space formed by joining a collection of half-planes along a shared line. Usually, the vertices of the embedding are required to be on the line, which is called the spine of the embedding, and the edges of the embedding are required to lie within a single half-plane, one of the pages of the book.

1.   In a graph embedding, a boundary walk is the subgraph containing all incident edges and vertices to a face.

A bramble is a collection of mutually touching connected subgraphs, where two subgraphs touch if they share a vertex or each includes one endpoint of an edge. The order of a bramble is the smallest size of a set of vertices that has a nonempty intersection with all of the subgraphs. The treewidth of a graph is the maximum order of any of its brambles.

A path of degree-two vertices, ending at vertices whose degree is unequal to two.

A branch-decomposition of G is a hierarchical clustering of the edges of G, represented by an unrooted binary tree with its leaves labeled by the edges of G. The width of a branch-decomposition is the maximum, over edges e of this binary tree, of the number of shared vertices between the subgraphs determined by the edges of G in the two subtrees separated by e. The branchwidth of G is the minimum width of any branch-decomposition of G.

1.  A bridge, isthmus, or cut edge is an edge whose removal would disconnect the graph. A bridgeless graph is one that has no bridges; equivalently, a 2-edge-connected graph.

2.  A bridge of a subgraph H is a maximal connected subgraph separated from the rest of the graph by H.  That is, it is a maximal subgraph that is edge-disjoint from H and in which each two vertices and edges belong to a path that is internally disjoint from H.  H may be a set of vertices.  A chord is a one-edge bridge.

In planarity testing, H is a cycle and a peripheral cycle is a cycle with at most one bridge; it must be a face boundary in any planar embedding of its graph.

3.  A bridge of a cycle can also mean a path that connects two vertices of a cycle but is shorter than either of the paths in the cycle connecting the same two vertices. A bridged graph is a graph in which every cycle of four or more vertices has a bridge.

A bridgeless or isthmus-free graph is a graph that has no bridge edges (i.e., isthmi); that is, each connected component is a 2-edge-connected graph.

1.  The butterfly graph has five vertices and six edges; it is formed by two triangles that share a vertex.

2.  The butterfly network is a graph used as a network architecture in distributed computing, closely related to the cube-connected cycles.

A cactus graph, cactus tree, cactus, or Husimi tree is a connected graph in which each edge belongs to at most one cycle. Its blocks are cycles or single edges. If, in addition, each vertex belongs to at most two blocks, then it is called a Christmas cactus.

A cage is a regular graph with the smallest possible order for its girth.

A canonical form of a graph is an invariant such that two graphs have equal invariants if and only if they are isomorphic. Canonical forms may also be called canonical invariants or complete invariants, and are sometimes defined only for the graphs within a particular family of graphs. Graph canonization is the process of computing a canonical form.

A graph formed from a given graph by deleting one vertex, especially in the context of the reconstruction conjecture. See also deck, the multiset of all cards of a graph.

Carving width is a notion of graph width analogous to branchwidth, but using hierarchical clusterings of vertices instead of hierarchical clusterings of edges.

A caterpillar tree or caterpillar is a tree in which the internal nodes induce a path.

The center of a graph is the set of vertices of minimum eccentricity.

A centroid of a tree is a vertex v such that if rooted at v, no other vertex has subtree size greater than half the size of the tree.

2.  When applying methods from algebraic topology to graphs, an element of a chain complex, namely a set of vertices or a set of edges.

χ(G) (using the Greek letter chi) is the chromatic number of G and χ ′(G) is its chromatic index; see chromatic and coloring.

In a rooted tree, a child of a vertex v is a neighbor of v along an outgoing edge, one that is directed away from the root.

1.  A chord of a cycle is an edge that does not belong to the cycle, for which both endpoints belong to the cycle.

2.  A chordal graph is a graph in which every cycle of four or more vertices has a chord, so the only induced cycles are triangles.

3.  A strongly chordal graph is a chordal graph in which every cycle of length six or more has an odd chord.

4.  A chordal bipartite graph is not chordal (unless it is a forest); it is a bipartite graph in which every cycle of six or more vertices has a chord, so the only induced cycles are 4-cycles.

5.  A chord of a circle is a line segment connecting two points on the circle; the intersection graph of a collection of chords is called a circle graph.

Having to do with coloring; see color. Chromatic graph theory is the theory of graph coloring. The chromatic number χ(G) is the minimum number of colors needed in a proper coloring of G. χ ′(G) is the chromatic index of G, the minimum number of colors needed in a proper edge coloring of G.

A graph is k-choosable if it has a list coloring whenever each vertex has a list of k available colors. The choosability of the graph is the smallest k for which it is k-choosable.

A circle graph is the intersection graph of chords of a circle.

A circuit may refer to a closed trail or an element of the cycle space (an Eulerian spanning subgraph). The circuit rank of a graph is the dimension of its cycle space.

The circumference of a graph is the length of its longest simple cycle. The graph is Hamiltonian if and only if its circumference equals its order.

1.  A class of graphs or family of graphs is a (usually infinite) collection of graphs, often defined as the graphs having some specific property. The word "class" is used rather than "set" because, unless special restrictions are made (such as restricting the vertices to be drawn from a particular set, and defining edges to be sets of two vertices) classes of graphs are usually not sets when formalized using set theory.

2.  A color class of a colored graph is the set of vertices or edges having one particular color.

3.  In the context of Vizing's theorem, on edge coloring simple graphs, a graph is said to be of class one if its chromatic index equals its maximum degree, and class two if its chromatic index equals one plus the degree. According to Vizing's theorem, all simple graphs are either of class one or class two.

A claw is a tree with one internal vertex and three leaves, or equivalently the complete bipartite graph K1,3. A claw-free graph is a graph that does not have an induced subgraph that is a claw.

A clique is a set of mutually adjacent vertices (or the complete subgraph induced by that set).  Sometimes a clique is defined as a maximal set of mutually adjacent vertices (or maximal complete subgraph), one that is not part of any larger such set (or subgraph). A k-clique is a clique of order k. The clique number ω(G) of a graph G is the order of its largest clique. The clique graph of a graph G is the intersection graph of the maximal cliques in G. See also biclique, a complete bipartite subgraph.

The clique-width of a graph G is the minimum number of distinct labels needed to construct G by operations that create a labeled vertex, form the disjoint union of two labeled graphs, add an edge connecting all pairs of vertices with given labels, or relabel all vertices with a given label. The graphs of clique-width at most 2 are exactly the cographs.

1.  A closed neighborhood is one that includes its central vertex; see neighbourhood.

2.  A closed walk is one that starts and ends at the same vertex; see walk.

3.  A graph is transitively closed if it equals its own transitive closure; see transitive.

4.  A graph property is closed under some operation on graphs if, whenever the argument or arguments to the operation have the property, then so does the result. For instance, hereditary properties are closed under induced subgraphs; monotone properties are closed under subgraphs; and minor-closed properties are closed under minors.

1.  For the transitive closure of a directed graph, see transitive.

2.  A closure of a directed graph is a set of vertices that have no outgoing edges to vertices outside the closure. For instance, a sink is a one-vertex closure. The closure problem is the problem of finding a closure of minimum or maximum weight.

This prefix has various meanings usually involving complement graphs. For instance, a cograph is a graph produced by operations that include complementation; a cocoloring is a coloring in which each vertex induces either an independent set (as in proper coloring) or a clique (as in a coloring of the complement).

1.  A graph coloring is a labeling of the vertices of a graph by elements from a given set of colors, or equivalently a partition of the vertices into subsets, called "color classes", each of which is associated with one of the colors.

2.  Some authors use "coloring", without qualification, to mean a proper coloring, one that assigns different colors to the endpoints of each edge. In graph coloring, the goal is to find a proper coloring that uses as few colors as possible; for instance, bipartite graphs are the graphs that have colorings with only two colors, and the four color theorem states that every planar graph can be colored with at most four colors. A graph is said to be k-colored if it has been (properly) colored with k colors, and k-colorable or k-chromatic if this is possible.

3.  Many variations of coloring have been studied, including edge coloring (coloring edges so that no two edges with the same endpoint share a color), list coloring (proper coloring with each vertex restricted to a subset of the available colors), acyclic coloring (every 2-colored subgraph is acyclic), co-coloring (every color class induces an independent set or a clique), complete coloring (every two color classes share an edge), and total coloring (both edges and vertices are colored).

4.  The coloring number of a graph is one plus the degeneracy. It is so called because applying a greedy coloring algorithm to a degeneracy ordering of the graph uses at most this many colors.

A commuting graph of a group or more generally a semigroup is an undirected graph in which the vertices are elements of the group/semigroup and there is an edge between any pair of elements that commute (that is, there is an edge between vertices x and y if and only if xy=yx).

An undirected graph is a comparability graph if its vertices are the elements of a partially ordered set and two vertices are adjacent when they are comparable in the partial order. Equivalently, a comparability graph is a graph that has a transitive orientation. Many other classes of graphs can be defined as the comparability graphs of special types of partial order.

of a simple graph G is another graph on the same vertex set as G, with an edge for each two vertices that are not adjacent in G.

1.  A complete graph is one in which every two vertices are adjacent: all edges that could exist are present. A complete graph with n vertices is often denoted Kn. A complete bipartite graph is one in which every two vertices on opposite sides of the partition of vertices are adjacent. A complete bipartite graph with a vertices on one side of the partition and b vertices on the other side is often denoted Ka,b. The same terminology and notation has also been extended to complete multipartite graphs, graphs in which the vertices are divided into more than two subsets and every pair of vertices in different subsets are adjacent; if the numbers of vertices in the subsets are a, b, c, ... then this graph is denoted Ka, b, c, ....

2.  A completion of a given graph is a supergraph that has some desired property. For instance, a chordal completion is a supergraph that is a chordal graph.

3.  A complete matching is a synonym for a perfect matching; see matching.

4.  A complete coloring is a proper coloring in which each pairs of colors is used for the endpoints of at least one edge. Every coloring with a minimum number of colors is complete, but there may exist complete colorings with larger numbers of colors. The achromatic number of a graph is the maximum number of colors in a complete coloring.

5.  A complete invariant of a graph is a synonym for a canonical form, an invariant that has different values for non-isomorphic graphs.

A connected component of a graph is a maximal connected subgraph. The term is also used for maximal subgraphs or subsets of a graph's vertices that have some higher order of connectivity, including biconnected components, triconnected components, and strongly connected components.

The condensation of a directed graph G is a directed acyclic graph with one vertex for each strongly connected component of G, and an edge connecting pairs of components that contain the two endpoints of at least one edge in G.

A connected graph is one in which each pair of vertices forms the endpoints of a path. Higher forms of connectivity include strong connectivity in directed graphs (for each two vertices there are paths from one to the other in both directions), k-vertex-connected graphs (removing fewer than k vertices cannot disconnect the graph), and k-edge-connected graphs (removing fewer than k edges cannot disconnect the graph).

Edge contraction is an elementary operation that removes an edge from a graph while merging the two vertices that it previously joined. Vertex contraction (sometimes called vertex identification) is similar, but the two vertices are not necessarily connected by an edge. Path contraction occurs upon the set of edges in a path that contract to form a single edge between the endpoints of the path. The inverse of edge contraction is vertex splitting.

The converse graph is a synonym for the transpose graph; see transpose.

1.  A k-core is the induced subgraph formed by removing all vertices of degree less than k, and all vertices whose degree becomes less than k after earlier removals. See degeneracy.

2.  A core is a graph G such that every graph homomorphism from G to itself is an isomorphism.

3.  The core of a graph G is a minimal graph H such that there exist homomorphisms from G to H and vice versa. H is unique up to isomorphism. It can be represented as an induced subgraph of G, and is a core in the sense that all of its self-homomorphisms are isomorphisms.

4.  In the theory of graph matchings, the core of a graph is an aspect of its Dulmage–Mendelsohn decomposition, formed as the union of all maximum matchings.

2.  A rooted tree structure used to describe a cograph, in which each cograph vertex is a leaf of the tree, each internal node of the tree is labeled with 0 or 1, and two cograph vertices are adjacent if and only if their lowest common ancestor in the tree is labeled 1.

A vertex cover is a set of vertices incident to every edge in a graph. An edge cover is a set of edges incident to every vertex in a graph. A set of subgraphs of a graph covers that graph if its union – taken vertex-wise and edge-wise – is equal to the graph.

A critical graph for a given property is a graph that has the property but such that every subgraph formed by deleting a single vertex does not have the property. For instance, a factor-critical graph is one that has a perfect matching (a 1-factor) for every vertex deletion, but (because it has an odd number of vertices) has no perfect matching itself. Compare hypo-, used for graphs which do not have a property but for which every one-vertex deletion does.

1.  Cube graph, the eight-vertex graph of the vertices and edges of a cube.

2.  Hypercube graph, a higher-dimensional generalization of the cube graph.

3.  Folded cube graph, formed from a hypercube by adding a matching connecting opposite vertices.

4.  Halved cube graph, the half-square of a hypercube graph.

5.  Partial cube, a distance-preserving subgraph of a hypercube.

7.  Cubic graph, another name for a 3-regular graph, one in which each vertex has three incident edges.

8.  Cube-connected cycles, a cubic graph formed by replacing each vertex of a hypercube by a cycle.

A cut is a partition of the vertices of a graph into two subsets, or the set (also known as a cut-set) of edges that span such a partition, if that set is non-empty. An edge is said to span the partition if it has endpoints in both subsets. Thus, the removal of a cut-set from a connected graph disconnects it.

The cut space of a graph is a GF(2)-vector space having the cut-sets of the graph as its elements and symmetric difference of sets as its vector addition operation.

1.  A cycle may be either a kind of graph or a kind of walk.  As a walk it may be either be a closed walk (also called a tour) or more usually a closed walk without repeated vertices and consequently edges (also called a simple cycle). In the latter case it is usually regarded as a graph, i.e., the choices of first vertex and direction are usually considered unimportant; that is, cyclic permutations and reversals of the walk produce the same cycle. Important special types of cycle include Hamiltonian cycles, induced cycles, peripheral cycles, and the shortest cycle, which defines the girth of a graph. A k-cycle is a cycle of length k; for instance a 2-cycle is a digon and a 3-cycle is a triangle. A cycle graph is a graph that is itself a simple cycle; a cycle graph with n vertices is commonly denoted Cn.

2.  The cycle space is a vector space generated by the simple cycles in a graph, often over the field of 2 elements but also over other fields.

Abbreviation for directed acyclic graph, a directed graph without any directed cycles.

The multiset of graphs formed from a single graph G by deleting a single vertex in all possible ways, especially in the context of the reconstruction conjecture. An edge-deck is formed in the same way by deleting a single edge in all possible ways. The graphs in a deck are also called cards. See also critical (graphs that have a property that is not held by any card) and hypo- (graphs that do not have a property that is held by all cards).

See tree decomposition, path decomposition, or branch-decomposition.

A k-degenerate graph is an undirected graph in which every induced subgraph has minimum degree at most k. The degeneracy of a graph is the smallest k for which it is k-degenerate. A degeneracy ordering is an ordering of the vertices such that each vertex has minimum degree in the induced subgraph of it and all later vertices; in a degeneracy ordering of a k-degenerate graph, every vertex has at most k later neighbours. Degeneracy is also known as the k-core number, width, and linkage, and one plus the degeneracy is also called the coloring number or Szekeres–Wilf number. k-degenerate graphs have also been called k-inductive graphs.

1.  The degree of a vertex in a graph is its number of incident edges. The degree of a graph G (or its maximum degree) is the maximum of the degrees of its vertices, often denoted Δ(G); the minimum degree of G is the minimum of its vertex degrees, often denoted δ(G). Degree is sometimes called valency; the degree of v in G may be denoted dG(v), d(G), or deg(v). The total degree is the sum of the degrees of all vertices; by the handshaking lemma it is an even number. The degree sequence is the collection of degrees of all vertices, in sorted order from largest to smallest. In a directed graph, one may distinguish the in-degree (number of incoming edges) and out-degree (number of outgoing edges).

2.  The homomorphism degree of a graph is a synonym for its Hadwiger number, the order of the largest clique minor.

Δ(G) (using the Greek letter delta) is the maximum degree of a vertex in G, and δ(G) is the minimum degree; see degree.

In a graph of n nodes, the density is the ratio of the number of edges of the graph to the number of edges in a complete graph on n nodes. See dense graph.

The depth of a node in a rooted tree is the number of edges in the path from the root to the node. For instance, the depth of the root is 0 and the depth of any one of its adjacent nodes is 1. It is the level of a node minus one. Note, however, that some authors instead use depth as a synonym for the level of a node.

The diameter of a connected graph is the maximum length of a shortest path. That is, it is the maximum of the distances between pairs of vertices in the graph. If the graph has weights on its edges, then its weighted diameter measures path length by the sum of the edge weights along a path, while the unweighted diameter measures path length by the number of edges.

For disconnected graphs, definitions vary: the diameter may be defined as infinite, or as the largest diameter of a connected component, or it may be undefined.

The diamond graph is an undirected graph with four vertices and five edges.

Strongly connected. (Not to be confused with disconnected)

A digon is a simple cycle of length two in a directed graph or a multigraph. Digons cannot occur in simple undirected graphs as they require repeating the same edge twice, which violates the definition of simple.

The tail of a directed edge whose head is the given vertex.

The head of a directed edge whose tail is the given vertex.

A directed graph is one in which the edges have a distinguished direction, from one vertex to another. In a mixed graph, a directed edge is again one that has a distinguished direction; directed edges may also be called arcs or arrows.

A path in which all the edges have the same direction. If a directed path leads from vertex x to vertex y, x is a predecessor of y, y is a successor of  x, and y is said to be reachable from x.

1.  The asymmetric relation between two adjacent vertices in a graph, represented as an arrow.

2.  The asymmetric relation between two vertices in a directed path.

1.  Two subgraphs are edge disjoint if they share no edges, and vertex disjoint if they share no vertices.

2.  The disjoint union of two or more graphs is a graph whose vertex and edge sets are the disjoint unions of the corresponding sets.

A subset of vertices in a graph G is called dissociation if it induces a subgraph with maximum degree 1.

The distance between any two vertices in a graph is the length of the shortest path having the two vertices as its endpoints.

A domatic partition of a graph is a partition of the vertices into dominating sets. The domatic number of the graph is the maximum number of dominating sets in such a partition.

A dominating set is a set of vertices that includes or is adjacent to every vertex in the graph; not to be confused with a vertex cover, a vertex set that is incident to all edges in the graph. Important special types of dominating sets include independent dominating sets (dominating sets that are also independent sets) and connected dominating sets (dominating sets that induced connected subgraphs). A single-vertex dominating set may also be called a universal vertex. The domination number of a graph is the number of vertices in the smallest dominating set.

dualA dual graph of a plane graph G is a graph that has a vertex for each face of G.

An ear of a graph is a path whose endpoints may coincide but in which otherwise there are no repetitions of vertices or edges.

An ear decomposition is a partition of the edges of a graph into a sequence of ears, each of whose endpoints (after the first one) belong to a previous ear and each of whose interior points do not belong to any previous ear. An open ear is a simple path (an ear without repeated vertices), and an open ear decomposition is an ear decomposition in which each ear after the first is open; a graph has an open ear decomposition if and only if it is biconnected. An ear is odd if it has an odd number of edges, and an odd ear decomposition is an ear decomposition in which each ear is odd; a graph has an odd ear decomposition if and only if it is factor-critical.

The eccentricity of a vertex is the farthest distance from it to any other vertex.

An edge is (together with vertices) one of the two basic units out of which graphs are constructed. Each edge has two (or in hypergraphs, more) vertices to which it is attached, called its endpoints. Edges may be directed or undirected; undirected edges are also called lines and directed edges are also called arcs or arrows. In an undirected simple graph, an edge may be represented as the set of its vertices, and in a directed simple graph it may be represented as an ordered pair of its vertices. An edge that connects vertices x and y is sometimes written xy.

A set of edges whose removal disconnects the graph. A one-edge cut is called a bridge, isthmus, or cut edge.

The set of edges of a given graph G, sometimes denoted by E(G).

The edgeless graph or totally disconnected graph on a given set of vertices is the graph that has no edges. It is sometimes called the empty graph, but this term can also refer to a graph with no vertices.

A graph embedding is a topological representation of a graph as a subset of a topological space with each vertex represented as a point, each edge represented as a curve having the endpoints of the edge as endpoints of the curve, and no other intersections between vertices or edges. A planar graph is a graph that has such an embedding onto the Euclidean plane, and a toroidal graph is a graph that has such an embedding onto a torus. The genus of a graph is the minimum possible genus of a two-dimensional manifold onto which it can be embedded.

1.  An edgeless graph on a nonempty set of vertices.

2.  The order-zero graph, a graph with no vertices and no edges.

An end of an infinite graph is an equivalence class of rays, where two rays are equivalent if there is a third ray that includes infinitely many vertices from both of them.

One of the two vertices joined by a given edge, or one of the first or last vertex of a walk, trail or path. The first endpoint of a given directed edge is called the tail and the second endpoint is called the head.

Graph enumeration is the problem of counting the graphs in a given class of graphs, as a function of their order. More generally, enumeration problems can refer either to problems of counting a certain class of combinatorial objects (such as cliques, independent sets, colorings, or spanning trees), or of algorithmically listing all such objects.

An Eulerian path is a walk that uses every edge of a graph exactly once. An Eulerian circuit (also called an Eulerian cycle or an Euler tour) is a closed walk that uses every edge exactly once. An Eulerian graph is a graph that has an Eulerian circuit. For an undirected graph, this means that the graph is connected and every vertex has even degree. For a directed graph, this means that the graph is strongly connected and every vertex has in-degree equal to the out-degree. In some cases, the connectivity requirement is loosened, and a graph meeting only the degree requirements is called Eulerian.

Divisible by two; for instance, an even cycle is a cycle whose length is even.

An expander graph is a graph whose edge expansion, vertex expansion, or spectral expansion is bounded away from zero.

1.  The edge expansion, isoperimetric number, or Cheeger constant of a graph G is the minimum ratio, over subsets S of at most half of the vertices of G, of the number of edges leaving S to the number of vertices in S.

2.  The vertex expansion, vertex isoperimetric number, or magnification of a graph G is the minimum ratio, over subsets S of at most half of the vertices of G, of the number of vertices outside but adjacent to S to the number of vertices in S.

3.  The unique neighbor expansion of a graph G is the minimum ratio, over subsets of at most half of the vertices of G, of the number of vertices outside S but adjacent to a unique vertex in S to the number of vertices in S.

4.  The spectral expansion of a d-regular graph G is the spectral gap between the largest eigenvalue d of its adjacency matrix and the second-largest eigenvalue.

5.  A family of graphs has bounded expansion if all its r-shallow minors have a ratio of edges to vertices bounded by a function of r, and polynomial expansion if the function of r is a polynomial.

In a plane graph or graph embedding, a connected component of the subset of the plane or surface of the embedding that is disjoint from the graph. For an embedding in the plane, all but one face will be bounded; the one exceptional face that extends to infinity is called the outer (or infinite) face.

A factor of a graph is a spanning subgraph: a subgraph that includes all of the vertices of the graph. The term is primarily used in the context of regular subgraphs: a k-factor is a factor that is k-regular. In particular, a 1-factor is the same thing as a perfect matching. A factor-critical graph is a graph for which deleting any one vertex produces a graph with a 1-factor.

A graph factorization is a partition of the edges of the graph into factors; a k-factorization is a partition into k-factors. For instance a 1-factorization is an edge coloring with the additional property that each vertex is incident to an edge of each color.

A graph is finite if it has a finite number of vertices and a finite number of edges. Many sources assume that all graphs are finite without explicitly saying so. A graph is locally finite if each vertex has a finite number of incident edges. An infinite graph is a graph that is not finite: it has infinitely many vertices, infinitely many edges, or both.

The first order logic of graphs is a form of logic in which variables represent vertices of a graph, and there exists a binary predicate to test whether two vertices are adjacent. To be distinguished from second order logic, in which variables can also represent sets of vertices or edges.

For a set of vertices X, an X-flap is a connected component of the induced subgraph formed by deleting X. The flap terminology is commonly used in the context of havens, functions that map small sets of vertices to their flaps. See also the bridge of a cycle, which is either a flap of the cycle vertices or a chord of the cycle.

A forbidden graph characterization is a characterization of a family of graphs as being the graphs that do not have certain other graphs as subgraphs, induced subgraphs, or minors. If H is one of the graphs that does not occur as a subgraph, induced subgraph, or minor, then H is said to be forbidden.

A forcing graph is a graph H such that evaluating the subgraph density of H in the graphs of a graph sequence G(n) is sufficient to test whether that sequence is quasi-random.

A forest is an undirected graph without cycles (a disjoint union of unrooted trees), or a directed graph formed as a disjoint union of rooted trees.

2.  The Frucht graph, one of the two smallest cubic graphs with no nontrivial symmetries.

3.  Frucht's theorem that every finite group is the group of symmetries of a finite graph.

A functional graph is a directed graph where every vertex has out-degree one.  Equivalently, a functional graph is a maximal directed pseudoforest.

The genus of a graph is the minimum genus of a surface onto which it can be embedded; see embedding.

As a noun, a geodesic is a synonym for a shortest path. When used as an adjective, it means related to shortest paths or shortest path distances.

In the theory of random graphs, a giant component is a connected component that contains a constant fraction of the vertices of the graph. In standard models of random graphs, there is typically at most one giant component.

The girth of a graph is the length of its shortest cycle.

The fundamental object of study in graph theory, a system of vertices connected in pairs by edges. Often subdivided into directed graphs or undirected graphs according to whether the edges have an orientation or not. Mixed graphs include both types of edges.

Produced by a greedy algorithm. For instance, a greedy coloring of a graph is a coloring produced by considering the vertices in some sequence and assigning each vertex the first available color.

2.  The Grötzsch graph, the smallest triangle-free graph requiring four colors in any proper coloring.

3.  Grötzsch's theorem that triangle-free planar graphs can always be colored with at most three colors.

1.  The Grundy number of a graph is the maximum number of colors produced by a greedy coloring, with a badly-chosen vertex ordering.

A variable often used to denote a graph, especially when another graph has already been denoted by G.

An H-coloring of a graph G (where H is also a graph) is a homomorphism from H to G.

A graph is H-free if it does not have an induced subgraph isomorphic to H, that is, if H is a forbidden induced subgraph. The H-free graphs are the family of all graphs (or, often, all finite graphs) that are H-free. For instance the triangle-free graphs are the graphs that do not have a triangle graph as a subgraph. The property of being H-free is always hereditary. A graph is H-minor-free if it does not have a minor isomorphic to H.

2.  The Hadwiger number of a graph is the order of the largest complete minor of the graph. It is also called the contraction clique number or the homomorphism degree.

3.  The Hadwiger conjecture is the conjecture that the Hadwiger number is never less than the chromatic number.

A Hamiltonian path or Hamiltonian cycle is a simple spanning path or simple spanning cycle: it covers all of the vertices in the graph exactly once. A graph is Hamiltonian if it contains a Hamiltonian cycle, and traceable if it contains a Hamiltonian path.

A k-haven is a function that maps every set X of fewer than k vertices to one of its flaps, often satisfying additional consistency conditions. The order of a haven is the number k. Havens can be used to characterize the treewidth of finite graphs and the ends and Hadwiger numbers of infinite graphs.

1.  The height of a node in a rooted tree is the number of edges in a longest path, going away from the root (i.e. its nodes have strictly increasing depth), that starts at that node and ends at a leaf.

2.  The height of a rooted tree is the height of its root. That is, the height of a tree is the number of edges in a longest possible path, going away from the root, that starts at the root and ends at a leaf.

3.  The height of a directed acyclic graph is the maximum length of a directed path in this graph.

A hereditary property of graphs is a property that is closed under induced subgraphs: if G has a hereditary property, then so must every induced subgraph of G. Compare monotone (closed under all subgraphs) or minor-closed (closed under minors).

A simple cycle consisting of exactly six edges and six vertices.

A hole is an induced cycle of length four or more. An odd hole is a hole of odd length. An anti-hole is an induced subgraph of order four whose complement is a cycle; equivalently, it is a hole in the complement graph. This terminology is mainly used in the context of perfect graphs, which are characterized by the strong perfect graph theorem as being the graphs with no odd holes or odd anti-holes. The hole-free graphs are the same as the chordal graphs.

Two graphs are homomorphically equivalent if there exist two homomorphisms, one from each graph to the other graph.

1.  A graph homomorphism is a mapping from the vertex set of one graph to the vertex set of another graph that maps adjacent vertices to adjacent vertices. This type of mapping between graphs is the one that is most commonly used in category-theoretic approaches to graph theory. A proper graph coloring can equivalently be described as a homomorphism to a complete graph.

2.  The homomorphism degree of a graph is a synonym for its Hadwiger number, the order of the largest clique minor.

A directed hyperedge having a source and target set.

An edge in a hypergraph, having any number of endpoints, in contrast to the requirement that edges of graphs have exactly two endpoints.

A hypercube graph is a graph formed from the vertices and edges of a geometric hypercube.

A hypergraph is a generalization of a graph in which each edge (called a hyperedge in this context) may have more than two endpoints.

This prefix, in combination with a graph property, indicates a graph that does not have the property but such that every subgraph formed by deleting a single vertex does have the property. For instance, a hypohamiltonian graph is one that does not have a Hamiltonian cycle, but for which every one-vertex deletion produces a Hamiltonian subgraph. Compare critical, used for graphs which have a property but for which every one-vertex deletion does not.

The number of incoming edges in a directed graph; see degree.

An incidence in a graph is a vertex-edge pair such that the vertex is an endpoint of the edge.

The incidence matrix of a graph is a matrix whose rows are indexed by vertices of the graph, and whose columns are indexed by edges, with a one in the cell for row i and column j when vertex i and edge j are incident, and a zero otherwise.

The relation between an edge and one of its endpoints.

An incomparability graph is the complement of a comparability graph; see comparability.

1.  An independent set is a set of vertices that induces an edgeless subgraph. It may also be called a stable set or a coclique. The independence number α(G) is the size of the maximum independent set.

2.  In the graphic matroid of a graph, a subset of edges is independent if the corresponding subgraph is a tree or forest. In the bicircular matroid, a subset of edges is independent if the corresponding subgraph is a pseudoforest.

An indifference graph is another name for a proper interval graph or unit interval graph; see proper.

An induced subgraph or full subgraph of a graph is a subgraph formed from a subset of vertices and from all of the edges that have both endpoints in the subset. Special cases include induced paths and induced cycles, induced subgraphs that are paths or cycles.

An infinite graph is one that is not finite; see finite.

A vertex of a path or tree is internal if it is not a leaf; that is, if its degree is greater than one. Two paths are internally disjoint (some people call it independent) if they do not have any vertex in common, except the first and last ones.

1.  The intersection of two graphs is their largest common subgraph, the graph formed by the vertices and edges that belong to both graphs.

2.  An intersection graph is a graph whose vertices correspond to sets or geometric objects, with an edge between two vertices exactly when the corresponding two sets or objects have a nonempty intersection. Several classes of graphs may be defined as the intersection graphs of certain types of objects, for instance chordal graphs (intersection graphs of subtrees of a tree), circle graphs (intersection graphs of chords of a circle), interval graphs (intersection graphs of intervals of a line), line graphs (intersection graphs of the edges of a graph), and clique graphs (intersection graphs of the maximal cliques of a graph). Every graph is an intersection graph for some family of sets, and this family is called an intersection representation of the graph. The intersection number of a graph G is the minimum total number of elements in any intersection representation of G.

1.  An interval graph is an intersection graph of intervals of a line.

2.  The interval [u, v] in a graph is the union of all shortest paths from u to v.

An arrow with an opposite direction compared to another arrow. The arrow (y, x) is the inverted arrow of the arrow (x, y).

An isolated vertex of a graph is a vertex whose degree is zero, that is, a vertex with no incident edges.

Two graphs are isomorphic if there is an isomorphism between them; see isomorphism.

A graph isomorphism is a one-to-one incidence preserving correspondence of the vertices and edges of one graph to the vertices and edges of another graph. Two graphs related in this way are said to be isomorphic.

Synonym for bridge, in the sense of an edge whose removal disconnects the graph.

The join of two graphs is formed from their disjoint union by adding an edge from each vertex of one graph to each vertex of the other. Equivalently, it is the complement of the disjoint union of the complements.

For the notation for complete graphs, complete bipartite graphs, and complete multipartite graphs, see complete.

κ(G) (using the Greek letter kappa) can refer to the vertex connectivity of G or to the clique number of G.

A kernel of a directed graph is a set of vertices which is both stable and absorbing.

An inescapable section of a directed graph. See knot (mathematics) and knot theory.

1.  Information associated with a vertex or edge of a graph. A labeled graph is a graph whose vertices or edges have labels. The terms vertex-labeled or edge-labeled may be used to specify which objects of a graph have labels. Graph labeling refers to several different problems of assigning labels to graphs subject to certain constraints. See also graph coloring, in which the labels are interpreted as colors.

2.  In the context of graph enumeration, the vertices of a graph are said to be labeled if they are all distinguishable from each other. For instance, this can be made to be true by fixing a one-to-one correspondence between the vertices and the integers from 1 to the order of the graph. When vertices are labeled, graphs that are isomorphic to each other (but with different vertex orderings) are counted as separate objects. In contrast, when the vertices are unlabeled, graphs that are isomorphic to each other are not counted separately.

1.  A leaf vertex or pendant vertex (especially in a tree) is a vertex whose degree is 1. A leaf edge or pendant edge is the edge connecting a leaf vertex to its single neighbour.

2.  A leaf power of a tree is a graph whose vertices are the leaves of the tree and whose edges connect leaves whose distance in the tree is at most a given threshold.

In an unweighted graph, the length of a cycle, path, or walk is the number of edges it uses. In a weighted graph, it may instead be the sum of the weights of the edges that it uses. Length is used to define the shortest path, girth (shortest cycle length), and longest path between two vertices in a graph.

1.  This is the depth of a node plus 1, although some define it instead to be synonym of depth. A node's level in a rooted tree is the number of nodes in the path from the root to the node. For instance, the root has level 1 and any one of its adjacent nodes has level 2.

2.  A set of all node having the same level or depth.

A synonym for an undirected edge. The line graph L(G) of a graph G is a graph with a vertex for each edge of G and an edge for each pair of edges that share an endpoint in G.

1.  An adjacency list is a computer representation of graphs for use in graph algorithms.

2.  List coloring is a variation of graph coloring in which each vertex has a list of available colors.

A local property of a graph is a property that is determined only by the neighbourhoods of the vertices in the graph. For instance, a graph is locally finite if all of its neighborhoods are finite.

A loop or self-loop is an edge both of whose endpoints are the same vertex. It forms a cycle of length 1. These are not allowed in simple graphs.

A matching is a set of edges in which no two share any vertex. A vertex is matched or saturated if it is one of the endpoints of an edge in the matching. A perfect matching or complete matching is a matching that matches every vertex; it may also be called a 1-factor, and can only exist when the order is even. A near-perfect matching, in a graph with odd order, is one that saturates all but one vertex. A maximum matching is a matching that uses as many edges as possible; the matching number α′(G) of a graph G is the number of edges in a maximum matching. A maximal matching is a matching to which no additional edges can be added.

1.  A subgraph of given graph G is maximal for a particular property if it has that property but no other supergraph of it that is also a subgraph of G also has the same property. That is, it is a maximal element of the subgraphs with the property. For instance, a maximal clique is a complete subgraph that cannot be expanded to a larger complete subgraph. The word "maximal" should be distinguished from "maximum": a maximum subgraph is always maximal, but not necessarily vice versa.

2.  A simple graph with a given property is maximal for that property if it is not possible to add any more edges to it (keeping the vertex set unchanged) while preserving both the simplicity of the graph and the property. Thus, for instance, a maximal planar graph is a planar graph such that adding any more edges to it would create a non-planar graph.

A subgraph of a given graph G is maximum for a particular property if it is the largest subgraph (by order or size) among all subgraphs with that property. For instance, a maximum clique is any of the largest cliques in a given graph.

1.  A median of a triple of vertices, a vertex that belongs to shortest paths between all pairs of vertices, especially in median graphs and modular graphs.

2.  A median graph is a graph in which every three vertices have a unique median.

2.  A Meyniel graph is a graph in which every odd cycle of length five or more has at least two chords.

A subgraph of given graph is minimal for a particular property if it has that property but no proper subgraph of it also has the same property. That is, it is a minimal element of the subgraphs with the property.

A cut whose cut-set has minimum total weight, possibly restricted to cuts that separate a designated pair of vertices; they are characterized by the max-flow min-cut theorem.

A graph H is a minor of another graph G if H can be obtained by deleting edges or vertices from G and contracting edges in G. It is a shallow minor if it can be formed as a minor in such a way that the subgraphs of G that were contracted to form vertices of H all have small diameter. H is a topological minor of G if G has a subgraph that is a subdivision of H. A graph is H-minor-free if it does not have H as a minor. A family of graphs is minor-closed if it is closed under minors; the Robertson–Seymour theorem characterizes minor-closed families as having a finite set of forbidden minors.

A mixed graph is a graph that may include both directed and undirected edges.

1.  Modular graph, a graph in which each triple of vertices has at least one median vertex that belongs to shortest paths between all pairs of the triple.

2.  Modular decomposition, a decomposition of a graph into subgraphs within which all vertices connect to the rest of the graph in the same way.

3.  Modularity of a graph clustering, the difference of the number of cross-cluster edges from its expected value.

A monotone property of graphs is a property that is closed under subgraphs: if G has a monotone property, then so must every subgraph of G. Compare hereditary (closed under induced subgraphs) or minor-closed (closed under minors).

A Moore graph is a regular graph for which the Moore bound is met exactly. The Moore bound is an inequality relating the degree, diameter, and order of a graph, proved by Edward F. Moore. Every Moore graph is a cage.

A multigraph is a graph that allows multiple adjacencies (and, often, self-loops); a graph that is not required to be simple.

A multiple adjacency or multiple edge is a set of more than one edge that all have the same endpoints (in the same direction, in the case of directed graphs). A graph with multiple edges is often called a multigraph.

The multiplicity of an edge is the number of edges in a multiple adjacency. The multiplicity of a graph is the maximum multiplicity of any of its edges.

1.  For the notation for open and closed neighborhoods, see neighbourhood.

2.  A lower-case n is often used (especially in computer science) to denote the number of vertices in a given graph.

The open neighbourhood (or neighborhood) of a vertex v is the subgraph induced by all vertices that are adjacent to v. The closed neighbourhood is defined in the same way but also includes v itself. The open neighborhood of v in G may be denoted NG(v) or N(v), and the closed neighborhood may be denoted NG[v] or N[v]. When the openness or closedness of a neighborhood is not specified, it is assumed to be open.

A graph in which attributes (e.g. names) are associated with the nodes and/or edges.

A non-edge or anti-edge is a pair of vertices that are not adjacent; the edges of the complement graph.

1.  An odd cycle is a cycle whose length is odd. The odd girth of a non-bipartite graph is the length of its shortest odd cycle. An odd hole is a special case of an odd cycle: one that is induced and has four or more vertices.

2.  An odd vertex is a vertex whose degree is odd. By the handshaking lemma every finite undirected graph has an even number of odd vertices.

3.  An odd ear is a simple path or simple cycle with an odd number of edges, used in odd ear decompositions of factor-critical graphs; see ear.

4.  An odd chord is an edge connecting two vertices that are an odd distance apart in an even cycle. Odd chords are used to define strongly chordal graphs.

5.  An odd graph is a special case of a Kneser graph, having one vertex for each (n − 1)-element subset of a (2n − 1)-element set, and an edge connecting two subsets when their corresponding sets are disjoint.

1.  The order of a graph G is the number of its vertices, |V(G)|. The variable n is often used for this quantity. See also size, the number of edges.

2.  A type of logic of graphs; see first order and second order.

3.  An order or ordering of a graph is an arrangement of its vertices into a sequence, especially in the context of topological ordering (an order of a directed acyclic graph in which every edge goes from an earlier vertex to a later vertex in the order) and degeneracy ordering (an order in which each vertex has minimum degree in the induced subgraph of it and all later vertices).

4.  For the order of a haven or bramble, see haven and bramble.

1.  An orientation of an undirected graph is an assignment of directions to its edges, making it into a directed graph. An oriented graph is one that has been assigned an orientation. So, for instance, a polytree is an oriented tree; it differs from a directed tree (an arborescence) in that there is no requirement of consistency in the directions of its edges. Other special types of orientation include tournaments, orientations of complete graphs; strong orientations, orientations that are strongly connected; acyclic orientations, orientations that are acyclic; Eulerian orientations, orientations that are Eulerian; and transitive orientations, orientations that are transitively closed.

2.  Oriented graph, used by some authors as a synonym for a directed graph.

An outerplanar graph is a graph that can be embedded in the plane (without crossings) so that all vertices are on the outer face of the graph.

In a rooted tree, a parent of a vertex v is a neighbor of v along the incoming edge, the one that is directed toward the root.

A path may either be a walk or a walk without repeated vertices and consequently edges (also called a simple path), depending on the source. Important special cases include induced paths and shortest paths.

A path decomposition of a graph G is a tree decomposition whose underlying tree is a path. Its width is defined in the same way as for tree decompositions, as one less than the size of the largest bag. The minimum width of any path decomposition of G is the pathwidth of G.

The pathwidth of a graph G is the minimum width of a path decomposition of G. It may also be defined in terms of the clique number of an interval completion of G. It is always between the bandwidth and the treewidth of G. It is also known as interval thickness, vertex separation number, or node searching number.

1.  A perfect graph is a graph in which, in every induced subgraph, the chromatic number equals the clique number. The perfect graph theorem and strong perfect graph theorem are two theorems about perfect graphs, the former proving that their complements are also perfect and the latter proving that they are exactly the graphs with no odd holes or anti-holes.

2.  A perfectly orderable graph is a graph whose vertices can be ordered in such a way that a greedy coloring algorithm with this ordering optimally colors every induced subgraph. The perfectly orderable graphs are a subclass of the perfect graphs.

3.  A perfect matching is a matching that saturates every vertex; see matching.

4.  A perfect 1-factorization is a partition of the edges of a graph into perfect matchings so that each two matchings form a Hamiltonian cycle.

1.  A peripheral cycle or non-separating cycle is a cycle with at most one bridge.

2.  A peripheral vertex is a vertex whose eccentricity is maximum. In a tree, this must be a leaf.

1.  Julius Petersen (1839–1910), Danish graph theorist.

2.  The Petersen graph, a 10-vertex 15-edge graph frequently used as a counterexample.

3.  Petersen's theorem that every bridgeless cubic graph has a perfect matching.

A planar graph is a graph that has an embedding onto the Euclidean plane. A plane graph is a planar graph for which a particular embedding has already been fixed. A k-planar graph is one that can be drawn in the plane with at most k crossings per edge.

A polytree is an oriented tree; equivalently, a directed acyclic graph whose underlying undirected graph is a tree.

1.  A graph power Gk of a graph G is another graph on the same vertex set; two vertices are adjacent in Gk when they are at distance at most k in G. A leaf power is a closely related concept, derived from a power of a tree by taking the subgraph induced by the tree's leaves.

2.  Power graph analysis is a method for analyzing complex networks by identifying cliques, bicliques, and stars within the network.

3.  Power laws in the degree distributions of scale-free networks are a phenomenon in which the number of vertices of a given degree is proportional to a power of the degree.

A vertex coming before a given vertex in a directed path.

1.  A prime graph is defined from an algebraic group, with a vertex for each prime number that divides the order of the group.

2.  In the theory of modular decomposition, a prime graph is a graph without any nontrivial modules.

3.  In the theory of splits, cuts whose cut-set is a complete bipartite graph, a prime graph is a graph without any splits. Every quotient graph of a maximal decomposition by splits is a prime graph, a star, or a complete graph.

4.  A prime graph for the Cartesian product of graphs is a connected graph that is not itself a product. Every connected graph can be uniquely factored into a Cartesian product of prime graphs.

1.  A proper subgraph is a subgraph that removes at least one vertex or edge relative to the whole graph; for finite graphs, proper subgraphs are never isomorphic to the whole graph, but for infinite graphs they can be.

2.  A proper coloring is an assignment of colors to the vertices of a graph (a coloring) that assigns different colors to the endpoints of each edge; see color.

3.  A proper interval graph or proper circular arc graph is an intersection graph of a collection of intervals or circular arcs (respectively) such that no interval or arc contains another interval or arc. Proper interval graphs are also called unit interval graphs (because they can always be represented by unit intervals) or indifference graphs.

A graph property is something that can be true of some graphs and false of others, and that depends only on the graph structure and not on incidental information such as labels. Graph properties may equivalently be described in terms of classes of graphs (the graphs that have a given property). More generally, a graph property may also be a function of graphs that is again independent of incidental information, such as the size, order, or degree sequence of a graph; this more general definition of a property is also called an invariant of the graph.

A pseudoforest is an undirected graph in which each connected component has at most one cycle, or a directed graph in which each vertex has at most one outgoing edge.

A pseudograph is a graph or multigraph that allows self-loops.

A quasi-line graph or locally co-bipartite graph is a graph in which the open neighborhood of every vertex can be partitioned into two cliques. These graphs are always claw-free and they include as a special case the line graphs. They are used in the structure theory of claw-free graphs.

A quasi-random graph sequence is a sequence of graphs that shares several properties with a sequence of random graphs generated according to the Erdős–Rényi random graph model.

A quiver is a directed multigraph, as used in category theory. The edges of a quiver are called arrows.

The radius of a graph is the minimum eccentricity of any vertex.

A Ramanujan graph is a graph whose spectral expansion is as large as possible. That is, it is a d-regular graph, such that the second-largest eigenvalue of its adjacency matrix is at most

A ray, in an infinite graph, is an infinite simple path with exactly one endpoint. The ends of a graph are equivalence classes of rays.

The ability to get from one vertex to another within a graph.

Has an affirmative reachability. A vertex y is said to be reachable from a vertex x if there exists a path from x to y.

In the context of the reconstruction conjecture, a graph property is recognizable if its truth can be determined from the deck of the graph. Many graph properties are known to be recognizable. If the reconstruction conjecture is true, all graph properties are recognizable.

The reconstruction conjecture states that each undirected graph G is uniquely determined by its deck, a multiset of graphs formed by removing one vertex from G in all possible ways. In this context, reconstruction is the formation of a graph from its deck.

A simple cycle consisting of exactly four edges and four vertices.

A graph is d-regular when all of its vertices have degree d. A regular graph is a graph that is d-regular for some d.

A regular tournament is a tournament where in-degree equals out-degree for all vertices.

1.  A designated vertex in a graph, particularly in directed trees and rooted graphs.

2.  The inverse operation to a graph power: a kth root of a graph G is another graph on the same vertex set such that two vertices are adjacent in G if and only if they have distance at most k in the root.

The second order logic of graphs is a form of logic in which variables may represent vertices, edges, sets of vertices, and (sometimes) sets of edges. This logic includes predicates for testing whether a vertex and edge are incident, as well as whether a vertex or edge belongs to a set. To be distinguished from first order logic, in which variables can only represent vertices.

Vertex separation number is a synonym for pathwidth.

In a rooted tree, a sibling of a vertex v is a vertex which has the same parent vertex as  v.

1.  A simple graph is a graph without loops and without multiple adjacencies. That is, each edge connects two distinct endpoints and no two edges have the same endpoints. A simple edge is an edge that is not part of a multiple adjacency. In many cases, graphs are assumed to be simple unless specified otherwise.

2.  A simple path or a simple cycle is a path or cycle that has no repeated vertices and consequently no repeated edges.

A sink, in a directed graph, is a vertex with no outgoing edges (out-degree equals 0).

The size of a graph G is the number of its edges, |E(G)|. The variable m is often used for this quantity. See also order, the number of vertices.

A small-world network is a graph in which most nodes are not neighbors of one another, but most nodes can be reached from every other node by a small number of hops or steps.  Specifically, a small-world network is defined to be a graph where the typical distance L between two randomly chosen nodes (the number of steps required) grows proportionally to the logarithm of the number of nodes N in the network

A snark is a simple, connected, bridgeless cubic graph with chromatic index equal to 4.

A source, in a directed graph, is a vertex with no incoming edges (in-degree equals 0).

In algebraic graph theory, several vector spaces over the binary field may be associated with a graph. Each has sets of edges or vertices for its vectors, and symmetric difference of sets as its vector sum operation. The edge space is the space of all sets of edges, and the vertex space is the space of all sets of vertices. The cut space is a subspace of the edge space that has the cut-sets of the graph as its elements. The cycle space has the Eulerian spanning subgraphs as its elements.

A spanner is a (usually sparse) graph whose shortest path distances approximate those in a dense graph or other metric space. Variations include geometric spanners, graphs whose vertices are points in a geometric space; tree spanners, spanning trees of a graph whose distances approximate the graph distances, and graph spanners, sparse subgraphs of a dense graph whose distances approximate the original graph's distances. A greedy spanner is a graph spanner constructed by a greedy algorithm, generally one that considers all edges from shortest to longest and keeps the ones that are needed to preserve the distance approximation.

A subgraph is spanning when it includes all of the vertices of the given graph.

Important cases include spanning trees, spanning subgraphs that are trees, and perfect matchings, spanning subgraphs that are matchings. A spanning subgraph may also be called a factor, especially (but not only) when it is regular.

A sparse graph is one that has few edges relative to its number of vertices. In some definitions the same property should also be true for all subgraphs of the given graph.

The spectrum of a graph is the collection of eigenvalues of its adjacency matrix. Spectral graph theory is the branch of graph theory that uses spectra to analyze graphs. See also spectral expansion.

1.  A split graph is a graph whose vertices can be partitioned into a clique and an independent set. A related class of graphs, the double split graphs, are used in the proof of the strong perfect graph theorem.

2.  A split of an arbitrary graph is a partition of its vertices into two nonempty subsets, such that the edges spanning this cut form a complete bipartite subgraph. The splits of a graph can be represented by a tree structure called its split decomposition. A split is called a strong split when it is not crossed by any other split. A split is called nontrivial when both of its sides have more than one vertex. A graph is called prime when it has no nontrivial splits.

3.  Vertex splitting (sometimes called vertex cleaving) is an elementary graph operation that splits a vertex into two, where these two new vertices are adjacent to the vertices that the original vertex was adjacent to. The inverse of  vertex splitting is vertex contraction.

1.  The square of a graph G is the graph power G2; in the other direction, G is the square root of G2. The half-square of a bipartite graph is the subgraph of its square induced by one side of the bipartition.

2.  A squaregraph is a planar graph that can be drawn so that all bounded faces are 4-cycles and all vertices of degree ≤ 3 belong to the outer face.

3.  A square grid graph is a lattice graph defined from points in the plane with integer coordinates connected by unit-length edges.

A star is a tree with one internal vertex; equivalently, it is a complete bipartite graph K1,n for some n ≥ 2. The special case of a star with three leaves is called a claw.

The strength of a graph is the minimum ratio of the number of edges removed from the graph to components created, over all possible removals; it is analogous to toughness, based on vertex removals.

1.  For strong connectivity and strongly connected components of directed graphs, see connected and component. A strong orientation is an orientation that is strongly connected; see orientation.

2.  For the strong perfect graph theorem, see perfect.

3.  A strongly regular graph is a regular graph in which every two adjacent vertices have the same number of shared neighbours and every two non-adjacent vertices have the same number of shared neighbours.

4.  A strongly chordal graph is a chordal graph in which every even cycle of length six or more has an odd chord.

5.  A strongly perfect graph is a graph in which every induced subgraph has an independent set meeting all maximal cliques. The Meyniel graphs are also called "very strongly perfect graphs" because in them, every vertex belongs to such an independent set.

A subgraph of a graph G is another graph formed from a subset of the vertices and edges of G. The vertex subset must include all endpoints of the edge subset, but may also include additional vertices. A spanning subgraph is one that includes all vertices of the graph; an induced subgraph is one that includes all the edges whose endpoints belong to the vertex subset.

A subtree is a connected subgraph of a tree. Sometimes, for rooted trees, subtrees are defined to be a special type of connected subgraph, formed by all vertices and edges reachable from a chosen vertex.

A vertex coming after a given vertex in a directed path.

A superconcentrator is a graph with two designated and equal-sized subsets of vertices I and O, such that for every two equal-sized subsets S of I and T of O there exists a family of disjoint paths connecting every vertex in S to a vertex in T. Some sources require in addition that a superconcentrator be a directed acyclic graph, with I as its sources and O as its sinks.

A graph formed by adding vertices, edges, or both to a given graph. If H is a subgraph of G, then G is a supergraph of H.

1.  A theta graph is the union of three internally disjoint (simple) paths that have the same two distinct end vertices.

2.  The theta graph of a collection of points in the Euclidean plane is constructed by constructing a system of cones surrounding each point and adding one edge per cone, to the point whose projection onto a central ray of the cone is smallest.

3.  The Lovász number or Lovász theta function of a graph is a graph invariant related to the clique number and chromatic number that can be computed in polynomial time by semidefinite programming.

The Thomsen graph is a name for the complete bipartite graph

1.  A topological graph is a representation of the vertices and edges of a graph by points and curves in the plane (not necessarily avoiding crossings).

2.  Topological graph theory is the study of graph embeddings.

3.  Topological sorting is the algorithmic problem of arranging a directed acyclic graph into a topological order, a vertex sequence such that each edge goes from an earlier vertex to a later vertex in the sequence.

A closed trail, a walk that starts and ends at the same vertex and has no repeated edges. Euler tours are tours that use all of the graph edges; see Eulerian.

A tournament is an orientation of a complete graph; that is, it is a directed graph such that every two vertices are connected by exactly one directed edge (going in only one of the two directions between the two vertices).

A traceable graph is a graph that contains a Hamiltonian path.

Having to do with the transitive property. The transitive closure of a given directed graph is a graph on the same vertex set that has an edge from one vertex to another whenever the original graph has a path connecting the same two vertices. A transitive reduction of a graph is a minimal graph having the same transitive closure; directed acyclic graphs have a unique transitive reduction. A transitive orientation is an orientation of a graph that is its own transitive closure; it exists only for comparability graphs.

The transpose graph of a given directed graph is a graph on the same vertices, with each edge reversed in direction. It may also be called the converse or reverse of the graph.

1.  A tree is an undirected graph that is both connected and acyclic, or a directed graph in which there exists a unique walk from one vertex (the root of the tree) to all remaining vertices.

2.  A k-tree is a graph formed by gluing (k + 1)-cliques together on shared k-cliques. A tree in the ordinary sense is a 1-tree according to this definition.

A tree decomposition of a graph G is a tree whose nodes are labeled with sets of vertices of G; these sets are called bags. For each vertex v, the bags that contain v must induce a subtree of the tree, and for each edge uv there must exist a bag that contains both u and v. The width of a tree decomposition is one less than the maximum number of vertices in any of its bags; the treewidth of G is the minimum width of any tree decomposition of G.

The treewidth of a graph G is the minimum width of a tree decomposition of G. It can also be defined in terms of the clique number of a chordal completion of G, the order of a haven of G, or the order of a bramble of G.

A cycle of length three in a graph. A triangle-free graph is an undirected graph that does not have any triangle subgraphs.

A trivial graph is a graph with 0 or 1 vertices. A graph with 0 vertices is also called null graph.

2.  A Turán graph is a balanced complete multipartite graph.

3.  Turán's theorem states that Turán graphs have the maximum number of edges among all clique-free graphs of a given order.

4.  Turán's brick factory problem asks for the minimum number of crossings in a drawing of a complete bipartite graph.

Two vertices u,v are true twins if they have the same closed neighborhood: NG[u] = NG[v] (this implies u and v are neighbors), and they are false twins if they have the same open neighborhood: NG(u) = NG(v)) (this implies u and v are not neighbors).

In a rooted tree, a unary vertex is a vertex which has exactly one child vertex.

An undirected graph is a graph in which the two endpoints of each edge are not distinguished from each other. See also directed and mixed. In a mixed graph, an undirected edge is again one in which the endpoints are not distinguished from each other.

A hypergraph is k-uniform when all its edges have k endpoints, and uniform when it is k-uniform for some k. For instance, ordinary graphs are the same as 2-uniform hypergraphs.

1.  A universal graph is a graph that contains as subgraphs all graphs in a given family of graphs, or all graphs of a given size or order within a given family of graphs.

2.  A universal vertex (also called an apex or dominating vertex) is a vertex that is adjacent to every other vertex in the graph. For instance, wheel graphs and connected threshold graphs always have a universal vertex.

3.  In the logic of graphs, a vertex that is universally quantified in a formula may be called a universal vertex for that formula.

A graph whose vertices and edges have not been assigned weights; the opposite of a weighted graph.

The utility graph is a name for the complete bipartite graph

A vertex (plural vertices) is (together with edges) one of the two basic units out of which graphs are constructed. Vertices of graphs are often considered to be atomic objects, with no internal structure.

A set of vertices whose removal disconnects the graph. A one-vertex cut is called an articulation point or cut vertex.

The set of vertices of a given graph G, sometimes denoted by V(G).

2.  Vizing's theorem that the chromatic index is at most one more than the maximum degree.

3.  Vizing's conjecture on the domination number of Cartesian products of graphs.

The letter W is used in notation for wheel graphs and windmill graphs. The notation is not standardized.

2.  The Wagner graph, an eight-vertex Möbius ladder.

3.  Wagner's theorem characterizing planar graphs by their forbidden minors.

4.  Wagner's theorem characterizing the K5-minor-free graphs.

A walk is a finite or infinite sequence of edges which joins a sequence of vertices. Walks are also sometimes called chains. A walk is open if its first and last vertices are distinct, and closed if they are repeated.

A directed graph is called weakly connected if replacing all of its directed edges with undirected edges produces a connected (undirected) graph.

A numerical value, assigned as a label to a vertex or edge of a graph. The weight of a subgraph is the sum of the weights of the vertices or edges within that subgraph.

A graph whose vertices or edges have been assigned weights.  A vertex-weighted graph has weights on its vertices and an edge-weighted graph has weights on its edges.

A well-colored graph is a graph all of whose greedy colorings use the same number of colors.

A well-covered graph is a graph all of whose maximal independent sets are the same size.

A wheel graph is a graph formed by adding a universal vertex to a simple cycle.

2.  For other graph invariants known as width, see bandwidth, branchwidth, clique-width, pathwidth, and treewidth.

3.  The width of a tree decomposition or path decomposition is one less than the maximum size of one of its bags, and may be used to define treewidth and pathwidth.

4.  The width of a directed acyclic graph is the maximum cardinality of an antichain.

A windmill graph is the union of a collection of cliques, all of the same order as each other, with one shared vertex belonging to all the cliques and all other vertices and edges distinct.
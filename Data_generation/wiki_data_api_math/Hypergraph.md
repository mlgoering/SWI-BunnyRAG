# Hypergraph

In mathematics, a hypergraph is a generalization of a graph in which an edge can join any number of vertices. In contrast, in an ordinary graph, an edge connects exactly two vertices.

is a set of elements called nodes, vertices, points, or elements and

. The size of the hypergraph is the number of edges in

: that is, the number of vertices in its tail followed by the number of vertices in its head.

The definition above generalizes from a directed graph to a directed hypergraph by defining the head or tail of each edge as a set of vertices (

) rather than as a single vertex. A graph is then the special case where each of these sets contains only one element. Hence any standard graph theoretic concept that is independent of the edge orders

is an undirected graph whose edges connect not just two vertices, but an arbitrary number. An undirected hypergraph is also called a set system or a family of sets drawn from the set of elements

Hypergraphs can be viewed as incidence structures. In particular, there is a bipartite "incidence graph" or "Levi graph" corresponding to every hypergraph, and conversely, every bipartite graph can be regarded as the incidence graph of a hypergraph when it is 2-colored and it is indicated which color class corresponds to hypergraph vertices and which to hypergraph edges.

Hypergraphs have many other names. In computational geometry, an undirected hypergraph may sometimes be called a range space and then the hyperedges are called ranges.

In cooperative game theory, hypergraphs are called simple games (voting games); this notion is applied to solve problems in social choice theory. In some literature edges are referred to as hyperlinks or connectors.

The collection of hypergraphs is a category with hypergraph homomorphisms as morphisms.

Undirected hypergraphs are useful in modelling such things as satisfiability problems, databases, machine learning, and Steiner tree problems. They have been extensively used in machine learning tasks as the data model and classifier regularization. The applications include recommender system (communities as hyperedges), image retrieval (correlations as hyperedges), and bioinformatics (biochemical interactions as hyperedges). Representative hypergraph learning techniques include hypergraph spectral clustering that extends the spectral graph theory with hypergraph Laplacian, and hypergraph semi-supervised learning that introduces extra hypergraph structural cost to restrict the learning results. For large scale hypergraphs, a distributed framework built using Apache Spark is also available. It can be desirable to study hypergraphs where all hyperedges have the same cardinality; a k-uniform hypergraph is a hypergraph such that all its hyperedges have size k. (In other words, one such hypergraph is a collection of sets, each such set a hyperedge connecting k nodes.) So a 2-uniform hypergraph is a graph, a 3-uniform hypergraph is a collection of unordered triples, and so on.

Directed hypergraphs can be used to model things including telephony applications, detecting money laundering, operations research, and transportation planning. They can also be used to model Horn-satisfiability.

Many theorems and concepts involving graphs also hold for hypergraphs, in particular:

Vertex cover in hypergraphs (also known as: transversal);

Hypergraph grammar - created by augmenting a class of hypergraphs with a set of replacement rules;

In directed hypergraphs: transitive closure, and shortest path problems.

Although hypergraphs are more difficult to draw on paper than graphs, several researchers have studied methods for the visualization of hypergraphs.

In one possible visual representation for hypergraphs, similar to the standard graph drawing style in which curves in the plane are used to depict graph edges, a hypergraph's vertices are depicted as points, disks, or boxes, and its hyperedges are depicted as trees that have the vertices as their leaves. If the vertices are represented as points, the hyperedges may also be shown as smooth curves that connect sets of points, or as simple closed curves that enclose sets of points.

In another style of hypergraph visualization, the subdivision model of hypergraph drawing, the plane is subdivided into regions, each of which represents a single vertex of the hypergraph. The hyperedges of the hypergraph are represented by contiguous subsets of these regions, which may be indicated by coloring, by drawing outlines around them, or both. An order-n Venn diagram, for instance, may be viewed as a subdivision drawing of a hypergraph with n hyperedges (the curves defining the diagram) and 2n − 1 vertices (represented by the regions into which these curves subdivide the plane). In contrast with the polynomial-time recognition of planar graphs, it is NP-complete to determine whether a hypergraph has a planar subdivision drawing, but the existence of a drawing of this type may be tested efficiently when the adjacency pattern of the regions is constrained to be a path, cycle, or tree.

An alternative representation of the hypergraph called PAOH is shown in the figure on top of this article. Edges are vertical lines connecting vertices. Vertices are aligned on the left. The legend on the right shows the names of the edges. It has been designed for dynamic hypergraphs but can be used for simple hypergraphs as well.

Classic hypergraph coloring is assigning one of the colors from set

to every vertex of a hypergraph in such a way that each hyperedge contains at least two vertices of distinct colors. In other words, there must be no monochromatic hyperedge with cardinality at least 2. In this sense it is a direct generalization of graph coloring. The minimum number of used distinct colors over all colorings is called the chromatic number of a hypergraph.

Hypergraphs for which there exists a coloring using up to k colors are referred to as k-colorable. The 2-colorable hypergraphs are exactly the bipartite ones.

There are many generalizations of classic hypergraph coloring. One of them is the so-called mixed hypergraph coloring, when monochromatic edges are allowed. Some mixed hypergraphs are uncolorable for any number of colors. A general criterion for uncolorability is unknown. When a mixed hypergraph is colorable, then the minimum and maximum number of used colors are called the lower and upper chromatic numbers respectively.

Non-simple (or multiple) - has loops (hyperedges with a single vertex) or repeated edges, which means there can be two or more edges containing the same set of vertices.

2-colorable - its vertices can be partitioned into two classes U and V in such a way that each hyperedge with cardinality at least 2 contains at least one vertex from both classes. An alternative term is Property B.

Two stronger properties are bipartite and balanced.

parts, and each hyperedge contains precisely one vertex of each type.

Reduced: no hyperedge is a strict subset of another hyperedge; equivalently, every hyperedge is maximal for inclusion. The reduction of a hypergraph is the reduced hypergraph obtained by removing every hyperedge which is included in another hyperedge.

Downward-closed - every subset of an undirected hypergraph's edges is a hyperedge too. A downward-closed hypergraph is usually called an abstract simplicial complex. It is generally not reduced, unless all hyperedges have cardinality 1.

An abstract simplicial complex with the augmentation property is called a matroid.

Laminar: for any two hyperedges, either they are disjoint, or one is included in the other. In other words, the set of hyperedges forms a laminar set family.

Because hypergraph links can have any cardinality, there are several notions of the concept of a subgraph, called subhypergraphs, partial hypergraphs and section hypergraphs.

{\displaystyle X=\lbrace x_{i}\mid i\in I_{v}\rbrace ,}

{\displaystyle E=\lbrace e_{i}\mid i\in I_{e},e_{i}\subseteq X,e_{i}\neq \emptyset \rbrace ,}

are the index sets of the vertices and edges respectively.

A subhypergraph is a hypergraph with some vertices removed. Formally, the subhypergraph

{\displaystyle H_{A}=\left(A,\lbrace e\cap A\mid e\in E,e\cap A\neq \emptyset \rbrace \right).}

An extension of a subhypergraph is a hypergraph where each hyperedge of

{\displaystyle E'=\lbrace e\in E\mid e\subseteq (A\cup A')\rbrace }

The partial hypergraph is a hypergraph with some edges removed. Given a subset

of the edge index set, the partial hypergraph generated by

{\displaystyle \left(X,\lbrace e_{i}\mid i\in J\rbrace \right).}

{\displaystyle H\times A=\left(A,\lbrace e_{i}\mid i\in I_{e},e_{i}\subseteq A\rbrace \right).}

is a hypergraph whose vertices and edges are interchanged, so that the vertices are given by

{\displaystyle X_{m}=\lbrace e_{i}\mid x_{m}\in e_{i}\rbrace .}

When a notion of equality is properly defined, as done below, the operation of taking the dual of a hypergraph is an involution, i.e.,

A connected graph G with the same vertex set as a connected hypergraph H is a host graph for H if every hyperedge of H induces a connected subgraph in G. For a disconnected hypergraph H, G is a host graph if there is a bijection between the connected components of G and of H, such that each connected component G' of G is a host of the corresponding H'.

The 2-section (or clique graph, representing graph, primal graph, Gaifman graph) of a hypergraph is the graph with the same vertices of the hypergraph, and edges between all pairs of vertices contained in the same hyperedge.

{\displaystyle b_{ij}=\left\{{\begin{matrix}1&\mathrm {if} ~v_{i}\in e_{j}\\0&\mathrm {otherwise} .\end{matrix}}\right.}

{\displaystyle e_{i}^{*}\in E^{*},~v_{j}^{*}\in e_{i}^{*}}

For a directed hypergraph, the heads and tails of each hyperedge

{\displaystyle b_{ij}=\left\{{\begin{matrix}-1&\mathrm {if} ~v_{i}\in T(e_{j})\\1&\mathrm {if} ~v_{i}\in H(e_{j})\\0&\mathrm {otherwise} .\end{matrix}}\right.}

A hypergraph H may be represented by a bipartite graph BG as follows: the sets X and E are the parts of BG, and (x1, e1) are connected with an edge if and only if vertex x1 is contained in edge e1 in H.

Conversely, any bipartite graph with fixed parts and no unconnected nodes in the second part represents some hypergraph in the manner described above. This bipartite graph is also called incidence graph.

A parallel for the adjacency matrix of a hypergraph can be drawn from the adjacency matrix of a graph. In the case of a graph, the adjacency matrix is a square matrix which indicates whether pairs of vertices are adjacent. Likewise, we can define the adjacency matrix

{\displaystyle a_{ij}=\left\{{\begin{matrix}w_{e_{k}}&\mathrm {if} ~(v_{i},v_{j})\in E\\0&\mathrm {otherwise} .\end{matrix}}\right.}

In contrast with ordinary undirected graphs for which there is a single natural notion of cycles and acyclic graphs. For hypergraphs, there are multiple natural non-equivalent definitions of cycles which collapse to the ordinary notion of cycle when the graph case is considered.

A first notion of cycle was introduced by Claude Berge. A Berge cycle in a hypergraph is an alternating sequence of distinct vertices and edges

Under this definition a hypergraph is acyclic if and only if its incidence graph (the bipartite graph defined above) is acyclic. Thus Berge-cyclicity can obviously be tested in linear time by an exploration of the incidence graph.

-uniform hypergraphs, where all hyperedges are of size

This notion was introduced by Katona and Kierstead and has since garnered considerable attention, particularly in the study of Hamiltonicity in extremal combinatorics.

This corresponds to an approximate hypergraph-extension of the celebrated Dirac's theorem about Hamilton cycles in graphs.

The maximum number of hyperedges in a (tightly) acyclic

-uniform hypergraph remains unknown. The best known bounds, obtained by Sudakov and Tomon, show that every

hyperedges must contain a tight cycle. This bound is optimal up to the

vertices which are not contained in the previous edge,

The definition of Berge-acyclicity might seem to be very restrictive: for instance, if a hypergraph has some pair

We can define a weaker notion of hypergraph acyclicity, later termed α-acyclicity. This notion of acyclicity is equivalent to the hypergraph being conformal (every clique of the primal graph is covered by some hyperedge) and its primal graph being chordal; it is also equivalent to reducibility to the empty graph through the GYO algorithm (also known as Graham's algorithm), a confluent iterative process which removes hyperedges using a generalized definition of ears. In the domain of database theory, it is known that a database schema enjoys certain desirable properties if its underlying hypergraph is α-acyclic. Besides, α-acyclicity is also related to the expressiveness of the guarded fragment of first-order logic.

We can test in linear time if a hypergraph is α-acyclic.

Note that α-acyclicity has the counter-intuitive property that adding hyperedges to an α-cyclic hypergraph may make it α-acyclic (for instance, adding a hyperedge containing all vertices of the hypergraph will always make it α-acyclic). Motivated in part by this perceived shortcoming, Ronald Fagin defined the stronger notions of β-acyclicity and γ-acyclicity. We can state β-acyclicity as the requirement that all subhypergraphs of the hypergraph are α-acyclic, which is equivalent to an earlier definition by Graham. The notion of γ-acyclicity is a more restrictive condition which is equivalent to several desirable properties of database schemas and is related to Bachman diagrams. Both β-acyclicity and γ-acyclicity can be tested in polynomial time.

Those four notions of acyclicity are comparable: γ-acyclicity which implies β-acyclicity which implies α-acyclicity. Moreover, Berge-acyclicity implies all of them. None of the reverse implications hold including the Berge one. In other words, these four notions are different.

A hypergraph homomorphism is a map from the vertex set of one hypergraph to another such that each edge maps to one other edge.

is then called the isomorphism of the graphs. Note that

When the edges of a hypergraph are explicitly labeled, one has the additional notion of strong isomorphism. One says that

if the permutation is the identity. One then writes

. Note that all strongly isomorphic graphs are isomorphic, but not vice versa.

When the vertices of a hypergraph are explicitly labeled, one has the notions of equivalence, and also of equality. One says that

. Note that, with this definition of equality, graphs are self-dual:

A hypergraph automorphism is an isomorphism from a vertex set into itself, that is a relabeling of vertices. The set of automorphisms of a hypergraph H (= (X, E)) is a group under composition, called the automorphism group of the hypergraph and written Aut(H).

{\displaystyle H=\lbrace e_{1}=\lbrace a,b\rbrace ,e_{2}=\lbrace b,c\rbrace ,e_{3}=\lbrace c,d\rbrace ,e_{4}=\lbrace d,a\rbrace ,e_{5}=\lbrace b,d\rbrace ,e_{6}=\lbrace a,c\rbrace \rbrace }

{\displaystyle G=\lbrace f_{1}=\lbrace \alpha ,\beta \rbrace ,f_{2}=\lbrace \beta ,\gamma \rbrace ,f_{3}=\lbrace \gamma ,\delta \rbrace ,f_{4}=\lbrace \delta ,\alpha \rbrace ,f_{5}=\lbrace \alpha ,\gamma \rbrace ,f_{6}=\lbrace \beta ,\delta \rbrace \rbrace }

, etc.), but they are not strongly isomorphic. So, for example, in

{\displaystyle e_{1}\cap e_{4}\cap e_{6}=\lbrace a\rbrace }

, there does not exist any vertex that meets edges 1, 4 and 6:

{\displaystyle f_{1}\cap f_{4}\cap f_{6}=\varnothing }

is the maximum cardinality of any of the edges in the hypergraph. If all edges have the same cardinality k, the hypergraph is said to be uniform or k-uniform, or is called a k-hypergraph. A graph is just a 2-uniform hypergraph.

The degree d(v) of a vertex v is the number of edges that contain it. H is k-regular if every vertex has degree k.

The dual of a uniform hypergraph is regular and vice versa.

Two vertices x and y of H are called symmetric if there exists an automorphism such that

are said to be symmetric if there exists an automorphism such that

A hypergraph is said to be vertex-transitive (or vertex-symmetric) if all of its vertices are symmetric. Similarly, a hypergraph is edge-transitive if all edges are symmetric. If a hypergraph is both edge- and vertex-symmetric, then the hypergraph is simply transitive.

Because of hypergraph duality, the study of edge-transitivity is identical to the study of vertex-transitivity.

A partition theorem due to E. Dauber states that, for an edge-transitive hypergraph

{\displaystyle \sum _{k=1}^{K}r\left(H_{X_{k}}\right)=r(H)}

As a corollary, an edge-transitive hypergraph that is not vertex-transitive is bicolorable.

Graph partitioning (and in particular, hypergraph partitioning) has many applications to IC design and parallel computing. Efficient and scalable hypergraph partitioning algorithms are also important for processing large scale hypergraphs in machine learning tasks.

One possible generalization of a hypergraph is to allow edges to point at other edges. There are two variations of this generalization. In one, the edges consist not only of a set of vertices, but may also contain subsets of vertices, subsets of subsets of vertices and so on ad infinitum. In essence, every edge is just an internal node of a tree or directed acyclic graph, and vertices are the leaf nodes. A hypergraph is then just a collection of trees with common, shared nodes (that is, a given internal node or leaf may occur in several different trees). Conversely, every collection of trees can be understood as this generalized hypergraph. Since trees are widely used throughout computer science and many other branches of mathematics, one could say that hypergraphs appear naturally as well. So, for example, this generalization arises naturally as a model of term algebra; edges correspond to terms and vertices correspond to constants or variables.

For such a hypergraph, set membership then provides an ordering, but the ordering is neither a partial order nor a preorder, since it is not transitive. The graph corresponding to the Levi graph of this generalization is a directed acyclic graph. Consider, for example, the generalized hypergraph whose vertex set is

. However, the transitive closure of set membership for such hypergraphs does induce a partial order, and "flattens" the hypergraph into a partially ordered set.

Alternately, edges can be allowed to point at other edges, irrespective of the requirement that the edges be ordered as directed, acyclic graphs. This allows graphs with edge-loops, which need not contain vertices at all. For example, consider the generalized hypergraph consisting of two edges

. As this loop is infinitely recursive, sets that are the edges violate the axiom of foundation. In particular, there is no transitive closure of set membership for such hypergraphs. Although such structures may seem strange at first, they can be readily understood by noting that the equivalent generalization of their Levi graph is no longer bipartite, but is rather just some general directed graph.

The generalized incidence matrix for such hypergraphs is, by definition, a square matrix, of a rank equal to the total number of vertices plus edges. Thus, for the above example, the incidence matrix is simply

{\displaystyle \left[{\begin{matrix}0&1\\1&0\end{matrix}}\right]}
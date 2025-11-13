from __future__ import annotations
from typing import List, Optional, Tuple, Set
from ...codes.elements import AncillaQubit, Edge, PseudoQubit
from .elements import Cluster
from .._template import Sim
from collections import defaultdict
from ..mwpm.sim import Planar as PlanarMWPM
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
import time

class Toric(Sim):
    """Union-Find decoder for the toric lattice.

    In this implementation, cluster properties are not stored at the root of the tree. Instead, ancillas are collected within `~.unionfind.elements.Cluster` objects, which contain the `~.unionfind.elements.Cluster.union` and `~.unionfind.elements.Cluster.find` methods.

    Default values for the following parameters can be supplied via a *decoders.ini* file under the section of ``[unionfind]``.

    The ``cluster`` and ``peeled`` attributes are monkey patched to the `~.codes.elements.AncillaQubit` object to assist the identification of its parent cluster and to assist peeling. The ``forest`` attribute is monkey-patched to `~codes.elements.AncillaQubit` and `~codes.elements.Edge` if a dynamic forest is not maintained to assist with the construction of the acyclic forest after cluster growth.

    Parameters
    ----------

    weighted_growth : bool, optional
        Enables weighted growth via bucket growth. Default is true. See `grow_clusters`.
    weighted_union : bool, optional
        Enables weighted union, Default is true. See `union_bucket`.
    dynamic_forest : bool, optional
        Enables dynamically mainted forests. Default is true.
    print_steps : bool, optional
        Prints additional decoding information. Default is false.
    kwargs
        Keyword arguments are forwarded to `~.decoders._template.Sim`.

    Attributes
    ----------
    support : dict

        Dictionary of growth states of all edges in the code.

        =====   ========================
        value   state
        =====   ========================
        2       fully grown
        1       half grown
        0       none
        -1      removed by cycle or peel
        -2      added to matching
        =====   ========================

    buckets : `~collections.defaultdict`
        Ordered dictionary (by index) for bucket growth (implementation of weighted growth). See `grow_clusters`.
    bucket_max_filled : int
        The hightest occupied bucket. Allows for break from bucket loop.
    clusters : list
        List of all clusters at initialization.
    cluster_index : int
        Index value for cluster differentiation.
    """

    name = "Union-Find-Breadth-First-Search"
    short = "ufbfs"
    _Cluster = Cluster

    compatibility_measurements = dict(
        PerfectMeasurements=True,
        FaultyMeasurements=False,
    )
    compatibility_errors = dict(
        pauli=True,
        erasure=False,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config["step_growth"] = not (self.config["step_bucket"] or self.config["step_cluster"])

        # Apply Monkey Patching
        self.code._AncillaQubit.cluster = None
        self.code._AncillaQubit.peeled = None
        if not self.config["dynamic_forest"]:
            self.code._AncillaQubit.forest = None
            self.code._Edge.forest = None

        # Initiated support table
        self.support = {}
        for layer in self.code.data_qubits.values():
            for data_qubit in layer.values():
                for edge in data_qubit.edges.values():
                    self.support[edge] = 0
        if self.code.layers > 1:
            for layer in self.code.ancilla_qubits.values():
                for ancilla_qubit in layer.values():
                    self.support[
                        ancilla_qubit.z_neighbors[
                            self.code.ancilla_qubits[(ancilla_qubit.z + 1) % self.code.layers][ancilla_qubit.loc]
                        ]
                    ] = 0
        if self.config["weighted_growth"]:
            self.buckets_num = self.code.size[0] * self.code.size[1] * self.code.layers * 2
        else:
            self.buckets_num = 2
        self.buckets = defaultdict(list)
        self.bucket_max_filled = 0
        self.clusters = []
        self.cluster_index = 0

    def decode(self, **kwargs):
        """Decodes the code using the Union-Find algorithm.

        Decoding process can be subdivided into 3 sections:

        1.  Finding the initial clusters.
        2.  Growing and merging these clusters.
        3.  Peeling the clusters using the Peeling algorithm.

        Parameters
        ----------
        kwargs
            Keyword arguments are passed on to `find_clusters`, `grow_clusters` and `peel_clusters`.
        """
        # t00 = time.time()
        self.buckets = defaultdict(list)
        self.bucket_max_filled = 0
        self.cluster_index = 0
        self.clusters = []
        self.support = {edge: 0 for edge in self.support}
        self.find_clusters(**kwargs)
        self.grow_clusters(**kwargs)
        # at this point, self.clusters is the updated clusters!!
        # t0 = time.time()
        phi = self.calc_phi()
        # t1 = time.time()
        # print('calc phi time: ', t1 - t0)

        self.peel_clusters(**kwargs)
        # timef = time.time()
        # print('whole process', timef - t00)

        return phi

    """
    -------------------------------------------------------------------------------------------
                                    General helper functions
    -------------------------------------------------------------------------------------------
    """

    # @njit
    def calc_phi(self):
        # Part 1: getting the edges
        # t01 = time.time()
        edges = [i for i in self.support.keys() if i.state_type == 'x']
        # t1 = time.time()
        # print('building list of edges', t1 - t01)
        p = self.code.error_rates['p_bitflip']
        # p = 0.05 # patch for error rate
        G = nx.Graph()

        # Helper to produce canonical node ids (Option B):
        # - boundary ancillas (qubit_type 'pA') map to their loc[0] integer (preserves old 0/d semantics)
        # - other ancilla-like nodes map to a tuple of their loc for stable, hashable ids
        def nid(n):
            try:
                # boundary pseudo-ancilla: keep integer column index so existing (0, d) queries continue to work
                if getattr(n, "qubit_type", None) == "pA":
                    return int(n.loc[0])
                # any object with a loc attribute -> tuple(loc)
                if hasattr(n, "loc"):
                    # coerce loc to tuple for hashability
                    loc = n.loc
                    if isinstance(loc, (list, tuple)):
                        return tuple(loc)
                    # single integer-like loc
                    return (int(loc),)
            except Exception:
                # fallback to object id string to avoid mixing object references
                return ("obj", id(n))
            # final fallback
            return ("obj", id(n))

        # edges_graph = np.empty(len(edges), dtype = object)
        for i, edge in enumerate(edges):
            # collect boundary nodes
            node0 = edge.nodes[0]
            node1 = edge.nodes[1]

            if self.support[edge] == 2:
                w = 0
            else:
                w = 1

            # use canonical ids for both endpoints so the graph node set is homogeneous
            G.add_edge(nid(node0), nid(node1), weight=w)

        # # list comprehension is like the same speed
        # edges_graph = [(edge.nodes[0].loc[0] if edge.nodes[0].qubit_type == 'pA' else edge.nodes[0], edge.nodes[1], {'weight': 0 if self.support[edge] == 2 else 1}) for edge in self.support.keys() if edge.state_type == 'x']
        # G.add_edges_from(edges_graph)
        # t2 = time.time()
        # print('generating graph: ', t2-t1)

        # get shortest path
        d = self.code.size[0]
        # shortest_path_length still queries integer column indices (0 and d) for boundaries — those
        # are produced by nid() for pA nodes, so this call will succeed.
        length = nx.shortest_path_length(G, 0, d, weight='weight')
        # t3 = time.time()
        # print('running dijkstras: ', t3-t2)

        # print('length', length)
        return length*np.log((1-p)/p)

    def calc_phi_slow(self):
        # Part 1: getting the edges
        edges = list(self.support.keys())
        p = self.code.error_rates['p_bitflip']
        # t1 = time.time()
        G = nx.Graph()
        boundary_nodes = set()
        for edge in edges:
            if edge.state_type == 'x':
                # collect boundary nodes
                if edge.nodes[0].qubit_type == 'pA':
                    boundary_nodes.add(edge.nodes[0])
                if edge.nodes[1].qubit_type == 'pA':
                    boundary_nodes.add(edge.nodes[1])

                # actually add the edge
                if self.support[edge] == 2:
                    G.add_edge(edge.nodes[0], edge.nodes[1], weight=0)
                else:

                    G.add_edge(edge.nodes[0], edge.nodes[1], weight=-np.log(p / (1 - p)))
                # some may have the same pA (boundary node) which is good
        t15 = time.time()
        # print('pre boundary nodes', t15 - t1)
        # connect boundary nodes on the same side into 1 node
        for elem1 in boundary_nodes:
            for elem2 in boundary_nodes:
                if elem1 != elem2:
                    # if vertical, change loc[0] to loc[1]
                    if (elem1.loc[0] == 0 and elem2.loc[0] == 0) or (elem1.loc[0] != 0 and elem2.loc[0] != 0):
                        G.add_edge(elem1, elem2, weight=0)
        t2 = time.time()
        # print('boundary nodes: ', t2-t15)
        # call dijkstras
        # get s & t nodes
        for elem in boundary_nodes:
            # arbitrarily pick one to be the
            if elem.loc[0] == 0 and elem.loc[1] == 0:
                s = elem
            elif elem.loc[1] == 0:
                t = elem
        length, path = nx.single_source_dijkstra(G, s, t)
        # t3 = time.time()
        # print('running dijkstras: ', t3-t2)


        return length

    def get_cluster(self, ancilla: AncillaQubit) -> Optional[Cluster]:
        """Returns the cluster to which ``ancilla`` belongs to.

        If ``ancilla`` has no cluster or the cluster is not from the current simulation, none is returned. Otherwise, the root element of the cluster-tree is found, updated to ``ancilla.cluster`` and returned.

        Parameters
        ----------
        ancilla
            The ancilla for which the cluster is to be found.
        """
        if ancilla.cluster is not None and ancilla.cluster.instance == self.code.instance:
            ancilla.cluster = ancilla.cluster.find()
            return ancilla.cluster

    def cluster_add_ancilla(
        self,
        cluster: Cluster,
        ancilla: AncillaQubit,
        parent: Optional[AncillaQubit] = None,
        **kwargs,
    ):
        """Recursively adds erased edges to ``cluster`` and finds the new boundary.

        For a given ``ancilla``, this function finds the neighboring edges and ancillas that are in the the currunt cluster. If the newly found edge is erased, the edge and the corresponding ancilla will be added to the cluster, and the function applied recursively on the new ancilla. Otherwise, the neighbor is added to the new boundary ``self.new_bound``.

        Parameters
        ----------
        cluster
            Current active cluster
        ancilla
            Ancilla from which the connected erased edges or boundary are searched.
        """
        cluster.add_ancilla(ancilla)

        for (new_ancilla, edge) in self.get_neighbors(ancilla).values():
            if (
                "erasure" in self.code.errors
                and edge.qubit.erasure == self.code.instance
                and new_ancilla is not parent
                and self.support[edge] == 0
            ):  # if edge not already traversed
                if new_ancilla.cluster == cluster:  # cycle detected, peel edge
                    self._edge_peel(edge, variant="cycle")
                else:  # if no cycle detected
                    self._edge_full(ancilla, edge, new_ancilla)
                    self.cluster_add_ancilla(cluster, new_ancilla, parent=ancilla)
            elif new_ancilla.cluster is not cluster:  # Make sure new bound does not lead to self
                cluster.new_bound.append((ancilla, edge, new_ancilla))

    def _edge_peel(self, edge: Edge, variant: str = ""):
        """Peels or removes an edge"""
        self.support[edge] = -1
        if self.config["print_steps"]:
            print(f"del {edge} ({variant})")

    def _edge_grow(self, ancilla, edge, new_ancilla, **kwargs):
        """Grows the edge in support."""
        if self.support[edge] == 1:
            self._edge_full(ancilla, edge, new_ancilla, **kwargs)
        else:
            self.support[edge] += 1

    def _edge_full(self, ancilla, edge, new_ancilla, **kwargs):
        """Fully grows an edge."""
        self.support[edge] = 2

    """
    -------------------------------------------------------------------------------------------
                                    1. Find clusters
    -------------------------------------------------------------------------------------------
    """

    def find_clusters(self, **kwargs):
        """Initializes the clusters on the lattice.

        For every non-trivial ancilla on the lattice, a `~.unionfind.elements.Cluster` is initiated. If any set of ancillas are connected by some set of erased qubits, all connected ancillas are found by `cluster_add_ancilla` and a single cluster is initiated for the set.

        The cluster is then placed into a bucket based on its size and parity by `place_bucket`. See `grow_clusters` for more information on buckets.
        """
        plaqs, stars = self.get_syndrome()
        for ancilla in plaqs + stars:
            if ancilla.cluster is None or ancilla.cluster.instance != self.code.instance:
                cluster = self._Cluster(self.cluster_index, self.code.instance)
                self.cluster_add_ancilla(cluster, ancilla)
                self.cluster_index += 1
                self.clusters.append(cluster)

        self.place_bucket(self.clusters, -1)

        if self.config["print_steps"]:
            print(f"Found clusters:\n" + ", ".join(map(str, self.clusters)) + "\n")

    """
    -------------------------------------------------------------------------------------------
                                    2(a). Grow clusters expansion
    -------------------------------------------------------------------------------------------
    """

    def grow_clusters(self, **kwargs):
        """
            Grows all active (odd-parity) clusters in parallel using BFS,
            one layer at a time, until all clusters are even (frozen).
            (Implements Algorithm 1, lines 7-30)
            """

        # 1. INITIALIZE ACTIVE LIST (Pseudocode lines 1-6, 8)
        # Get all root clusters that are initially active (odd)
        active_roots: List[Cluster] = [cluster.find() for cluster in self.clusters if cluster.is_active]
        # active_roots = {anc.cluster.find() for anc in self.code.ancillas if anc.syndrome}

        while active_roots:

            union_list = []
            newly_occupied_nodes = []

            # --- 2. GROWTH PHASE (Pseudocode lines 9-10, 14-16) ---
            # Grow ALL active clusters by one layer simultaneously
            for cluster_root in active_roots:

                # Use cluster.new_bound, which holds the current boundary edges
                current_boundary_edges = cluster_root.new_bound
                cluster_root.new_bound = []  # Clear for next layer

                for ancilla, edge, new_ancilla in current_boundary_edges:

                    new_cluster: Optional[Cluster] = self.get_cluster(new_ancilla)  # This can be None

                    # Check if new_ancilla is part of our own cluster
                    if new_cluster is not None and new_cluster.find() == cluster_root:
                        continue

                    # Check if the node is a boundary
                    is_boundary_node = isinstance(new_ancilla, PseudoQubit)

                    # Check if the node belongs to another cluster *from this instance*
                    is_other_cluster = (new_cluster is not None and
                                        new_cluster.instance == self.code.instance)

                    # A node is "occupied" if it's a boundary OR another cluster
                    if is_boundary_node or is_other_cluster:
                        # Occupied: Add to list for merge phase (Pseudocode line 11)
                        union_list.append((ancilla, edge, new_ancilla))
                    else:
                        # Unoccupied: claim it (Pseudocode line 15)
                        self.cluster_add_ancilla(cluster_root, new_ancilla, parent=ancilla)
                        newly_occupied_nodes.append(new_ancilla)

            # --- 3. FIND NEXT BOUNDARY (INLINED) ---
            # For all nodes we *just* claimed, find their neighbors
            # and add those edges to the cluster's *new* boundary list
            # This replaces the old 'grow_boundary' logic.
            for node in newly_occupied_nodes:
                cluster_root = node.cluster.find()

                for cur_direction, (cur_ancilla, edge) in self.get_neighbors(node).items():
                    neighbor_tuple = self.get_neighbor(node, cur_direction)
                    neighbor_ancilla = neighbor_tuple[0]

                    # --- REPLACEMENT ---
                    new_cluster_neighbor = self.get_cluster(neighbor_ancilla)

                    # Check if the neighbor is a boundary (None) or part of another cluster
                    if new_cluster_neighbor is None:
                        neighbor_root = None  # It's a boundary
                    else:
                        neighbor_root = new_cluster_neighbor.find()
                    # --- END REPLACEMENT ---

                    if neighbor_root != cluster_root:
                        # This edge leads to another cluster or a boundary
                        # Add it to the *root's* boundary list for the *next* iteration
                        cluster_root.new_bound.append((node, edge, neighbor_ancilla))

            # --- 4. MERGE PHASE (Pseudocode lines 11-13) ---
            # Call your *unmodified* union_bucket function.
            # It already handles parity updates correctly.
            self.union_bucket(union_list)

            # --- 5. UPDATE ACTIVE LIST (Pseudocode line 18-20) ---
            # Re-calculate the set of active *roots* from *all* clusters
            # to prepare for the next 'while' loop iteration.
            active_roots = [c.find() for c in self.clusters if c.find().is_active]

    def grow_bucket(self, bucket: List[Cluster], bucket_i: int, **kwargs) -> Tuple[List, List]:
        """Grows the clusters which are contained in the current bucket.

        See `grow_clusters` for more information.

        Parameters
        ----------
        bucket
            List of clusters to be grown.
        bucket_i
            Current bucket number.

        Returns
        -------
        list
            List of potential mergers between two cluster-distinct ancillas.
        list
            List of odd-parity clusters to be placed in new buckets.
        """
        if self.config["print_steps"]:
            string = f"Growing bucket {bucket_i} of clusters:"
            print("=" * len(string) + "\n" + string)

        union_list, place_list = [], []
        while bucket:  # Loop over all clusters in the current bucket\
            cluster = bucket.pop().find()
            if cluster.bucket == bucket_i and cluster.support == bucket_i % 2:
                place_list.append(cluster)
                self.grow_boundary(cluster, union_list)

        if self.config["print_steps"]:
            print("\n")

        return union_list, place_list

    def grow_boundary(self, cluster: Cluster, union_list: List[Tuple[AncillaQubit, Edge, AncillaQubit]], **kwargs):
        """Grows the boundary of the ``cluster``.

        See `grow_clusters` for more information.

        Parameters
        ----------
        cluster
            The cluster to be grown.
        union_list
            List of potential mergers between two cluster-distinct ancillas.
        """
        cluster.support = 1 - cluster.support
        cluster.bound, cluster.new_bound = cluster.new_bound, []

        while cluster.bound:  # grow boundary
            boundary = cluster.bound.pop()
            new_edge = boundary[1]

            if self.support[new_edge] != 2:  # if not already fully grown
                self._edge_grow(*boundary)  # Grow boundaries by half-edge
                if self.support[new_edge] == 2:  # if edge is fully grown
                    union_list.append(boundary)  # Append to union_list list of edges
                else:
                    cluster.new_bound.append(boundary)

        if self.config["print_steps"]:
            print(f"{cluster}, ", end="")

    """
    -------------------------------------------------------------------------------------------
                                    2(b). Grow clusters union
    -------------------------------------------------------------------------------------------
    """

    def union_bucket(self, union_list: List[Tuple[AncillaQubit, Edge, AncillaQubit]], **kwargs):
        """Merges clusters, handling boundaries (Pseudocode lines 11-13)."""
        if union_list and self.config["print_steps"]:
            print("Cluster unions.")

        for ancilla, edge, new_ancilla in union_list:
            cluster_root = self.get_cluster(ancilla).find()
            new_cluster = self.get_cluster(new_ancilla)  # This can be None

            if new_cluster is None:
                # --- THIS IS A BOUNDARY MERGE ---
                # This ancilla is a PseudoQubit (boundary)
                if not cluster_root.on_bound:
                    cluster_root.on_bound = True
                    if self.config["print_steps"]:
                        print(f"{cluster_root} hit boundary, now frozen.")

            else:
                # --- THIS IS A CLUSTER-CLUSTER MERGE ---
                new_cluster_root = new_cluster.find()
                if cluster_root != new_cluster_root:

                    string = "{}∪{}=".format(cluster_root, new_cluster_root) if self.config["print_steps"] else ""

                    # Your .union() method already handles parity, size, etc.
                    # Just make sure to merge the roots correctly.
                    if self.config["weighted_union"] and cluster_root.size < new_cluster_root.size:
                        new_cluster_root.union(cluster_root)  # Merges cluster into new_cluster
                        if string: print(string, new_cluster_root)
                    else:
                        cluster_root.union(new_cluster_root)  # Merges new_cluster into cluster
                        if string: print(string, cluster_root)

                elif self.config["dynamic_forest"]:
                    self._edge_peel(edge, variant="cycle")

        if union_list and self.config["print_steps"]:
            print("")

    def union_check(
        self,
        edge: Edge,
        ancilla: AncillaQubit,
        new_ancilla: AncillaQubit,
        cluster: Cluster,
        new_cluster: Cluster,
    ) -> bool:
        """Checks whether ``cluster`` and ``new_cluster`` can be joined on ``edge``.

        See `union_bucket` for more information.
        """
        if new_cluster is None or new_cluster.instance != self.code.instance:
            self.cluster_add_ancilla(cluster, new_ancilla, parent=ancilla)
        elif new_cluster is cluster:
            if self.config["dynamic_forest"]:
                self._edge_peel(edge, variant="cycle")
        else:
            return True
        return False

    """
    -------------------------------------------------------------------------------------------
                                    2(c). Place clusters in buckets
    -------------------------------------------------------------------------------------------
    """

    def place_bucket(self, clusters: List[Cluster], bucket_i: int):
        """Places all clusters in ``clusters`` in a bucket if parity is odd.

        If ``weighted_growth`` is enabled. the cluster is placed in a new bucket based on its size, otherwise it is placed in ``self.buckets[0]``

        Parameters
        ----------
        clusters
            Clusters to place in buckets.
        bucket_i
            Current bucket number.
        """
        for cluster in clusters:

            cluster = cluster.find()

            if cluster.parity % 2 == 1:
                if self.config["weighted_growth"]:
                    cluster.bucket = 2 * (cluster.size - 1) + cluster.support
                    self.buckets[cluster.bucket].append(cluster)
                    if cluster.bucket > self.bucket_max_filled:
                        self.bucket_max_filled = cluster.bucket
                else:
                    self.buckets[0].append(cluster)
                    cluster.bucket = bucket_i + 1
            else:
                cluster.bucket = None

    """
    -------------------------------------------------------------------------------------------
                                    3. Peel clusters
    -------------------------------------------------------------------------------------------
    """

    def peel_clusters(self, **kwargs):
        """Loops over all clusters to find pendant ancillas to peel.

        To make sure that all cluster-trees are fully peeled, all ancillas are considered in the loop. If the ancilla has not been peeled before and belongs to a cluster of the current simulation, the ancilla is considered for peeling by `peel_leaf`.
        """
        if self.config["print_steps"]:
            print("================\nPeeling clusters")
        for layer in self.code.ancilla_qubits.values():
            for ancilla in layer.values():
                if ancilla.peeled != self.code.instance and ancilla.cluster and ancilla.cluster.instance == self.code.instance:
                    if not self.config["dynamic_forest"]:
                        self.static_forest(ancilla)
                    cluster = self.get_cluster(ancilla)
                    self.peel_leaf(cluster, ancilla)

    def peel_leaf(self, cluster, ancilla):
        """Recursive function which peels a branch of the tree if the input ancilla is a pendant ancilla

        If there is only one neighbor of the input ancilla that is in the same cluster, this ancilla is a pendant ancilla and can be peeled. The function calls itself on the other ancilla of the edge leaf.

        If ["dynamic_forest"] is disabled, once a pendant leaf is found, the acyclic forest is constructed by `static_forest`.

        Parameters
        ----------
        cluster
            Current cluster being peeled.
        ancilla
            Pendant ancilla of the edge to be peeled.
        """
        leaf = self.find_leaf(cluster, ancilla)
        if leaf:
            key, (new_ancilla, edge) = leaf
            if ancilla.syndrome:
                self.flip_edge(ancilla, edge, new_ancilla)
                if type(key) is not int:
                    self.correct_edge(self.code.ancilla_qubits[self.code.decode_layer][ancilla.loc], key)
            else:
                self._edge_peel(edge, variant="peel")
            ancilla.peeled = self.code.instance
            self.peel_leaf(cluster, new_ancilla)

    def find_leaf(self, cluster: Cluster, ancilla: AncillaQubit, **kwargs):
        num_connect, leaf = 0, ()
        neighbors = self.get_neighbors(ancilla)
        for key, neighbor in neighbors.items():
            (new_ancilla, edge) = neighbor
            if self.support[edge] == 2:
                new_cluster = self.get_cluster(new_ancilla)
                if new_cluster is cluster:
                    num_connect += 1
                    leaf = (key, neighbor)
            if num_connect > 1:
                return
        else:
            if num_connect == 1:
                return leaf

    def flip_edge(self, ancilla: AncillaQubit, edge: Edge, new_ancilla: AncillaQubit, **kwargs):
        """Flips the values of the ancillas connected to ``edge``."""
        ancilla.syndrome = not ancilla.syndrome
        new_ancilla.syndrome = not new_ancilla.syndrome
        self.support[edge] = -2
        if self.config["print_steps"]:
            print(f"{edge} to matching")

    def static_forest(self, ancilla: AncillaQubit):
        """Constructs an acyclic forest in the cluster of ``ancilla``.

        Applies recursively to all neighbors of ``ancilla``. If a cycle is detected, edges are removed from the cluster.

        Parameters
        ----------
        ancilla
        """
        ancilla.forest = self.code.instance
        neighbors = self.get_neighbors(ancilla)
        for neighbor in neighbors.values():
            (new_ancilla, edge) = neighbor
            if self.support[edge] == 2:
                if new_ancilla.forest != self.code.instance:
                    edge.forest = self.code.instance
                    self.static_forest(new_ancilla)
                elif new_ancilla.forest == self.code.instance and edge.forest != self.code.instance:
                    self._edge_peel(edge, variant="cycle")


class Planar(Toric):
    """Union-Find decoder for the planar lattice.

    See the description of `.unionfind.sim.Toric`.
    """

    def cluster_add_ancilla(
        self,
        cluster: Cluster,
        ancilla: AncillaQubit,
        parent: Optional[AncillaQubit] = None,
        **kwargs,
    ):
        """Recursively adds erased edges to ``cluster`` and finds the new boundary.

        For a given ``ancilla``, this function finds the neighboring edges and ancillas that are in the the currunt cluster. If the newly found edge is erased, the edge and the corresponding ancilla will be added to the cluster, and the function applied recursively on the new ancilla. Otherwise, the neighbor is added to the new boundary ``self.new_bound``.

        Parameters
        ----------
        cluster
            Current active cluster
        ancilla
            Ancilla from which the connected erased edges or boundary are searched.
        """
        cluster.add_ancilla(ancilla)

        for (new_ancilla, edge) in self.get_neighbors(ancilla).values():
            if (
                "erasure" in self.code.errors
                and edge.qubit.erasure == self.code.instance
                and new_ancilla is not parent
                and self.support[edge] == 0
            ):
                if isinstance(new_ancilla, PseudoQubit):
                    if cluster.on_bound:
                        self._edge_peel(edge, variant="cycle")
                    else:
                        self._edge_full(ancilla, edge, new_ancilla)
                        cluster.add_ancilla(new_ancilla)
                else:
                    if new_ancilla.cluster == cluster:
                        self._edge_peel(edge, variant="cycle")
                    else:
                        self._edge_full(ancilla, edge, new_ancilla)
                        self.cluster_add_ancilla(cluster, new_ancilla, parent=ancilla)
            elif new_ancilla.cluster is not cluster and not (
                isinstance(new_ancilla, PseudoQubit) and cluster.on_bound
            ):  # Make sure new bound does not lead to self
                cluster.new_bound.append((ancilla, edge, new_ancilla))

    def union_check(
        self,
        edge: Edge,
        ancilla: AncillaQubit,
        new_ancilla: AncillaQubit,
        cluster: Cluster,
        new_cluster: Cluster,
    ) -> bool:
        """Checks whether ``cluster`` and ``new_cluster`` can be joined on ``edge``.

        See `union_bucket` for more information.
        """
        if new_cluster is cluster or (isinstance(new_ancilla, PseudoQubit) and cluster.on_bound):
            if self.config["dynamic_forest"]:
                self._edge_peel(edge, variant="cycle")
        elif new_cluster is None or new_cluster.instance != self.code.instance:
            self.cluster_add_ancilla(cluster, new_ancilla, parent=ancilla)
        else:
            return True
        return False

    def place_bucket(self, clusters: List[Cluster], bucket_i: int):
        """Places all clusters in ``clusters`` in a bucket if parity is odd.

        If ``weighted_growth`` is enabled. the cluster is placed in a new bucket based on its size, otherwise it is placed in ``self.buckets[0]``

        Parameters
        ----------
        clusters
            Clusters to place in buckets.
        bucket_i
            Current bucket number.
        """
        for cluster in clusters:

            cluster = cluster.find()

            if cluster.parity % 2 == 1 and not cluster.on_bound:
                if self.config["weighted_growth"]:
                    cluster.bucket = 2 * (cluster.size - 1) + cluster.support
                    self.buckets[cluster.bucket].append(cluster)
                    if cluster.bucket > self.bucket_max_filled:
                        self.bucket_max_filled = cluster.bucket
                else:
                    self.buckets[0].append(cluster)
                    cluster.bucket = bucket_i + 1
            else:
                cluster.bucket = None

    def static_forest(self, ancilla: AncillaQubit, found_bound: str = False, **kwargs) -> bool:
        # Inherited docsting
        if not found_bound and ancilla.cluster.find().parity % 2 == 0:
            found_bound = True

        ancilla.forest = self.code.instance
        for key in ancilla.parity_qubits:
            (new_ancilla, edge) = self.get_neighbor(ancilla, key)

            if self.support[edge] == 2:

                if type(new_ancilla) is PseudoQubit:
                    if found_bound:
                        self._edge_peel(edge, variant="cycle")
                    else:
                        edge.forest = self.code.instance
                        found_bound = True
                    continue

                if new_ancilla.forest == self.code.instance:
                    if edge.forest != self.code.instance:
                        self._edge_peel(edge, variant="cycle")
                else:
                    edge.forest = self.code.instance
                    found_bound = self.static_forest(new_ancilla, found_bound=found_bound)
        return found_bound

    def peel_clusters(self, **kwargs):
        # Inherited docstring
        super().peel_clusters(**kwargs)
        for layer in self.code.pseudo_qubits.values():
            for ancilla in layer.values():
                if ancilla.peeled != self.code.instance and ancilla.cluster and ancilla.cluster.instance == self.code.instance:
                    if not self.config["dynamic_forest"]:
                        self.static_forest(ancilla)
                    cluster = self.get_cluster(ancilla)
                    leaf = self.find_leaf(cluster, ancilla)
                    if leaf:
                        key, (new_ancilla, edge) = leaf
                        self._edge_peel(edge, variant="peel")
                        self.peel_leaf(cluster, new_ancilla)

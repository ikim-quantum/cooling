#######################################
# MPS library for algorithmic cooling #
# Galit Anikeeva, Isaac Kim           #
# 2/13/2020                           #
# Use it as you wish. Just don't sue  #
# us.                                 #
#######################################
import numpy as np
import networkx as nx


class TensorNetwork:
    """
    Tensor Network class. 

    Some invariants:
      * self.unused >= 0
      * self.unused > key for all keys in edges 
      * keys >= 0
    """

    graph: nx.Graph
    unused: int

    def __init__(self):
        # Using MultiDiGraph because we need a concept of left and right
        # And we need a way for multiple edges to exist.
        self.graph = nx.MultiGraph()

        # Also number of edges
        # We should guarrantee that this greater or equal
        # to the keys in the edges at any point
        self.unused = 0

    @staticmethod
    def _relabel_data(relabel, data):
        axes = data['axes']
        if isinstance(axes, dict):
            axes = {relabel(k): v for k, v in axes.items()}
        return {'axes': axes}

    def copy(self, shift=0, relabel=None):
        """
        Copies the tensor network. 
        It relabels the nodes and shifts the keys of the edges.
        """
        if relabel is None:
            relabel = lambda x: x

        copy = TensorNetwork()

        relabeled_edges = (
            (relabel(a), relabel(b), key + shift, TensorNetwork._relabel_data(relabel, data))
            for a, b, key, data in self.graph.edges(keys=True, data=True)
        )

        copy.graph.add_nodes_from(
            (relabel(a), data) for a, data in self.graph.nodes(data=True)
        )
        copy.graph.add_edges_from(relabeled_edges)

        copy.unused = self.unused + shift
        return copy

    @staticmethod
    def merge(left, right, left_relabel=None, right_relabel=None):
        """
        This is an important method. It merges two tensor networks, applying the relabels 
        so that we can (if desired) guarantee that the union is disjoint.

        In case of clashes, left takes priority.
        """
        if left_relabel is None:
            left_relabel = lambda x: x

        if right_relabel is None:
            right_relabel = lambda x: x

        merged = TensorNetwork()
        merged.graph.add_nodes_from(
            (right_relabel(a), data) for a, data in right.graph.nodes(data=True)
        )
        merged.graph.add_nodes_from(
            (left_relabel(a), data) for a, data in left.graph.nodes(data=True)
        )

        relabeled_edges_right = (
            (right_relabel(a), right_relabel(b), key + left.unused, TensorNetwork._relabel_data(right_relabel, data))
            for a, b, key, data in left.graph.edges(keys=True, data=True)
        )
        relabeled_edges_left = (
            (left_relabel(a), left_relabel(b), key, TensorNetwork._relabel_data(left_relabel, data))
            for a, b, key, data in left.graph.edges(keys=True, data=True)
        )

        merged.graph.add_edges_from(relabeled_edges_left)
        merged.graph.add_edges_from(relabeled_edges_right)

        merged.unused = left.unused + right.unused
        return merged

    def add_node(self, name, tensor: np.ndarray):
        """
        Adds a single node to the tensor network.
        
        """
        self.graph.add_node(name, tensor=tensor)

    def new_key(self):
        """
        Outputs an unused key and inceases the unsued key counter.
        """
        self.unused += 1
        return self.unused - 1

    def connect_nodes(self, left, left_axis, right, right_axis):
        """
        Connects the left and right node, such that the left is connected 
        at the axis `left_axis` and the right at the position `right_axis`.
        These axis must be of the same size.

        Returns the key of the edge, useful for idenfitying contractions.
        
        Example:
        If we have the left node with shape (2,3,4)
        and a right node with shape (3,5,7), 
        the only posibility is left_axis = 1 and right_axis = 0.

        Throws an exception if the axis have different dimension.
        """
        left_dim = self.graph.nodes[left]['tensor'].shape[left_axis]
        right_dim = self.graph.nodes[right]['tensor'].shape[right_axis]

        if left_dim != right_dim:
            raise ValueError("The dimensions must match up.")

        if left == right:
            axes = [left_axis, right_axis]
        else:
            axes = {}
            axes[left] = left_axis
            axes[right] = right_axis

        label = self.graph.add_edge(left, right, axes=axes, key=self.new_key())
        return label

    def contract(self, left, right, key=None, all_edges=False, no_merge_ok=False):
        """
        Contracts left and right node, over the edge with that key.
        If no key is provided, and there is a single edge, we use that edge.

        If all_edges is true, we contract over all the edges shared, so that no 
        self-loops are created.

        If no_merge_ok, this returns without changes or exceptions when 
        there are no edges.
        """
        n_edges = self.graph.number_of_edges(left, right)
        if n_edges == 0:
            if no_merge_ok:
                return
            raise ValueError("There is no edge between these nodes to contract.")

        if key is None:
            if all_edges:
                keys = list(self.graph.get_edge_data(left, right).keys())
                self.contract(left, right, key=keys[0])

                for key in keys[1:]:
                    self.contract(left, left, key=key)
                return

            if n_edges != 1:
                raise ValueError(
                    "Must provide a key if there is no or more than one edge."
                )
            e = self.graph.get_edge_data(left, right).items()
            key, attrs = next(iter(e))
        else:
            attrs = self.graph.get_edge_data(left, right, key=key)

        axes = attrs["axes"]

        if left != right:
            # We perform a product
            left_tensor = self.graph.nodes[left]["tensor"]
            right_tensor = self.graph.nodes[right]["tensor"]
            new_tensor = np.tensordot(
                left_tensor, right_tensor, axes=[axes[left], axes[right]]
            )
            self.graph.nodes[left]["tensor"] = new_tensor

            left_rank = len(left_tensor.shape) - 1

            self.graph.remove_edge(left, right, key)

            # Get new edges
            new_edges = []

            for _, neighbor, key, attrs in self.graph.edges(right, keys=True, data=True):
                if neighbor == left:
                    if attrs["axes"][left] >= axes[left]:
                        attrs["axes"][left] -= 1

                    if attrs["axes"][right] >= axes[right]:
                        attrs["axes"][right] -= 1

                    new_axes = [attrs["axes"][left], attrs["axes"][right] + left_rank]
                    new_edges.append((left, left, key, {"axes": new_axes}))

                elif neighbor == right:
                    for i in range(2):
                        if attrs["axes"][i] >= axes[right]:
                            attrs["axes"][i] -= 1

                    new_axes = [
                        attrs["axes"][0] + left_rank,
                        attrs["axes"][1] + left_rank,
                    ]
                    new_edges.append((left, left, key, {"axes": new_axes}))

                else:
                    if attrs["axes"][right] >= axes[right]:
                        attrs["axes"][right] -= 1

                    new_axes = {
                        left: attrs["axes"][right] + left_rank,
                        neighbor: attrs["axes"][neighbor],
                    }
                    new_edges.append((left, neighbor, key, {"axes": new_axes}))

            self.graph.remove_node(right)

            for _, neighbor, attrs in self.graph.edges(left, data=True):
                if left == neighbor:
                    for i in range(2):
                        if attrs["axes"][i] >= axes[left]:
                            attrs["axes"][i] -= 1
                else:
                    if attrs["axes"][left] >= axes[left]:
                        attrs["axes"][left] -= 1

            self.graph.add_edges_from(new_edges)

        else:
            # We perform a trace
            old_tensor = self.graph.nodes[left]["tensor"]
            new_tensor = np.trace(old_tensor, axis1=axes[0], axis2=axes[1])
            self.graph.nodes[left]["tensor"] = new_tensor

            self.graph.remove_edge(left, right, key)

            # Fix old indices
            for _, neighbor, attrs in self.graph.edges(left, data=True):
                if neighbor == left:
                    for i in range(2):
                        prev = attrs["axes"][i]
                        if prev >= axes[0]:
                            attrs["axes"][i] -= 1
                        if prev >= axes[1]:
                            attrs["axes"][i] -= 1
                else:
                    prev = attrs["axes"][left]
                    if prev >= axes[0]:
                        attrs["axes"][left] -= 1
                    if prev >= axes[1]:
                        attrs["axes"][left] -= 1


class MPS:
    """
    Matrix Product State class

    Attr:
        l(int): Length of the chain
        ds(list of ints): A list of local dimensions.
                          The length must be l.
        Ds_left(list of ints): A list of "left" bond
                               dimensions.
        Ds_right(list of ints): A list of "right" bond
                                dimensions.
        matrices(list of ndarrays): A list of matrices

    Data format for the matrix: Recall that a matrix
    product state is defined by a set of matrices. If
    we fix the local (physical) dimension to be d, for
    each site we can associate d different matrices. Each
    of the matrices are D x D matrices, where D is the 
    bond dimension of the underlying MPS. We can access
    each of these matrices as follows. Suppose we want
    to pick the k'th matrix of the ith site. Then one 
    can just use

    >>> a.matrices[i][k]

    where a is an instance of the MPS class.
    Instead, if one wants to see all the matrices, one 
    can use
    
    >>> a.matrices[i]

    which will return a (d x D x D) numpy ndarray.
    """

    def __init__(self, l, d, D):
        """
        Initialize a matrix product state with length
        l, ds=[d,...,d], Ds_left=Ds_right=[D,...,D].
        """
        self.l = l
        self.ds = [d] * l
        self.Ds_left = [D] * l
        self.Ds_right = [D] * l

        self.tn = TensorNetwork()

        # Initialize the matrices to zero matrices
        for i in range(l):
            self.tn.add_node(
                i, np.zeros([self.ds[i], self.Ds_left[i], self.Ds_right[i]])
            )

        for i in range(l):
            j = (i + 1) % l
            self.tn.connect_nodes(i, 1, j, 2)

    @staticmethod
    def inner_product(left, right):
        """
        Returns the inner product of two MPS.

        Args:
            left: An MPS.
            right: Another MPS, of same outer bond dimension and length.


        Returns:
            double: Norm of the MPS.
        """
        if left.l != right.l:
            raise ValueError("Must have same number of legs")

        merged_tn = TensorNetwork.merge(
            left.tn,
            right.tn,
            left_relabel=lambda x: (0, x),
            right_relabel=lambda x: (1, x),
        )
        for i in range(left.l):
            merged_tn.connect_nodes((0, i), 0, (1, i), 0)
            merged_tn.graph.nodes[(1, i)]["tensor"] = np.conjugate(merged_tn.graph.nodes[(1, i)]["tensor"])

        # breakpoint()
        merged_tn.contract((0, 0), (1, 0))
        for i in range(1, left.l):
            # breakpoint()
            merged_tn.contract(
                (0, 0), (0, i), all_edges=True
            )  # Should not need the all edges
            merged_tn.contract((0, 0), (1, i), all_edges=True)
        merged_tn.contract((0, 0), (0, 0), all_edges=True, no_merge_ok=True)
        return merged_tn.graph.nodes[(0, 0)]["tensor"]

    def norm(self):
        """
        Returns the norm of the MPS.

        Returns:
            double: Norm of the MPS.
        """
        return np.sqrt(np.real(MPS.inner_product(self, self)))


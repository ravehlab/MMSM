"""HierarchicalMSMVertex"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from collections import defaultdict
import warnings
import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from HMSM.util import util, linalg
from HMSM.models.serial.util import get_parent_update_condition


class HierarchicalMSMVertex:
    """HierarchicalMSMVertex.
    This class represents a single vertex in an HMSM tree.
    A vertex has the ids of its children and its parent, as well as having a transition matrix
    describing transition probabilities between its children. This transition matrix is updated
    when changes to the tree structure affecting the vertex are made, or when the transition
    probabilities change due to new data.

    Parameters
    ----------
    tree : HierarchicalMSMTree
        The tree this vertex is part of
    children : set
        A set of indices of this vertices children
    parent : int
        the id of the parent vertex to this vertex.
        Note that the root of an HierarchicalMSMTree, is its own parent
    tau : int
        the timestep resolution of this MSM, in multiples of basic timesteps - the timesteps of
        discrete trajectories given as input to the HierarchicalMSMTree in
        update_transition_counts.

        For example, if a Molecular dynamics simulation is sampled in timesteps of 2e-15 seconds,
        resulting in (x_0, x_1, ..., x_n), and then one every tenth sample is taken (x_0, x_10,...)
        and discretized into microstates, resulting in (microstate_0, microstate_10, ...) and
        this trajectory is used as input to HierarchicalMSMTree, the timestep resolution of this
        discrete trajectory is 10*2e-15=2e-14 seconds. Now if this MSM has tau=5, then the
        timestep resolution of this MSM is 5*2e-14=1e-13 seconds.
    height : int
        the height of this vertex in the tree, where microstates have height 0.
    config : dict
        a dictionary of model parameters.

    Attributes
    ----------
    id : int
        The unique identifier of this vertex.
    tree : HierarchicalMSMTree
        The HierarchicalMSMTree this vertex is part of.
    height : int
        The height of this vertex in the tree, where a height of 1 means this vertex is on the
        lowest level of the tree, and its MSM represents transitions between microstates.
    tau : int
        The number of (TODO rephrase this) steps on the base graph a single step in this MSM
        represents.
    children : list[int]
        A list of the ids of all the children of this vertex in the tree. This vertex represents
        an MSM describing transitions between these vertices.
    n : int
        The number of children of this vertex; the number of vertices in the MSM this vertex
        represents.
    neighbors : list[int]
        A list of the ids of other vertices on the same level in the tree as this vertex, such that
        there is a nontrivial probability of a transition occuring from one of the children of this
        vertex to one of the children of the other vertices, in tau steps.
    parent : int
        The id of the parent of this vertex in the tree.
    T : np.ndarray of shape (n+len(neighbors),n+len(neighbors))
        The transition probability matrix of this MSM over tau basic timesteps (see tau), from row to column.
        The upper-left n x n block represents transition probabilities among self.children().
        The upper-right n x len(neighbors) block stores transition probabilities from self.children() to neighbor vertices.
        The lower-left len(neighbors) x n block is 0
        The lower-right len(neighbors) x len(neighbors) block is an identity block matrix
    timescale : float # TODO: documnet units of timescale
        The timescale associated with the slowest process described by this MSM.
    is_root : bool
        True iff this vertex is the root of the tree.
    """

    SPLIT = "SPLIT"
    DISOWN_CHILDREN = "DISOWN_CHILDREN"
    UPDATE_PARENT = "UPDATE_PARENT"
    SUCCESS = "SUCCESS"

    def __init__(self, tree, children, parent, tau, height, partition_estimator, vertex_sampler, config):
        self.__id = util.get_unique_id()
        self.tree = tree
        self._children = set(children)
        if parent is None:
            self.parent = self.__id
        else:
            self.parent = parent
        self.tau = tau #TODO: maybe this should be calculated locally, not given as a parameter?
        self.height = height
        self._partition_estimator = partition_estimator
        self._vertex_sampler = vertex_sampler
        self._neighbors = []
        self._parent_update_condition = get_parent_update_condition(config.parent_update_condition,\
                                                                    config.parent_update_threshold)
        self.config = config

        self._T_is_updated = False
        self._last_update_sent = None # last external T sent to parent - output of last call to self.get_external_T() # TODO: refine this description


    @property
    def children(self):
        return list(self._children)

    @property
    def n(self):
        return len(self._children)
    @property
    def id(self):
        return self.__id

    @property
    def T(self):
        assert self._T_is_updated, "T should be updated before accessing."
        return self._T_tau

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_id):
        self._parent = parent_id

    @property
    def timescale(self):
        assert self._T_is_updated
        assert hasattr(self, "_timescale"), "timescale called before T was calculated"
        return self._timescale

    def _update_timescale(self):
        n = self.n
        if self.n == 1:
            self._timescale = 0
        else:
            T_inner = linalg.normalize_rows(self._T[:n, :n], norm=1)
            self._timescale = linalg.get_longest_timescale(T_inner, self.tau)

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def is_root(self):
        return self.parent == self.id

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, children_ids):
        """
        Add children to this tree.
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children.update(children_ids)
        self._T_is_updated = False

    def remove_children(self, children_ids):
        """
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children -= set(children_ids)
        self._T_is_updated = False

    def update(self):
        self.n_samples = np.sum([self.tree.get_n_samples(child) for child in self.children])
        return self._update_T()

    def _check_parent_update_condition(self):
        if self.is_root:
            return False
        if self._last_update_sent is None:
            return True
        if set(self._last_update_sent[0]) != set(self.get_external_T()[0]):
            return True

        return self._parent_update_condition(self)


    def _update_T(self):
        """
        Update the transition matrix of this vertex, and all the quantities associated with it;

        Returns:
        --------
        result : {"DISOWN_CHILDREN", "SPLIT", "UPDATE_PARENT", "SUCCESS"}
            If not "SUCCESS", the value of this indicates that some action needs to be taken by the
            tree.
        update : depends on the value of result
            If result is "DISOWN_CHILDREN", this will be a dictionary mapping parents to sets of
            children, such that each set of children should be moved to its parent.
            If result is "SPLIT", this is will be (partition, taus, self, self.parent), where
            partition is a partition of this vertices children, and taus is the tau of the new
            vertices for each subset in the partition.
            If result is "UPDATE_PARENT" or "SUCCESS" this will be None.
        """
        assert self.n > 0
        T_rows = dict()
        column_ids = set(self.children)

        # Get the rows of T corresponding to each child
        for child in self._children:
            ids, row = self.tree.get_external_T(child)
            T_rows[child] = (ids, row)
            linalg._assert_stochastic(row, axis=0)
            column_ids = column_ids.union(ids)

        # Get a mapping between indices of the matrix T and the vertex_ids they represent
        id_2_index, index_2_id, full_n = self._get_id_2_index_map(column_ids)
        self.index_2_id = index_2_id

        # Initialize the transition matrix
        self._T = np.zeros((full_n, full_n))
        n_external = full_n - self.n # number of external vertices

        # Now fill the matrix self._T
        for child, (ids, T_row) in T_rows.items():
            row = id_2_index[child]
            for id, transition_probability in zip(ids, T_row):
                column = id_2_index[id]
                self._T[row, column] += transition_probability

        if self.config.transition_estimator != 'Dirichlet_MMSE':
            #TODO:
            #if MLE_reversible estimator is implemented, it should be used here on self._T[:n, :n]
            raise NotImplementedError(f"transition estimator {self.config.transition_estimator}\
                                        not implemented, Only Dirichlet_MMSE is currently supported")


        #make the external vertices sinks
        self._T[self.n:, self.n:] = np.eye(n_external)
        linalg._assert_stochastic(self._T)
        # get the transition matrix in timestep resolution self.tau #TODO Move down 8 lines if it doesn't break anything.
        self._T_tau = np.linalg.matrix_power(self._T, self.tau) # TODO: rename _T with _T_1.

        # check if there are any children which should be one of my neighbors children instead
        vertices_to_disown = self._check_disown(id_2_index, n_external)
        if vertices_to_disown:
            self._T_is_updated = False
            return HierarchicalMSMVertex.DISOWN_CHILDREN, vertices_to_disown

        # update variables derived from self._T
        self._T_is_updated = True
        self._update_timescale()
        self._update_external_T()

        if self._check_split_condition():
            return HierarchicalMSMVertex.SPLIT, self._split()
        if self._check_parent_update_condition():
            return HierarchicalMSMVertex.UPDATE_PARENT, None
        return HierarchicalMSMVertex.SUCCESS, None

    def _get_id_2_index_map(self, column_ids):
        """Return a dict that maps each id to the index of its row/column in T, and the
        dimension of the full transition matrix.
        """
        id_2_parent_id = {}
        id_2_index = {}
        index_2_id = {}
        neighbors = set()
        n = self.n
        # get the parents of vertices in column_ids which arent me (my neighbors)
        for column_id in column_ids:
            if column_id in self.children:
                parent = self.id
            else:
                parent = self.tree.get_parent(column_id)
                id_2_parent_id[column_id] = parent
                if parent not in column_ids:
                    neighbors.add(parent)

        # my own children are the first n columns and rows:
        for j, id in enumerate(self.children):
            id_2_index[id] = j
            index_2_id[j] = id
        # the last columns and rows are my neighbors:
        for j, id in enumerate(neighbors):
            id_2_index[id] = j + n
            index_2_id[j+n] = id
        # map children of my neighbors to their parents:
        for id, parent_id in id_2_parent_id.items():
            id_2_index[id] = id_2_index[parent_id]
        full_n = len(index_2_id)
        return id_2_index, index_2_id, full_n

    def _check_disown(self, id_2_index, n_external):
        """
        Check if there are vertices among self.children that should be one of my neighbors children
        instead. If there are, return a dict mapping new parents to the children to move to them.
        """
        vertices_to_disown = defaultdict(list)
        for child in self.children:
            row = id_2_index[child]
            most_likely_parent = self._get_most_likely_parent(row, n_external)
            if most_likely_parent != self.id:
                # Double check, otherwise we get into loops of children being passed back and forth
                if self.id not in self._get_most_likely_parents_MC(child):
                    vertices_to_disown[most_likely_parent].append(child)

        if len(vertices_to_disown) > 0:
            return vertices_to_disown
        return False

    def _get_most_likely_parents_MC(self, child):
        """
        Sample 100 steps of length self.tau, starting from child, and return the most likely
        parents of a vertex 1 tau-step away from child.
        """
        parent_sample = []
        for _ in range(31): # maybe this number should depend on number of neighbors?
            step = child
            for __ in range(self.tau):
                next_states, transition_probabilities = self.tree.get_external_T(step)
                step = np.random.choice(next_states, p=transition_probabilities)
                parent_sample.append(self.tree.get_parent(step))
        parents, counts = np.unique(parent_sample, return_counts=True)
        return parents[np.where(counts==max(counts))]


    def _get_most_likely_parent(self, row, n_external):
        temp_row = np.ndarray(1+n_external)
        temp_row[0] = np.sum(self._T[row, :self.n]) - self._T[row, row]
        temp_row[1:] = self._T[row, self.n:]
        maximum = np.argmax(temp_row)
        if maximum > 0:
            # argmax returns the highest maximal index, if it's the same as 0, we want to keep it
            if temp_row[maximum] == temp_row[0]:
                return self.id
            new_parent = self.index_2_id[self.n + maximum - 1]
            return new_parent
        return self.id


    def get_local_stationary_distribution(self):
        n = self.n
        T = self.T[:n, :n]
        T = linalg.normalize_rows(T)
        return stationary_distribution(T)


    def get_external_T(self, tau=1) -> tuple:
        """get_external_T.
        Get the transition probabilities between this vertex and its neighbors at the same level
        of the tree.

        Parameters
        ----------
        tau : int
            the time-resolution of the transition probabilities
        Returns
        -------
        ids : ndarray
            An array of the ids of the neighbors of this vertex
        transition_probabilities : ndarray
            An array of the transition probabilities to the neighbors of this vertex, such that
            transition_probabilities[i] is the probability of getting to ids[i] in tau steps from
            this vertex.
        """
        ids, external_T = self._external_T
        ids = ids.copy()
        external_T = external_T.copy()
        if tau != 1:
            external_T[0] = np.power(external_T[0], tau) # The probability of no transition
            # The relative probabilities of the other transitions haven't changed, so normalize:
            external_T[1:] = (external_T[1:]/np.sum(external_T[:,1])) * (1-external_T[0])

        self._last_update_sent = [ids, external_T]
        return ids.copy(), external_T.copy()

    def _update_external_T(self):
        T = self._T # this is T(1) as opposed to self.T which is T(tau)
        n = self.n
        local_stationary = self.get_local_stationary_distribution()
        local_stationary = np.resize(local_stationary, T.shape[0])
        local_stationary[n:] = 0 # pad with 0's.

        full_transition_probabilities = local_stationary.dot(T)
        external_T = np.ndarray(T.shape[0] - self.n + 1)
        ids = np.ndarray(T.shape[0] - self.n + 1, dtype=int)

        # the probability of staying in this vertex in external_tau steps:
        external_T[0] = np.sum(full_transition_probabilities[:n])
        # the probability of getting to neighboring vertices:
        external_T[1:] = full_transition_probabilities[n:]

        ids[0] = self.id
        for j in range(1, ids.shape[0]):
            ids[j] = self.index_2_id[n+j-1]

        self._external_T = ids, external_T
        self._neighbors = ids
        #NOTE: the assumption underlying the whole concept, is that full_transition_probabilities[n:]
        # is similar to T_ext_tau[i, n:] for all i<n. In other words, that a with high probability,
        # a random walk mixes before leaving this MSM.
        # This could be used as validation, and maybe as a clustering criterion.


    def _check_split_condition(self):
        return self._partition_estimator.check_split_condition(self)

    def _get_partition(self):
        # return (partition, taus) where partition is an iterable of iterables of children_ids
        return self._partition_estimator.get_metastable_partition(self)

    def _split(self):
        index_partition, taus = self._get_partition()
        if len(index_partition)==1:
            warnings.warn("split was called, but no partition was found", RuntimeWarning) #TODO logger
        id_partition = [ [self.index_2_id[index] for index in subset] \
                            for subset in index_partition]

        return id_partition, taus, self, self.parent

    def sample_microstate(self, n_samples):
        """Get a set of microstate from this MSM, ideally chosen such that sampling a random walk
        from these microstates is expected to increase some objective function.

        Returns:
        -------
        samples : list
            A list of ids of microstates.
        """
        # a sample of n_samples vertices from this msm:
        sample = list(self._vertex_sampler(self, n_samples))
        if self.height == 1:
            return sample

        # now for each of those samples, get a sample from that vertex, the number of times
        # it appeared in sample
        vertices, counts = np.unique(sample, return_counts=True)
        recursive_sample = []
        for i, vertex in enumerate(vertices):
            recursive_sample += self.tree.sample_states(counts[i], vertex)
        return recursive_sample

    def sample_from_stationary(self):
        return np.random.choice(self.children, p=self.get_local_stationary_distribution())

    def get_all_microstates(self):
        """get_all_microstates.

        Returns
        -------
        microstates : set
            A set of all the ids of the microstates at the leaves of the subtree of which this
            vertex is the root.
        """
        if self.height == 1:
            return self.children
        microstates = set()
        for child in self.children:
            microstates.update(self.tree.get_microstates(child))
        return microstates

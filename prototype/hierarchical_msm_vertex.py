from collections import defaultdict
import warnings
import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from HMSM.util import util, linalg


class HierarchicalMSMVertex:
    """HierarchicalMSMVertex.
    This class represents a single vertex in an HMSM tree.
    A vertex has the ids of its children and its parent, as well as having a transition matrix
    describing transition probabilities between its children. This transition matrix is updated
    when changes to the tree structure affecting the vertex are made, or when the transition
    probabilities change due to new data.
    """

    SPLIT = "SPLIT"
    DISOWN_CHILDREN = "DISOWN_CHILDREN"
    UPDATE_PARENT = "UPDATE_PARENT"
    SUCCESS = "SUCCESS"

    def __init__(self, tree, children, parent, tau, height, config):
        """__init__.

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
            the timestep resolution of this MSM, in multiples of the timestep resolution of
            discrete trajectories given as input to the HierarchicalMSMTree in
            update_transition_counts.

            For example, if a Molecular Dynamics simulation is sampled in timesteps of 2e-15 seconds,
            resulting in (x_0, x_1, ..., x_n), and then one every tenth sample is taken (x_0, x_10,...)
            and discretized into microstates, resulting in (microstate_0, microstate_10, ...) and
            this trajectory is used as input to HierarchicalMSMTree, the timestep resolution of this
            discrete trajectory is 10*2e-15=2e-14 seconds. Now if this MSM has tau=5, then the
            timestep resolution of this MSM is 5*2e-14=1e-13 seconds.
        height : int
            the height of this vertex in the tree, where microstates have height 0.
        config : dict
            a dictionary of model parameters.
        """
        self.__id = util.get_unique_id()
        self.tree = tree
        self._children = set(children)
        if parent is None:
            self.parent = self.__id
        else:
            self.parent = parent
        self.tau = tau
        self.height = height
        self._neighbors = []


        self.config = config

        self._T_is_updated = False
        self._last_update_sent = None


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

    @property
    def parent_update_threshold(self):
        return self.config["parent_update_threshold"]

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
        return self._update_T()


    def _check_parent_update_condition(self):
        if self.is_root:
            return False
        if self._last_update_sent is None:
            return True
        if set(self._last_update_sent[0]) != set(self.get_external_T()[0]):
            return True

        max_change_factor = util.max_fractional_difference(self._last_update_sent, \
                                                      self.get_external_T())
        return max_change_factor >= self.parent_update_threshold

    def _update_T(self):
        assert self.n > 0
        T_rows = dict()
        column_ids = set(self.children)

        # Get the rows of T corresponding to each child
        for child in self._children:
            ids, row = self.tree.get_external_T(child)
            T_rows[child] = (ids, row)
            linalg._assert_stochastic(row, axis=0)
            column_ids = column_ids.union(ids)

        id_2_index, index_2_id, full_n = self._get_id_2_index_map(column_ids)
        self.index_2_id = index_2_id

        self._T = np.zeros((full_n, full_n))
        n_external = full_n - self.n # number of external vertices

        # Now fill the matrix self._T
        for child, (ids, T_row) in T_rows.items():
            row = id_2_index[child]
            for id, transition_probability in zip(ids, T_row):
                column = id_2_index[id]
                self._T[row, column] += transition_probability

        #make the external vertices sinks
        self._T[self.n:, self.n:] = np.eye(n_external)
        linalg._assert_stochastic(self._T)
        # get the transition matrix in timestep resolution self.tau
        self._T_tau = np.linalg.matrix_power(self._T, self.tau)

        # check if there are any children which should be one of my neighbors children instead
        vertices_to_disown = self._check_disown(id_2_index, n_external)
        if vertices_to_disown:
            self._T_is_updated = False
            return HierarchicalMSMVertex.DISOWN_CHILDREN, vertices_to_disown
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
            parent = self.tree.get_parent(column_id)
            if parent != self.id:
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
        vertices_to_disown = defaultdict(list)
        for child in self.children:
            row = id_2_index[child]
            most_likely_parent = self._get_most_likely_parent(self._T[row], n_external)
            if most_likely_parent != self.id:
                # Double check, otherwise we get into loops of children being passed back and forth
                if self.id not in self._get_most_likely_parents_MC(child):
                    vertices_to_disown[most_likely_parent].append(child)

        if len(vertices_to_disown) > 0:
            return vertices_to_disown
        return False

    def _get_most_likely_parents_MC(self, child):
        parent_sample = []
        for i in range(100):
            step = child
            for j in range(self.tau):
                next_states, transition_probabilities = self.tree.get_external_T(step)
                step = np.random.choice(next_states, p=transition_probabilities)
            parent_sample.append(self.tree.get_parent(step))
        parents, counts = np.unique(parent_sample, return_counts=True)
        return parents[np.where(counts==max(counts))]



    def _get_most_likely_parent(self, row, n_external):
        temp_row = np.ndarray(1+n_external)
        temp_row[0] = np.sum(row[:self.n])
        temp_row[1:] = row[self.n:]
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

    def get_external_T(self, ext_tau=1) -> tuple:
        """get_external_T.

        Parameters
        ----------
        self :
            self
        ext_tau :
            The timestep resolution of the transition probabilities (see return value)

        Returns
        -------
        (ids, external_T) ids and external_T are both arrays, such that external_T[i] is the
        probability of a transition from this vertex to the vertex with id ids[i] in tau steps.
        """
        # returns: ext_T: an (m,2) array, such that if ext_T[j] = v,p, then p is the probability
        # of a transition from this vertex to the vertex v.

        ids, external_T = self._external_T
        ids = ids.copy()
        external_T = external_T.copy()
        if ext_tau != 1:
            external_T[0] = np.power(external_T[0], ext_tau) # The probability of no transition
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
        self._neighbors = ids[1:]

        #NOTE: the assumption underlying the whole concept, is that full_transition_probabilities[n:]
        # is similar to T_ext_tau[i, n:] for all i<n. In other words, that a with high probability,
        # a random walk mixes before leaving this MSM.
        # This could be used as validation, and maybe as a clustering criterion.


    def _check_split_condition(self):
        return self.config["split_condition"](self)

    def _get_partition(self):
        # return (partition, taus) where partition is an iterable of iterables of children_ids
        return self.config["split_method"](self)

    def _split(self):
        index_partition, taus = self._get_partition()
        if len(index_partition)==1:
            warnings.warn(f"split was called, but {self.config['split_method']} was unable to \
                            find a partition", RuntimeWarning)
        id_partition = [ [self.index_2_id[index] for index in subset] \
                            for subset in index_partition]
        
        return id_partition, taus, self, self.parent

    def sample_microstate(self, n_samples):
        """Get a microstate from this MSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        # a sample of n_samples vertices from this msm:
        sample = [self.config["sample_method"](self) for _ in range(n_samples)]
        if self.height == 1:
            return sample

        # now for each of those samples, get a sample from that vertex, the number of times
        # it appeared in sample
        vertices, counts = np.unique(sample, return_counts=True)
        recursive_sample = []
        for i, vertex in enumerate(vertices):
            recursive_sample += self.tree.sample_microstate(counts[i], vertex)
        return recursive_sample

    def get_all_microstates(self):
        if self.height == 1:
            return self.children
        microstates = set()
        for child in self.children:
            microstates.update(self.tree.get_microstates(child))
        return microstates

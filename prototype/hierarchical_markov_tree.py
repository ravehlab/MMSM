import warnings
import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from HMSM.util import util, linalg

all = ["HierarchicalMarkovTree", "HierarchicalMSM"]

def _assert_valid_config(config):
    min_keys = ["split_condition", "split_method", "sample_method", "parent_update_threshold"]
    for key in min_keys:
        if config.get(key) is None:
            raise ValueError(f"config dictionary missing required value: {key}")

def max_fractional_difference(ext_T1, ext_T2):
    #TODO put this somewhere better
    assert ext_T1.shape == ext_T2.shape, "external_T has changed shape since last update"
    assert ext_T1.shape[1] == 2
    sort1 = np.argsort(ext_T1[:,0]) # sort by ids
    sort2 = np.argsort(ext_T2[:,0]) # sort by ids
    max_diff = 0

    for i in range(len(sort1)):
        p1 = ext_T1[sort1[i], 1]
        p2 = ext_T2[sort2[i], 1]
        diff = np.abs(1-(p1/p2))
        max_diff = np.max(max_diff, diff)

    return max_diff


class HierarchicalMarkovTree:

    def __init__(self, config):
        self._microstate_parents = dict()
        self._microstate_counts = util.count_dict(depth=2)
        self._microstate_MMSE = dict()
        self._last_update_sent = dict()
        _assert_valid_config(config)
        self.config = config

        self.vertices = dict()

        self.alpha = 1 # parameter of the dirichlet prior

        self.height = 1
        self.root = HierarchicalMSM(self, children=set(),\
                                           parent=None,\
                                           tau=1,\
                                           height=1,
                                           config=config)

    def _assert_valid_vertex(self, vertex_id):
        return self._is_microstate(vertex_id) or \
               (self.vertices.get(vertex_id) is not None)

    def _is_root(self, vertex_id):
        return vertex_id==self.root.id

    def _is_microstate(self, vertex_id):
        if self._microstate_parents.get(vertex_id) is None:
            return False
        else:
            return True

    def _init_unseen_vertex(self):
        self.unseen_id = util.get_unique_id()
        self._microstate_parents[self.unseen_id] = self.unseen_id
        self._microstate_counts = np.array([self.unseen_id, 1])

    def set_parent(self, child_id, parent_id):
        if not self._assert_valid_vertex(parent_id):
            raise ValueError("Got parent_id for non-existing id")
        if not self._assert_valid_vertex(child_id):
            raise ValueError("Got child_id for non-existing id")

        if self._is_microstate(child_id):
            self._microstate_parents[child_id] = parent_id
        else:
            self.vertices[child_id].parent = parent_id

    def get_parent(self, child_id):
        if not self._assert_valid_vertex(child_id):
            raise ValueError("Got child_id for non-existing id")

        if self._is_microstate(child_id):
            return self._microstate_parents[child_id]
        else:
            return self.vertices[child_id].parent.id


    def get_external_T(self, vertex_id, tau=1):
        if self._is_microstate(vertex_id):
            assert tau==1
            ext_T = self._microstate_MMSE[vertex_id]
            self._last_update_sent[vertex_id] = ext_T
            return ext_T
        else:
            return self.vertices[vertex_id].get_external_T(tau)

    def update_transition_counts(self, dtrajs, update_MMSE=True):
        updated_microstates = set()
        for dtraj in dtrajs:
            updated_microstates.update(dtraj)
            src = dtraj[0]
            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                self._microstate_counts[src, dst] += 1
                if self._microstate_counts[dst, src] == 0:
                    self._microstate_counts[dst, src] = self.alpha
                src = dst

        if update_MMSE:
            for vertex_id in updated_microstates:
                self._microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)

        # add any newly discovered microstates
        for vertex_id in updated_microstates:
            if not self._is_microstate(vertex_id):
                self._get_new_microstate_parent(vertex_id)

        # update from the leaves upwards
        for vertex_id in updated_microstates:
            if self._check_parent_update_condition(vertex_id):
                parent_id = self._microstate_parents[vertex_id]
                self.vertices[parent_id].update_T()

    def _dirichlet_MMSE(self, vertex_id):
        MMSE_ids = np.array(list(self._microstate_counts[vertex_id].keys()))
        MMSE_counts = np.array(list(self._microstate_counts[vertex_id].values())) + self.alpha
        MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
        return np.array([MMSE_ids, MMSE_counts]).T

    def _get_new_microstate_parent(self, vertex_id):
        if self.height == 1:
            self._microstate_parents[vertex_id] = self.root.id
        if self._microstate_MMSE.get(vertex_id) is None:
            self._microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)
        # sample a random walk on the microstates, until reaching one with a parent, and join
        # that one. Collect any other orphans we find along the way.
        orphans = set()
        next_state = vertex_id
        parent = None
        while parent is None:
            orphans.add(next_state)
            next_state = self._random_step(next_state)
            parent = self._microstate_parents.get(next_state)
        for orphan in orphans:
            self._microstate_parents[orphan] = parent

    def _check_parent_update_condition(self, microstate):
        if self._last_update_sent.get(microstate) is None:
            return True
        else:
            max_change_factor = max_fractional_difference(self._last_update_sent[microstate], \
                                                          self._microstate_MMSE[microstate])
            return max_change_factor >= self.config["parent_update_threshold"]


    def add_vertex(self, vertex):
        assert vertex.tree is self
        assert isinstance(vertex, HierarchicalMSM)
        self.vertices[vertex.id] = vertex

    def add_children(self, parent_id, child_ids):
        if not self._assert_valid_vertex(parent_id):
            raise ValueError("Got parent_id for non-existing id")
        for child_id in child_ids:
            if not self._assert_valid_vertex(child_id):
                raise ValueError("Got child_id for non-existing id")
            self.vertices[parent_id].add_child(child_id)

    def remove_vertex(self, vertex_id):
        assert not self._is_microstate(vertex_id)
        #TODO maybe add assertion that there are no orphans leftover

        if self._is_root(vertex_id):
            for child_id in self.root.children:
                self.update_vertex(child_id)
            self.root.update()
            self.height += 1
            self.root.height = self.height
        else:
            parent_id = self.vertices[vertex_id].parent
            parent = self.vertices[parent_id]
            parent.remove_child(vertex_id)
            del self.vertices[vertex_id]
            self.update_vertex(parent_id)

    def update_vertex(self, vertex_id):
        assert not self._is_microstate(vertex_id)
        vertex = self.vertices[vertex_id]
        vertex.update()

    def sample_random_walk(self, length, start=None):
        sample = np.ndarray(length)
        if start is None:
            start = np.random.choice(list(self._microstate_MMSE.keys()))
        current = start
        for i in range(length):
            current = self._random_step(current)
            sample[i] = current
        return sample

    def _random_step(self, microstate_id):
        next_states, transition_probabilities = self._microstate_MMSE[microstate_id]
        return np.random.choice(next_states, p=transition_probabilities)

    def sample_microstate(self, vertex_id=None):
        """Get a microstate from this HMSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        if vertex_id is None:
            vertex_id = self.root.id
        return self.vertices[vertex_id].sample_vertex()

class HierarchicalMSM:

    def __init__(self, tree, children, parent, tau, height, config):
        self.tree = tree
        self._children = children # should be a set
        self.parent = parent
        self.tau = tau
        self.height = height
        self.__id = util.get_unique_id()

        self.config = config

        self._T_is_updated = False

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

    @property
    def is_root(self):
        return self.parent == self.id

    @property
    def parent_update_threshold(self):
        return self.config["parent_update_threshold"]

    def add_child(self, child_id):
        self._children.add(child_id)
        self.tree.set_parent(child_id, self.id)
        self._T_is_updated = False

    def remove_child(self, child_id):
        self._children.discard(child_id)
        for child in self._children:
            self.tree.update_vertex(child)
        self.update()

    def update(self):
        self._update_T()
        if self._check_parent_update_condition():
            self.tree.update_vertex(self.parent)

    def _check_parent_update_condition(self):
        if self._last_update_sent is None:
            return True
        else:
            max_change_factor = max_fractional_difference(self._last_update_sent, \
                                                          self.get_external_T())
            return max_change_factor >= self.parent_update_threshold


    def _update_T(self):
        T_rows = []
        T_ids = []
        column_ids = set(self.children)

        # Get the rows of T corresponding to each child
        for child in self._children:
            ids, row = self.tree.get_external_T(child)
            T_rows.append(row)
            T_ids.append(ids)
            column_ids = column_ids.union(ids)

        id_2_index, full_n = self._get_id_2_index_map(column_ids)
        self._T = np.zeros((full_n, full_n))

        # Now fill the matrix self._T
        for i, T_row in enumerate(T_rows):
            for _j, i_2_j_probability in enumerate(T_row):
                j = id_2_index[ T_ids[i][_j] ]
                self._T[i,j] = i_2_j_probability
        self._T_is_updated = True #Set this to false to always recalculate T

        #make the external vertices sinks
        n_external = full_n - self.n
        self._T[self.n:, self.n:] = np.eye(n_external)
        self._T_tau = np.matrix_power(self._T, self.tau)

        if self._check_split_condition():
            self.split()
        else:
            #save the ids of the indices
            self.index_2_id = util.inverse_dict(id_2_index)

        self._update_timescale()
        self._update_external_T()

    def _get_id_2_index_map(self, column_ids):
        """Return a dict that maps each id to the index of its row/column in T, and the
        dimension of the full transition matrix.
        """
        id_2_parent_id = {}
        id_2_index = {}
        for column_id in column_ids:
            parent = self.tree.get_parent(column_id)
            if parent != self.id:
                column_ids.discard(column_id)
                id_2_parent_id[column_id] = parent
                if parent not in column_ids:
                    column_ids.add(parent)
        for j, id in enumerate(column_ids):
            id_2_index[id] = j
        for id, parent_id in id_2_parent_id.items():
            id_2_index[id] = id_2_index[parent_id]
        full_n = len(column_ids)
        return id_2_index, full_n

    def _update_timescale(self):
        self._timescale = linalg.get_longest_timescale(self.T, self.tau)


    def get_local_stationary_distribution(self):
        n = self.n
        T = self.T[:n, :n]
        T = linalg.normalize_rows(T)
        return stationary_distribution(T)

    def get_external_T(self, ext_tau=1):
        # returns: ext_T: an (m,2) array, such that if ext_T[j] = v,p, then p is the probability
        # of a transition from this vertex to the vertex v.

        external_T = self._external_T.copy()
        if ext_tau != 1:
            transitions = external_T[:1].T # just the transition probabilities, as a row vector
            transitions[0] = np.power(transitions[0], ext_tau) # The probability of no transition
            # The relative probabilities of the other transitions haven't changed, so normalize:
            transitions[1:] = transitions[1:]/np.sum(transitions[:,1]) * (1-transitions[0])
            #put the result back:
            external_T[:1] = transitions.T

        self._last_update_sent = external_T.copy()
        return external_T

    def _update_external_T(self):
        assert self._T_is_updated
        T = self._T # this is T(1) as opposed to self.T which is T(tau)
        n = self.n
        local_stationary = self.get_local_stationary_distribution()
        local_stationary.resize(T.shape[0]) # pad with 0's.

        full_transition_probabilities = local_stationary.dot(T)
        external_T = np.ndarray(T.shape[0]-self.n + 1)
        ids = np.ndarray(T.shape[0]-self.n + 1)

        # the probability of staying in this vertex in external_tau steps:
        external_T[0] = np.sum(full_transition_probabilities[:n])
        # the probability of getting to neighboring vertices:
        external_T[1:] = full_transition_probabilities[n:]

        ids[0] = self.id
        for j in range(n, T.shape[0]):
            ids[j] = self.index_2_id[j]

        self._external_T = np.stack([ids, external_T]).T

        #NOTE: the assumption underlying the whole concept, is that full_transition_probabilities[n:]
        # is similar to T_ext_tau[i, n:] for all i<n. In other words, that a with high probability,
        # a random walk mixes before leaving this MSM.
        # This could be used as validation, and maybe as a clustering criterion.


    def _check_split_condition(self):
        return self.config["split_condition"](self)

    def _get_partition(self):
        # return (partition, taus) where partition is an iterable of iterables of children_ids
        return self.config["split_method"](self)

    def _get_new_vertex_with(self, children_ids, tau):
        vertex = HierarchicalMSM(self.tree, children_ids, self.parent, \
                                 tau, self.height, self.config)
        self.tree.add_vertex(vertex)
        self.tree.vertices[self.parent].add_child(vertex.id)
        for child_id in children_ids:
            self.tree.set_parent(child_id, vertex.id)
        return vertex.id

    def split(self):
        # 1. get partition compatible with max_timescale
        # 2. create new vertices according to this partition
        # 3. update tree structure
        partition, taus = self._get_partition()

        if len(partition)==1:
            warnings.warn(f"split was called, but {self.config.split_method} was unable to \
                            find a partition", RuntimeWarning)
            return

        new_vertices = []
        if self.is_root():
            self._children = set()
        for i, children_ids in enumerate(partition):
            new_vertex = self._get_new_vertex_with(children_ids, taus[i])
            new_vertices.append(new_vertex)
        self.tree.add_children(self.parent, new_vertices)
        # this will trigger all the vertices on this level (the newly created ones, and the already
        # existing siblings) and the parent to update.
        self.tree.remove_vertex(self.id)

    def sample_microstate(self):
        """Get a microstate from this MSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        sample = self.config["sample_method"](self)
        if self.height == 1:
            return sample
        else:
            return self.tree.sample_microstate(sample)

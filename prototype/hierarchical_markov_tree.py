from HMSM.util import get_unique_id

all = ["HierarchicalMarkovTree", "HierarchicalMSM"]

class HierarchicalMarkovTree:

    def __init__(self):
        self.microstate_parents = dict()
        self.microstate_counts = util.count_dict(depth=2)
        self.microstate_MMSE = dict()

        self.vertices = dict()

        self.alpha = 1 # parameter of the dirichlet prior

    def _assert_valid_vertex(self, vertex_id):
        return self._is_microstate(vertex_id) or \
               (self.vertices.get(vertex_id) is not None):

    def _is_microstate(self, vertex_id):
        if self.microstate_parents.get(vertex_id) is None:
            return False
        else:
            return True

    def _init_unseen_vertex(self):
        self.unseen_id = get_unique_id()
        self.microstate_parents[self.unseen_id] = self.unseen_id
        self.microstate_counts = np.array([self.unseen_id, 1])

    def set_parent(self, child_id, parent_id):
        if not self._assert_valid_vertex(parent_id):
            raise ValueError("Got parent_id for non-existing id")
        if not self._assert_valid_vertex(child_id):
            raise ValueError("Got child_id for non-existing id")

        if self._is_microstate(child_id):
            self.microstate_parents[child_id] = parent_id
        else:
            self.vertices[child_id].parent = parent_id

    def get_parent(self, child_id):
        if not self._assert_valid_vertex(child_id):
            raise ValueError("Got child_id for non-existing id")

        if self._is_microstate(child_id):
            return self.microstate_parents[child_id]
        else:
            return self.vertices[child_id].parent.id


    def get_external_T(self, vertex_id):
        if self._is_microstate(vertex_id):
            return self.microstate_MMSE[vertex_id]
        else:
            return self.get_vertex(vertex_id).get_external_T()

    def update_transition_counts(self, dtrajs, update_MMSE=True):
        updated_microstates = set()
        for dtraj in dtrajs:
            src = dtraj[0]
            updated_microstates.add(src)
            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                self.microstate_counts[src, dst] += 1
                src = dst

        if update_MMSE:
            for vertex_id in updated_microstates:
                self.microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)

        for vertex_id in updated_microstates:
            if self._check_parent_update_condition(vertex_id):
                parent_id = self.microstate_parents(vertex_id)
                self.vertices[parent_id].update_T()

    def _dirichlet_MMSE(self, vertex_id):
            MMSE_ids = np.array(list(self.microstate_counts[vertex_id].keys()))
            MMSE_counts = np.array(list(self.microstate_counts[vertex_id].values())) + self.alpha
            MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
            return np.array([MMSE_ids, MMSE_counts])

    def _check_parent_update_condition(self, microstate):
        pass

    def add_vertex(self, vertex):
        assert vertex.tree is self
        assert isinstance(vertex, hierarchicalMSM)
        self.vertices[vertex.id] = vertex

    def remove_vertex(self, vertex_id):
        assert not self._is_microstate(vertex_id)
        #TODO maybe add assertion that there are no orphans leftover

        parent_id = self.vertices[vertex_id].parent
        parent = self.vertices[parent_id]
        parent.remove_child(vertex_id)
        del self.vertices[vertex.id]
        self.update_vertex(parent_id)

    def update_vertex(self, vertex_id):
        assert not self._is_microstate(vertex_id)
        vertex = self.vertices[vertex_id]
        vertex.update()

class hierarchicalMSM:

    def __init__(self, tree, children, parent, tau, split_condition=None, split_method=None, sample_method=None):
        self.tree = tree
        self._children = children # should be a set
        self.parent = parent
        self.tau = tau
        self.__id = util.get_unique_id()

        self.split_condition = split_condition
        self.split_method = split_method
        self.sample_method = sample_method

    @property
    def children(self):
        return self._children.copy()

    @property
    def id(self):
        return self.__id

    @property
    def T(self)
        assert self._T_is_updated
        return self._T

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_id):
        self._parent = parent_id

    def add_child(self, child_id):
        self._children.add(child_id)
        self.tree.set_parent(self.id)
        self._T_is_updated = False

    def remove_child(self, child_id):
        self._children.discard(child_id)
        for child in self._children:
            self.tree.update_vertex(child)
        self.update()

    def update(self, update_parent=True):
        self._update_T()
        if update_parent and self._check_parent_update_condition():
            self.tree.update_vertex(self.parent)

    def _update_T(self):
        # T_rows is a list of np arrays s.t. T_rows[i][j, 1] is the probability of a transition i->j where j is the vertex with id T_rows[i][j][0]
        T_rows = [tree.get_external_T(child, self.tau) for child in self._children]
        column_ids = self._children.union([T_row[0] for T_row in T_rows])
        id_2_index, n_cols = self._get_id_2_index_map(column_ids)
        n_rows = n_cols
        self._T = np.zeros((n_rows, n_cols))

        #TODO maybe vectorize this?
        for i, T_row in enumerate(T_rows):
            for _j, id in enumerate(T_row[0]):
                j = id_2_index[id]
                self_T[i,j] = T_row[1,_j]
        self._T_is_updated = True #Set this to false to always recalculate T

        #make the external vertices sinks
        n_external = n_cols - self.n
        self_T[self.n:, self.n:] = np.eye(n_external)

        if self._check_split_condition():
            self.split()
        else:
            #save the ids of the indices
            self.index_2_id = util.inverse_dict(id_2_index)

    def _get_id_2_index_map(self, column_ids): #TODO column_ids is a set, need to update this function accordingly
        id_2_parent_id = dict()
        ids_to_remove = []
        for j, column_id in enumerate(column_ids):
            parent = get_parent(column_id)
            if parent != self.id:
                # remove them at the end to not change indices while looping
                ids_to_remove.append(j)
                id_2_parent_id[column_id] = parent
                if parent not in column_ids:
                    column_ids.append(parent)
        for j in range(len(ids_to_remove)): #TODO pop(j) is O(N), we can make this faster
            # remove in reverse order to not change indices of the ones we haven't removed yet
            id_to_remove = ids_to_remove[-(j+1)]
            column_ids.pop(id_to_remove)
        id_2_index = dict()
        for j, id in enumerate(column_ids):
            id_2_index[id] = j
        for id in id_2_parent_id:
            id_2_index[id] = id_2_index[id_2_parent_id[id]]
        n_cols = column_ids.shape[0]
        return id_2_index, n_cols

    def get_external_T(self, external_tau):
        # returns an (m,2) array as specified in _update_T. This should be the MMSE estimator of the row.
        T = self.T
        n = self.n
        local_stationary = self.get_local_stationary_distribution() #TODO implement this
        local_stationary.resize(T.shape[0]) # pad with 0's. TODO make sure there are no references to local_stationary
        # T on the timescale of external_tau:
        T_ext_tau = np.linalg.matrix_power(T, external_tau//self.tau)
        # the distribution over states in external_tau steps, starting from the
        # local stationary distribution:
        full_transition_probabilities = local_stationary.dot(T_ext_tau)

        #NOTE: the assumption underlying the whole concept, is that full_transition_probabilities[n:]
        # is similar to T_ext_tau[i, n:] for all i<n. In other words, that a with high probability,
        # a random walk mixes before leaving this MSM.
        # This could be used as validation, and maybe as a clustering criteria.

        external_T = np.ndarray((2, T.shape[0]-self.n+1))
        # the probability of staying in this vertex in external_tau steps:
        external_T[0,0] = self.id
        external_T[0,1] = np.sum(full_transition_probabilities[:n])

        external_T[1:,1] = full_transition_probabilities[n:]
        for j in range(n, T.shape[0]):
            external_T[j, 0] = self.index_2_id[j]

        return external_T
    def _check_split_condition(self):
        pass

    def _get_partition(self):
        # return (partition, taus) where partition is an iterable of iterables of children_ids
        # use self.split_method. Need to decide what parameters it will take
        pass

    def split(self):
        # 1. get partition compatible with max_timescale
        # 2. create new vertices according to this partition
        # 3. update tree structure
        partition, taus = self._get_partition()
        new_vertices = []
        for i, children_ids in enumerate(partition):
            vertex = hierarchicalMSM(self.tree, children_ids, self.parent, taus[i])
            self.tree.add_vertex(vertex)
            self.tree.vertices[self.parent].add_child(vertex.id)
            for child_id in children_ids:
                self.tree.set_parent(child_id, vertex.id)
            new_vertices.append(vertex)

        # this will trigger all the vertices on this level (the newly created ones, and the already
        # existing siblings) and the parent to update.
        self.tree.remove_vertex(self.id)



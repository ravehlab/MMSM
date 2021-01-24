from HMSM.util import get_unique_id

class HierarchicalMarkovTree:

    # each vertex is an object. Has a pointer to it's children and its parent

    # each vertex needs to have a pointer to its parent and all of its children, and needs to be aware (through its parent or children probably) of its siblings.
    # through its parent - seems to make intuitive sense, structuraly.
    # through its children - they know about the actual transition probability. If v is my child, and v->u, where v is my nephew, when I get v's transition probabilities,
    # it will tell me about u. Since u isnt my child, I will ask it "who's your daddy?", then add an edge from v to u's parents vertex.
    def __init__(self):
        self.microstate_parents = dict()
        self.microstate_counts = util.count_dict(depth=2) #TODO debug this
        self.microstate_MMSE = dict()
        self.cluster_indices = [] #cluster_indices[j] is the id of the j'th cluster
        self.microstate_indices = dict() #microstate_indices[id]==j iff cluster_indices[j]==id

        self.vertices = dict()

        self.alpha = 1 # parameter of the dirichlet prior

    def _is_microstate(self, vertex_id):
        if self.microstate_parents.get(vertex_id) is None:
            return False
        else:
            return True

    def _init_unseen_vertex(self):
        self.unseen_id = get_unique_id()
        self.microstate_parents[self.unseen_id] = self.unseen_id
        self.microstate_counts = np.array([self.unseen_id, 1])

    def get_external_T(self, vertex_id):
        if self.microstate_parents.get(vertex_id): # if this is a microstate
            return self.microstate_MMSE[vertex_id]
        else:
            return self.get_vertex(vertex_id).get_external_T

    def update_transition_counts(self, dtrajs, dt=1, update_MMSE=True):
        #TODO what dt are the dtrajs in? 
        updated = set()
        for dtraj in dtrajs:
            src = dtraj[0]
            updated.add(src)
            for i in range(1, len(dtraj), dt):
                dst = dtraj[i]
                self.microstate_counts[src, dst] += 1
                src = dst

        if update_MMSE:
            for vertex_id in updated:
                self.microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)

    def _dirichlet_MMSE(self, vertex_id):
            MMSE_ids = np.array(list(self.microstate_counts[vertex_id].keys()))
            MMSE_counts = np.array(list(self.microstate_counts[vertex_id].values())) + self.alpha
            MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
            return np.array([MMSE_ids, MMSE_counts])

    def add_vertex(self, vertex):
        assert vertex.tree is self 
        assert isinstance(vertex, hierarchicalMSM)
        self.vertices[vertex.id] = vertex

    def remove_vertex(self, vertex_id):
        assert not self._is_microstate(vertex_id)

        parent = self.vertices[vertex_id].parent
        del self.vertices[vertex.id]
        del parent.children[vertex_id]

        parent._T_is_updated = False
        for child_id in parent.children:
            self.vertices[child_id]._T_is_updated = False
 
class hierarchicalMSM:

    def __init__(self, tree, children, parent, tau=None):
        self.tree = tree
        self.children = children #TODO inform children
        self.parent = parent #TODO inform parent - parent needs to add my id to its children
        self.tau = tau

    @property
    def T(self)
        #TODO level0 and level>0 handle this differently, although maybe this just should never be called on a level0 MSM.
        if self._T_is_updated:
            return self._T
        else:
            return self._update_T

    def _update_T(self):
        # T_rows is a list of np arrays s.t. T_rows[i][j, 1] is the probability of a transition i->j where j is the vertex with id T_rows[i][j][0] 
        T_rows = [tree.get_external_T(child, self.tau) for child in self.children] 
        column_ids = list(np.unique([T_row[0] for T_row in T_rows]))
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

        #save the ids of the indices
        self.index_2_id = util.inverse_dict(id_2_index)
        return self._T
    
    def _get_id_2_index_map(self, column_ids):
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
        local_stationary = self.get_local_stationary_distribution()
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

    def split(self, max_timescale):
        # do PCCA to get partition compatible with max_timescale
        # create new vertices according to this partition
        # inform parent of change
        partition = pass #a list of lists of ids
        for children_ids in partition:
            self.tree.add_vertex(hierarchicalMSM(self.tree, children_ids, self.parent))
        self.tree.remove_vertex(self.id) #when this happens, all my siblings need to set _T_is_updated = False 



import pdb
from collections.abc import Iterable
import warnings
import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from HMSM.util import util, linalg

all = ["HierarchicalMSMTree", "HierarchicalMSMVertex"]

def _assert_valid_config(config):
    min_keys = ["split_condition", "split_method", "sample_method", "parent_update_threshold"]
    for key in min_keys:
        if config.get(key) is None:
            raise ValueError(f"config dictionary missing required value: {key}")

def max_fractional_difference(ext_T1, ext_T2):
    #TODO put this somewhere better
    ids1, T1 = ext_T1
    ids2, T2 = ext_T2
    assert T1.shape == T2.shape, "external_T has changed shape since last update"
    sort1 = np.argsort(ids1) # sort by ids
    sort2 = np.argsort(ids2) # sort by ids
    max_diff = 0

    for i in range(len(sort1)):
        p1 = T1[sort1[i]]
        p2 = T2[sort2[i]]
        diff = np.abs(1-(p1/p2))
        max_diff = max(max_diff, diff)

    return max_diff


class HierarchicalMSMTree:
    """HierarchicalMSMTree.
    This class encapsulates the HMSM data structure, and provides an interface for it.
    The vertices of the tree are HierarchicalMSMVertex objects, which can be accessed from this
    class.
    Most importantly, the method "update_transition_counts" allows the user to simply input
    observed discrete trajectories, (in the form of an iterable of state ids), and the estimation
    of a Hierarchical Markov Model is entirely automated - including estimating a hierarchy of
    metastable states and transition rates between them in different timescales.
    """


    def __init__(self, config):
        """__init__.

        Parameters
        ----------
        config : dict
            A dictionary of model parameters.
        """
        self._microstate_parents = dict()
        self._microstate_counts = util.count_dict(depth=2)
        self._microstate_MMSE = dict()
        self._last_update_sent = dict()
        _assert_valid_config(config) #TODO maybe assign default values for missing values instead?
        self.config = config

        self.vertices = dict()

        #TODO move to config
        self.alpha = 1 # parameter of the dirichlet prior


        #self._init_unseen_vertex()
        self._init_root()

    def _assert_valid_vertex(self, vertex_id):
        return self._is_microstate(vertex_id) or \
               (self.vertices.get(vertex_id) is not None)

    def _is_root(self, vertex_id):
        return vertex_id==self.root

    def _is_microstate(self, vertex_id):
        if self._microstate_parents.get(vertex_id) is None:
            return False
        return True

    def _init_unseen_vertex(self):
        """
        Create the "unseen vertex" - a stand in for all possible but unobserved transitions
        """
        self.unseen_id = util.get_unique_id()
        self._microstate_parents[self.unseen_id] = self.unseen_id
        self._microstate_counts[self.unseen_id][self.unseen_id] = 1

    def _init_root(self):
        root = HierarchicalMSMVertex(self, children=set(),\
                                           parent=None,\
                                           tau=1,\
                                           height=1,
                                           config=self.config)
        self.add_vertex(root)
        self.root = root.id

    @property
    def height(self):
        return self.vertices[self.root].height

    def get_parent(self, child_id):
        """
        Get the id of a vertex's parent vertex

        Parameters
        ----------
        child_id : int
            id of the vertex
        Returns
        -------
        parent_id : int
            the id of the vertex's parent
        """
        #if not self._assert_valid_vertex(child_id):
        #    raise ValueError("Got child_id for non-existing id (%d)"%child_id)

        if self._is_microstate(child_id):
            return self._microstate_parents[child_id]
        return self.vertices[child_id].parent


    def get_external_T(self, vertex_id, tau=1):
        """
        Get the transition probabilities between this vertex and other vertexs on the same level
        of the tree.

        Parameters
        ----------
        vertex_id : int
            the id of the vertex
        tau : int
            the time-resolution of the transitions
        Returns
        -------
        ext_T : np.ndarray
            an (m,2) array such that if ext_T[j] = [v,p] then p is the probability of a transition
            from this vertex to the vertex with id v in tau steps
        """
        if self._is_microstate(vertex_id):
            assert tau==1
            ids, ext_T = self._microstate_MMSE[vertex_id]
            self._last_update_sent[vertex_id] = [ids, ext_T]
            return ids, ext_T
        return self.vertices[vertex_id].get_external_T(tau)

    def update_transition_counts(self, dtrajs, update_MMSE=True):
        """
        Update the Hierarchical MSM with data from observed discrete trajectories.

        Parameters
        ----------
        dtrajs : iterable
            an iterable of ids of states
        update_MMSE : bool
            if True, update the estimated transition probabilities of each vertex that was observed
        """
        updated_microstates = set()
        for dtraj in dtrajs:
            updated_microstates.update(dtraj)
            src = dtraj[0]
            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                self._microstate_counts[src][dst] += 1
                if self._microstate_counts[dst][src] == 0:
                    self._microstate_counts[dst][src] = self.alpha
                src = dst

        if update_MMSE:
            for vertex_id in updated_microstates:
                self._microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)

        # add any newly discovered microstates
        new_microstates_2_parents = []
        for vertex_id in updated_microstates:
            if self._microstate_parents.get(vertex_id) is None:
                orphans, parent = self._get_new_microstate_parent(vertex_id)
                new_microstates_2_parents.append((orphans, parent))


        self._assign_children_to_parents(new_microstates_2_parents)
        # update from the leaves upwards
        for vertex_id in updated_microstates:
            if self._check_parent_update_condition(vertex_id):
                parent_id = self._microstate_parents[vertex_id]
                self.vertices[parent_id].update()

    def _dirichlet_MMSE(self, vertex_id):
        """
        Get the equivalent of external_T, but for a microstate.
        """
        MMSE_ids = np.array(list(self._microstate_counts[vertex_id].keys()), dtype=int)
        MMSE_counts = np.array(list(self._microstate_counts[vertex_id].values())) + self.alpha
        MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
        return (MMSE_ids, MMSE_counts)

    def _get_new_microstate_parent(self, vertex_id):
        """
        Set the most likely parent of a newly discovered microstate.
        """
        orphans = set([vertex_id])
        if self.height == 1:
            parent = self.root
        else:
            if self._microstate_MMSE.get(vertex_id) is None:
                self._microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)
            # sample a random walk on the microstates, until reaching one with a parent, and join
            # that one. Collect any other orphans we find along the way.
            next_state = vertex_id
            parent = None
            while parent is None:
                orphans.add(next_state)
                next_state = self._random_step(next_state)
                parent = self._microstate_parents.get(next_state)

        for orphan in orphans:
            self._microstate_parents[orphan] = parent
        return orphans, parent

    def _check_parent_update_condition(self, microstate):
        if self._last_update_sent.get(microstate) is None:
            return True

        max_change_factor = max_fractional_difference(self._last_update_sent[microstate], \
                                                      self._microstate_MMSE[microstate])
        return max_change_factor >= self.config["parent_update_threshold"]

    def _disconnect_from_parents(self, children_ids, parent_id):
        """This function should ONLY be called by set_parent
        """
        parent_2_children = dict()
        for child_id in children_ids:
            prev_parent_id = self.get_parent(child_id)
            if prev_parent_id == parent_id:
                # this is already the parent no changes need to be made
                continue
            if parent_2_children.get(prev_parent_id) is None:
                parent_2_children[prev_parent_id] = [child_id]
            else:
                parent_2_children[prev_parent_id].append(child_id)
        for prev_parent, children in parent_2_children.items():
            self.vertices[prev_parent]._remove_children(children)
            self.vertices[prev_parent].update()

    def _connect_to_new_parent(self, children_ids, parent_id):
        """This function should ONLY be called by set_parent
        """
        for child_id in children_ids:
            vertex = self.vertices.get(child_id)
            if vertex is None: # this means the child is a microstate
                self._microstate_parents[child_id] = parent_id
            else:
                vertex._set_parent(parent_id)
        self.vertices[parent_id]._add_children(children_ids)

    def update_split(self, partition, taus, split_vertex, parent):
        new_vertices = []
        for i, subset in enumerate(partition):
            vertex = HierarchicalMSMVertex(self, subset, parent, \
                                           taus[i], split_vertex.height, self.config)
            self.add_vertex(vertex)
            self._connect_to_new_parent(subset, vertex.id)
            new_vertices.append(vertex.id)


        if not self._is_root(split_vertex.id):
            # remove the vertex that split
            self.vertices[parent]._remove_children([split_vertex.id])
            del self.vertices[split_vertex.id]
        else:
            # the root is going up a level, so it's children will be the new vertices
            split_vertex._remove_children(split_vertex.children)

        self.vertices[parent]._add_children(new_vertices)
        self.update_vertex(parent, update_children=True)

    def _assign_children_to_parents(self, new_microstates_2_parents):
        for children, new_parent in new_microstates_2_parents:
            for child in children:
                self._microstate_parents[child] = new_parent
            self.vertices[new_parent]._add_children(children)

        # we want to update (transition probabilities) only after all microstates are assigned
        for _, new_parent in new_microstates_2_parents:
            self.update_vertex(new_parent)



    def set_parent(self, parent_id, children_ids):
        if not self._assert_valid_vertex(parent_id):
            raise ValueError("Got parent_id for non-existing id")

        if not isinstance(children_ids, Iterable):
            children_ids = [children_ids]

        self._disconnect_from_parents(children_ids, parent_id)
        self._connect_to_new_parent(children_ids, parent_id)
        self.vertices[parent_id].update(update_children=True)

    def update_vertex(self, vertex_id, update_children=False, update_parent=True):
        """
        Trigger a vertex to update its state (transition matrix, timescale, etc).

        Parameters
        ----------
        vertex_id : int
            id of the vertex to update
        """
        if self._is_microstate(vertex_id):
            return
        vertex = self.vertices[vertex_id]
        vertex.update(update_children, update_parent)

    def add_vertex(self, vertex):
        """
        Add a vertex to the tree

        Parameters
        ----------
        vertex : HierarchicalMSMVertex
            the new vertex
        """
        assert isinstance(vertex, HierarchicalMSMVertex)
        assert vertex.tree is self
        self.vertices[vertex.id] = vertex

    def sample_random_walk(self, sample_length, start=None):
        """
        Sample a discrete trajectory of microstates.

        Parameters
        ----------
        sample_length : int
            length of the random walk
        start : ind
            the id of the microstate to start from. If None (default), chooses one uniformly.

        Returns
        -------
        sample : np.ndarray
            an array of size sample_length of microstate ids

        """
        sample = np.ndarray(sample_length)
        if start is None:
            start = np.random.choice(list(self._microstate_MMSE.keys()))
        current = start
        for i in range(sample_length):
            current = self._random_step(current)
            sample[i] = current
        return sample

    def _random_step(self, microstate_id):
        """_random_step.
        A single step in a random walk

        Parameters
        ----------
        microstate_id :
            microstate_id
        """
        next_states, transition_probabilities = self._microstate_MMSE[microstate_id].T
        return np.random.choice(next_states, p=transition_probabilities)

    def sample_microstate(self, vertex_id=None):
        """Get a microstate from this HMSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        if vertex_id is None:
            vertex_id = self.root
        return self.vertices[vertex_id].sample_vertex()

class HierarchicalMSMVertex:
    """HierarchicalMSMVertex.
    This class represents a single vertex in an HMSM tree.
    A vertex has the ids of its children and its parent, as well as having a transition matrix
    describing transition probabilities between its children. This transition matrix is updated
    when changes to the tree structure affecting the vertex are made, or when the transition
    probabilities change due to new data.
    """


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
        T_inner = linalg.normalize_rows(self.T[:n, :n], norm=1)
        self._timescale = linalg.get_longest_timescale(T_inner, self.tau)

    @property
    def is_root(self):
        return self.parent == self.id

    @property
    def parent_update_threshold(self):
        return self.config["parent_update_threshold"]

    def _set_parent(self, parent):
        self.parent = parent

    def _add_children(self, children_ids):
        """
        Add children to this tree.
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children.update(children_ids)
        self._T_is_updated = False

    def _remove_children(self, children_ids):
        """
        NOTE: this should *only* be called by HierarchicalMSMTree as this does NOT do anything to
        maintain a valid tree structure - only changes local variables of this vertex
        """
        self._children -= set(children_ids)

    def update(self, update_children=False, update_parent=True):
        if update_children:
            for child in self._children:
                self.tree.update_vertex(child, update_parent=False)
        self._update_T()
        if self._check_parent_update_condition() and update_parent:
            self.tree.update_vertex(self.parent)


    def _check_parent_update_condition(self):
        if self.is_root:
            return False
        if self._last_update_sent is None:
            return True
        max_change_factor = max_fractional_difference(self._last_update_sent, \
                                                      self.get_external_T())
        return max_change_factor >= self.parent_update_threshold

    def _update_T(self):
        T_rows = []
        T_ids = []
        if len(self._children) == 0:
            self._T = self._T_tau = self._timescale = None
            self._T_is_updated = True
            return
        column_ids = set(self.children)

        # Get the rows of T corresponding to each child
        for child in self._children:
            ids, row = self.tree.get_external_T(child)
            T_rows.append(row)
            T_ids.append(ids)
            column_ids = column_ids.union(ids)
        id_2_index, index_2_id, full_n = self._get_id_2_index_map(column_ids)
        self.index_2_id = index_2_id
        self._T = np.zeros((full_n, full_n))

        # Now fill the matrix self._T
        for i, T_row in enumerate(T_rows):
            for _j, i_2_j_probability in enumerate(T_row):
                j = id_2_index[ T_ids[i][_j] ]
                self._T[i,j] += i_2_j_probability
        self._T_is_updated = True #Set this to false to always recalculate T

        #make the external vertices sinks
        n_external = full_n - self.n
        self._T[self.n:, self.n:] = np.eye(n_external)
        # get the transition matrix in timestep resolution self.tau
        self._T_tau = np.linalg.matrix_power(self._T, self.tau)
        self._update_timescale()
        self._update_external_T()

        if self._check_split_condition():
            self.split()

    def _get_id_2_index_map(self, column_ids):
        """Return a dict that maps each id to the index of its row/column in T, and the
        dimension of the full transition matrix.
        """
        id_2_parent_id = {}
        id_2_index = {}
        index_2_id = {}
        not_my_children = set()
        ids_to_add = set()
        for column_id in column_ids:
            parent = self.tree.get_parent(column_id)
            if parent != self.id:
                not_my_children.add(column_id)
                id_2_parent_id[column_id] = parent
                if parent not in column_ids:
                    ids_to_add.add(parent)
        column_ids -= not_my_children
        column_ids |= ids_to_add
        for j, id in enumerate(column_ids):
            id_2_index[id] = j
            index_2_id[j] = id
        for id, parent_id in id_2_parent_id.items():
            id_2_index[id] = id_2_index[parent_id]
        full_n = len(column_ids)
        return id_2_index, index_2_id, full_n



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
        assert self._T_is_updated
        T = self._T # this is T(1) as opposed to self.T which is T(tau)
        n = self.n
        local_stationary = self.get_local_stationary_distribution()
        local_stationary = np.resize(local_stationary, T.shape[0]) # pad with 0's.

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

        #NOTE: the assumption underlying the whole concept, is that full_transition_probabilities[n:]
        # is similar to T_ext_tau[i, n:] for all i<n. In other words, that a with high probability,
        # a random walk mixes before leaving this MSM.
        # This could be used as validation, and maybe as a clustering criterion.


    def _check_split_condition(self):
        return self.config["split_condition"](self)

    def _get_partition(self):
        # return (partition, taus) where partition is an iterable of iterables of children_ids
        return self.config["split_method"](self)

    def split(self):
        # 1. get partition compatible with max_timescale
        # 2. create new vertices according to this partition
        # 3. update tree structure
        index_partition, taus = self._get_partition()
        if len(index_partition)==1:
            warnings.warn(f"split was called, but {self.config['split_method']} was unable to \
                            find a partition", RuntimeWarning)
            return
        id_partition = [ [self.index_2_id[index] for index in subset] \
                            for subset in index_partition]

        self.tree.update_split(id_partition, taus, self, self.parent)

    def sample_microstate(self):
        """Get a microstate from this MSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        sample = self.config["sample_method"](self)
        if self.height == 1:
            return sample

        return self.tree.sample_microstate(sample)

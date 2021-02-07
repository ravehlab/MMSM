from collections.abc import Iterable
from collections import defaultdict
import numpy as np
from HMSM.util import util
from HMSM.prototype.hierarchical_msm_vertex import HierarchicalMSMVertex

class HierarchicalMSMTree:
    """HierarchicalMSMTree.
    This class encapsulates the HMSM data structure, and provides an interface for it.
    The vertices of the tree are HierarchicalMSMVertex objects, which can be accessed from this
    class.
    Most importantly, the method "update_model_from_trajectories" allows the user to simply input
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

    def get_longest_timescale(self, dt=1):
        return self.vertices[self.root].timescale * dt
    def update_model_from_trajectories(self, dtrajs, update_MMSE=True):
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
        parents_2_new_microstates = defaultdict(set)
        for vertex_id in updated_microstates:
            if self._microstate_parents.get(vertex_id) is None:
                orphans, parent = self._get_new_microstate_parent(vertex_id)
                parents_2_new_microstates[parent] |= orphans
        self._assign_children_to_parents(parents_2_new_microstates)

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

        max_change_factor = util.max_fractional_difference(self._last_update_sent[microstate], \
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

    def _assign_children_to_parents(self, parents_2_new_microstates):
        for new_parent, children in parents_2_new_microstates.items():
            for child in children:
                self._microstate_parents[child] = new_parent
            self.vertices[new_parent]._add_children(children)

        # we want to update (transition probabilities) only after all microstates are assigned
        for new_parent in parents_2_new_microstates.keys():
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

    def sample_microstate(self, n_samples, vertex_id=None):
        """Get a microstate from this HMSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        if vertex_id is None:
            vertex_id = self.root
        return self.vertices[vertex_id].sample_vertex(n_samples)


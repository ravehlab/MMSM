from collections import defaultdict
import numpy as np
from HMSM.util import util, UniquePriorityQueue
from HMSM.prototype.hierarchical_msm_vertex import HierarchicalMSMVertex


class HierarchicalMSMTree:
    """HierarchicalMSMTree.
    This class encapsulates the HMSM data structure, and provides an interface for it.
    The vertices of the tree are HierarchicalMSMVertex objects, which can be accessed from this
    class.
    Most importantly, the method update_model_from_trajectories() allows the user to simply input
    observed discrete trajectories, (in the form of an iterable of state ids), and the estimation
    of a Hierarchical Markov Model is entirely automated - including estimating a hierarchy of
    metastable states and transition rates between them in different timescales.

    # TODO: add a short usage example
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
        self.config = config
        self._update_queue = UniquePriorityQueue()

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
        self._add_vertex(root)
        self.root = root.id

    @property
    def height(self):
        """Length of the longest path from the root of the tree to another vertex.
        """
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
        """
        Get the timescale associated with the slowest process represented by the HMSM

        Parameters
        ----------
        dt : float
            The timestep size of the simulation
        """
        return self.vertices[self.root].timescale * dt * self.config["base_tau"]

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
        parents_2_new_microstates = defaultdict(set)
        for dtraj in dtrajs:
            updated_microstates.update(dtraj)
            src = dtraj[0]
            #TODO clean this up
            if self._microstate_parents.get(src) is None:
                assert self.height == 1
                self._microstate_parents[src] = self.root
                parents_2_new_microstates[self.root].add(src)

            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                # count the observed transition, and a pseudocount for the reverse transition
                self._microstate_counts[src][dst] += 1
                if self._microstate_counts[dst][src] == 0:
                    self._microstate_counts[dst][src] = self.alpha

                # assign newly discovered microstates to the MSM they were discovered from
                if self._microstate_parents.get(dst) is None:
                    if self.height == 1:
                        parent = self.root
                    else:
                        parent = self._microstate_parents[src]
                    self._microstate_parents[dst] = parent
                    parents_2_new_microstates[parent].add(dst)
                src = dst
        # add newly discovered microstates to their parents
        for parent, children in parents_2_new_microstates.items():
            self.vertices[parent].add_children(children)

        if update_MMSE:
            for vertex_id in updated_microstates:
                self._microstate_MMSE[vertex_id] = self._dirichlet_MMSE(vertex_id)

        # update from the leaves upwards
        for vertex_id in updated_microstates:
            if self._check_parent_update_condition(vertex_id):
                parent_id = self._microstate_parents[vertex_id]
                self._update_vertex(parent_id)

        self._do_all_updates_by_height()

    def _dirichlet_MMSE(self, vertex_id):
        """
        Get the equivalent of external_T, but for a microstate.
        """
        MMSE_ids = np.array(list(self._microstate_counts[vertex_id].keys()), dtype=int)
        MMSE_counts = np.array(list(self._microstate_counts[vertex_id].values())) + self.alpha
        MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
        return (MMSE_ids, MMSE_counts)

    def _check_parent_update_condition(self, microstate):
        if self._last_update_sent.get(microstate) is None:
            return True
        if set(self._last_update_sent[microstate][0])!=set(self._microstate_MMSE[microstate][0]):
            return True

        max_change_factor = util.max_fractional_difference(self._last_update_sent[microstate], \
                                                      self._microstate_MMSE[microstate])
        return max_change_factor >= self.config["parent_update_threshold"]


    def _connect_to_new_parent(self, children_ids, parent_id):
        for child_id in children_ids:
            vertex = self.vertices.get(child_id)
            if vertex is None: # this means the child is a microstate
                self._microstate_parents[child_id] = parent_id
            else:
                vertex.set_parent(parent_id)
        self.vertices[parent_id].add_children(children_ids)

    def _update_split(self, partition, taus, split_vertex, parent):
        if len(partition)==1: # this is a trivial partition
            return
        new_vertices = []
        for i, subset in enumerate(partition):
            vertex = HierarchicalMSMVertex(self, subset, parent, \
                                           taus[i], split_vertex.height, self.config)
            self._add_vertex(vertex)
            self._connect_to_new_parent(subset, vertex.id)
            new_vertices.append(vertex.id)
            self._update_vertex(vertex.id)



        if not split_vertex.is_root:
            self.vertices[parent].add_children(new_vertices)
            self._remove_vertex(split_vertex.id)
        else:
            # the root is going up a level, so it's children will be the new vertices
            split_vertex.remove_children(split_vertex.children)
            self.vertices[parent].add_children(new_vertices)
            split_vertex.height += 1
            split_vertex.tau *= 2 #TODO: this should be dependent on the partition
        self._update_vertex(parent)


    def _move_children(self, previous_parent_id, parent_2_children):
        previous_parent = self.vertices[previous_parent_id]
        # move all the children:
        for new_parent, children in parent_2_children.items():
            previous_parent.remove_children(children)
            self.vertices[new_parent].add_children(children)
            if previous_parent.height==1: # the children are microstates
                for child in children:
                    self._microstate_parents[child] = new_parent
            else:
                for child in children:
                    self.vertices[child].set_parent(new_parent)
            self._update_vertex(new_parent)

        if previous_parent.n == 0: # this vertex is now empty, we want to delete it
            self._remove_vertex(previous_parent_id)
        else:
            self._update_vertex(previous_parent_id)


    def _update_vertex(self, vertex_id):
        if self._is_microstate(vertex_id):
            return
        height = self.vertices[vertex_id].height
        result = self._update_queue.put(item=(height, vertex_id))

    def _do_all_updates_by_height(self):
        """
        Update the tree, from the bottom (highest resolution MSMs) upwards.
        Vertices needing updates will be in self._update_queue prioritized by their height,
        if updating a vertex triggers the need to update another vertex (by a vertex split, moving
        children to another vertex, or a change in transition rates that neccesitates the parent
        being updated), the other vertex will be added to the queue.
        """
        while self._update_queue.not_empty():
            vertex_id = self._update_queue.get()
            vertex = self.vertices.get(vertex_id)
            if vertex is None:
                continue # this vertex no longer exists
            result, update = vertex.update()
            if result==HierarchicalMSMVertex.SPLIT:
                partition, taus, split_vertex, parent = update
                self._update_split(partition, taus, split_vertex, parent)
            elif result==HierarchicalMSMVertex.DISOWN_CHILDREN:
                self._move_children(vertex_id, update)
            elif result==HierarchicalMSMVertex.UPDATE_PARENT:
                parent = self.get_parent(vertex_id)
                self._update_vertex(parent)
            else:
                assert result==HierarchicalMSMVertex.SUCCESS, f"Got unknown update result {result} \
                                                                from vertex {vertex_id}"


    def _add_vertex(self, vertex):
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

    def _remove_vertex(self, vertex):
        parent = self.get_parent(vertex)
        del self.vertices[vertex]
        self.vertices[parent].remove_children([vertex])
        # update the parent and neighbors of the removed vertex
        for other_vertex in self.vertices.values(): #TODO find less wasteful way to do this
            if hasattr(other_vertex, '_external_T'):
                if vertex in other_vertex._external_T[0]:
                    self._update_vertex(other_vertex.id)
        
        # If the parent is now empty, remove it too.
        if self.vertices[parent].n==0:
            self._remove_vertex(parent)
        else:
            self._update_vertex(parent)

        assert vertex not in self._microstate_parents.values()
        for v in self.vertices.values():
            assert v.parent is not vertex

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
        next_states, transition_probabilities = self._microstate_MMSE[microstate_id]
        return np.random.choice(next_states, p=transition_probabilities)

    def sample_microstate(self, n_samples, vertex_id=None):
        """Get a microstate from this HMSM, ideally chosen such that sampling a random walk from
        this microstate is expected to increase some objective function.

        return: microstate_id, the id of the sampled microstate.
        """
        if vertex_id is None:
            vertex_id = self.root
        return self.vertices[vertex_id].sample_microstate(n_samples)


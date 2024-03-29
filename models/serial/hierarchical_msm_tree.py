"""HierarchicalMSMTree"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from collections import defaultdict
import numpy as np
from HMSM.ext.stationary_vector import stationary_distribution #TODO include this function in package
from HMSM import estimators, HMSMConfig, samplers
from HMSM.util import util, UniquePriorityQueue, linalg
from HMSM.models import BaseHierarchicalMSMTree
from . import HierarchicalMSMVertex
from .util import max_fractional_difference
import msmtools

class HierarchicalMSMTree(BaseHierarchicalMSMTree):
    """HierarchicalMSMTree.
    This class encapsulates the HMSM data structure, and provides an interface for it.
    The vertices of the tree are HierarchicalMSMVertex objects, which can be accessed from this
    class.
    Most importantly, the method update_model_from_trajectories() allows the user to simply input
    observed discrete trajectories (in the form of an iterable of state ids), and the estimation
    of a Hierarchical Markov Model is entirely automated - including estimating a hierarchy of
    metastable states and transition rates between them in different timescales.

    Parameters
    ----------
    config : dict
        A dictionary of model parameters.
        # TODO: define a stadard for these parameters, and also - these are probably not model parameters but parameters of this class/inner algorithms/etc.

    Attributes
    ----------
    height : int
        The height of the tree
    vertices : dict
        A dictionary mapping the vertex ids of all nodes in the tree to the correspoding
        HierarchicalMSMVertex objects
    alpha : int or float
        The parameter of the Dirichlet distribution used as a prior for transition probabilities
        between microstates.
    root : int
        The id of the root vertex
    """


    def __init__(self, config:HMSMConfig):
        super().__init__(config)
        self._partition_estimator = estimators.metastable_partition.get_metastable_partition(config)
        self._vertex_sampler = samplers.vertex_samplers.get_vertex_sampler(config)
        self._microstate_parents = dict()
        self._microstate_counts = util.count_dict(depth=2)
        self._microstate_transitions = dict()
        self._last_update_sent = dict()
        self._update_queue = UniquePriorityQueue()
        self._levels = defaultdict(list)
        self.vertices = dict()
        self._init_root()
        self._cache = defaultdict(dict) # we can use this to cache results of functions that get called often
        self._updated_height = -1
        self._oom_reweighted = config.oom
        self._reversible = config.reversible
        if self._oom_reweighted:
            # self._2_step_transitions[i][(j,k)] is the number of j->k->i transitions observed
            # The ordering is backwards so that it can be efficiently converted to a csc matrix
            self._2_step_transitions = util.count_dict(depth=2)

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
        heuristics = ["uniform", "exploration"]
        weights = [0.5, 0.5]
        sampler = samplers.vertex_samplers.get_vertex_sampler(sampling_heuristics=heuristics, 
                                                              weights=weights)
        root = HierarchicalMSMVertex(self, children=set(),\
                                           parent=None,\
                                           tau=1,
                                           height=1,
                                           partition_estimator=self._partition_estimator,
                                           vertex_sampler=sampler,
                                           config=self.config)
        self._add_vertex(root)
        self._root = root.id

    @property
    def alpha(self):
        return self.config.alpha

    @property
    def root(self):
        return self._root

    @property
    def height(self):
        """Length of the longest path from the root of the tree to another vertex.
        """
        return self.vertices[self.root].height

    def get_level(self, level):
        assert level <= self.height
        if level==0:
            return list(self._microstate_parents.keys())
        return self._levels[level].copy()

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
        """get_external_T.
        Get the transition probabilities between a vertex and other vertices on the same level
        of the tree.

        Parameters
        ----------
        vertex_id : int
            the id of the vertex
        tau : int
            the time-resolution of the transitions
        Returns
        -------
        ids : ndarray
            An array of the ids of the neighbors of this vertex
        transition_probabilities : ndarray
            An array of the transition probabilities to the neighbors of this vertex, such that
            transition_probabilities[i] is the probability of getting to ids[i] in tau steps from
            the vertex with id vertex_id.
        """
        if self._is_microstate(vertex_id):
            assert tau==1
            ids, ext_T = self._microstate_transitions[vertex_id]
            self._last_update_sent[vertex_id] = [ids, ext_T]
            return ids, ext_T
        return self.vertices[vertex_id].get_external_T(tau)

    def get_n_samples(self, vertex):
        """get_n_samples.
        Get the total number of times this vertex has appeared in trajectories used to estimate
        this HMSM.

        Parameters
        ----------
        vertex : int
            The id of the vertex

        Returns
        -------
        n_samples : int
            If the vertex is a set of vertices, this is the sum of this value over all the vertices
            in the set, otherwise this is simply the number of appearances of this vertex in
            trajectories passed to the method self.update_model_from_trajectories.
        """
        if self._is_microstate(vertex):
            return max(1, sum(self._microstate_counts[vertex].values()))
        return self.vertices[vertex].n_samples


    def get_microstates(self, vertex=None):
        """get_microstates.
        Get the ids of all the microstates in this tree.

        Parameters
        ----------
        vertex : int, optional
            If supplied, returns only the microstates under this vertex.

        Returns
        -------
        microstates : set
            A set of all the microstates in the tree, or under a given vertex if the argument
            vertex is given.
        """
        if vertex is None:
            vertex = self.root
        return self.vertices[vertex].get_all_microstates()

    def get_longest_timescale(self):
        """get_longest_timescale.
        Get the timescale associated with the slowest process represented by the HMSM.
        """
        return self.vertices[self.root].timescale

    def update_model_from_trajectories(self, dtrajs):
        """update_model_from_trajectories.
        Update the Hierarchical MSM with data from observed discrete trajectories.

        Parameters
        ----------
        dtrajs : iterable
            an iterable of trahjectories, where each trajectory is an iterable of state ids
        update_transitions : bool
            if True, update the estimated transition probabilities of each vertex that was observed
        """

        updated_microstates, parents_2_new_microstates = self._count_transitions(dtrajs)
        if self._oom_reweighted:
            self._count_2_step_transitions(dtrajs)
        # add newly discovered microstates to their parents
        for parent, children in parents_2_new_microstates.items():
            self.vertices[parent].add_children(children)

        #update transition probabilities from observed transitions
        if self.config.transition_estimator == 'Dirichlet_MMSE':
            for vertex_id in updated_microstates:
                self._microstate_transitions[vertex_id] = estimators.transitions.dirichlet_MMSE(\
                                                  self._microstate_counts[vertex_id], self.alpha)
        else:
            raise ValueError("Only Dirchlet_MMSE transition estimator is currently implemented")

        # update the vertices of the tree from the leaves upwards
        for vertex_id in updated_microstates:
            if self._check_parent_update_condition(vertex_id):
                parent_id = self._microstate_parents[vertex_id]
                self._update_vertex(parent_id)
        self._do_all_updates_by_height()

    def force_update_all(self):
        """force_update_all.
        Update all vertices in the tree, regardless of the last time they were updated.
        """
        for vertex in self.vertices:
            self._update_vertex(vertex)
        self._do_all_updates_by_height()

    def force_rebuild_tree(self):
        """force_rebuild_tree.
        Remove all existing vertices, and rebuild entire tree, keeping only microstates and all
        observed transitions between them.
        """
        del self._levels
        del self.vertices
        self._levels = defaultdict(list)
        self.vertices = dict()
        self._init_root()
        self._connect_to_new_parent(self._microstate_parents.keys(), self.root)
        self.force_update_all()

    def sample_full_T(self, nsamples):
        n = len(self._microstate_parents)
        T = np.zeros((nsamples,n,n))
        level = self.get_level(0)
        for i, src in enumerate(level):
            ids, row = self._microstate_transitions[src]

            neighbor_ids = self._microstate_counts[src].keys()
            neighbor_indices = [level.index(id) for id in neighbor_ids]
            counts = np.fromiter(self._microstate_counts[src].values(), dtype=float)
            T[:,i,neighbor_indices] = np.random.dirichlet(counts + self.alpha, nsamples)
        return T, level

    def get_full_T(self):
        """get_full_T.
        Get the full transition matrix between all microstates.

        Returns
        -------
        T : np.ndarray of shape (n,n), where n is the number of microstates in the tree
            Transition probability matrix, in timestep of config.base_tau
        id_2_index : List
            List mapping indices in T to their corresponding microstate ids
        """
        level = self.get_level(0)
        if self._oom_reweighted:
            OOM_reweighted_T = linalg.get_OOM_reweighted_transition_matrix(self._microstate_counts,
                                                                           self._2_step_transitions,
                                                                           level)
            return OOM_reweighted_T, level
        elif self._reversible:
            count_matrix = util.sparse_matrix_from_count_dict(self._microstate_counts, level)
            T = msmtools.estimation.transition_matrix(count_matrix, reversible=True) 
        n = len(level)
        T = np.zeros((n,n))
        for i, src in enumerate(level):
            ids, row = self._microstate_transitions[src]
            for id, transition_probability in zip(ids, row):
                assert transition_probability >= 0
                j = level.index(id)
                T[i,j] = transition_probability

        return T, level

    def get_full_stationary_distribution(self, return_indices=False):
        """get_full_stationary_distribution.
        Get the stationary distribution over all microstates.

        Returns
        -------
        pi : dict
            A dictionary mapping microstate ids to their stationary distribution.
        """
        T, level = self.get_full_T()
        st = stationary_distribution(T)
        pi = {}
        for index, id in enumerate(level):
            pi[id] = st[index]
        return pi

    def _count_2_step_transitions(self, dtrajs):
        for dtraj in dtrajs:
            if len(dtraj)<3:
                continue
            for i in range(2, len(dtraj)):
                self._2_step_transitions[dtraj[i]][(dtraj[i-2], dtraj[i-1])] += 1

    def _count_transitions(self, dtrajs):
        updated_microstates = set()
        parents_2_new_microstates = defaultdict(set)
        for dtraj in dtrajs:
            if len(dtraj)==0:
                continue
            updated_microstates.update(dtraj)
            src = dtraj[0]

            # check if the first state is new, this should only happen at initialization, where
            # the microstate level is the top level MSM, sot he parent is always the root.
            if not self._is_microstate(src): #TODO clean this bit up
                assert self.height==1
                self._microstate_parents[src] = self.root
                _=self._microstate_counts[src][src]
                parents_2_new_microstates[self.root].add(src)

            for i in range(1, len(dtraj)):
                dst = dtraj[i]
                # count the observed transition
                self._microstate_counts[src][dst] += 1
                # evaluate the reverse transition, so that it will be set to 0 if none have been
                # observed yet. This is so that the prior weight of this transition will be alpha
                _=self._microstate_counts[dst][src]

                # assign newly discovered microstates to the MSM they were discovered from
                if self._microstate_parents.get(dst) is None:
                    parent = self._microstate_parents[src]
                    self._microstate_parents[dst] = parent
                    parents_2_new_microstates[parent].add(dst)
                src = dst
        return updated_microstates, parents_2_new_microstates


    def _check_parent_update_condition(self, microstate):
        """Check if a microstates transition probabilities have changed enough to trigger
        its parent to update
        #TODO use config.parent_update_condition
        """
        # check if no updates have been made yet
        if self._last_update_sent.get(microstate) is None:
            return True
        # check if new transitions have been observed since last update
        if set(self._last_update_sent[microstate][0])!=set(self._microstate_transitions[microstate][0]):
            return True

        max_change_factor = max_fractional_difference(self._last_update_sent[microstate], \
                                                           self._microstate_transitions[microstate])
        return max_change_factor >= self.config.parent_update_threshold


    def _connect_to_new_parent(self, children_ids, parent_id):
        """
        Set the parent of all children_ids to be parent_id, and add them to the parents
        children
        """
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
                                           taus[i], split_vertex.height, \
                                           partition_estimator=self._partition_estimator,\
                                           vertex_sampler=self._vertex_sampler,\
                                           config=self.config)
            self._add_vertex(vertex)
            self._connect_to_new_parent(subset, vertex.id)
            new_vertices.append(vertex.id)
            self._update_vertex(vertex.id)

        if not split_vertex.is_root:
            self.vertices[parent].add_children(new_vertices)
            self._remove_vertex(split_vertex.id)
        else:
            # the root is going up a level, so its children will be the new vertices
            self._levels[self.height].remove(self.root)
            split_vertex.remove_children(split_vertex.children)
            self.vertices[parent].add_children(new_vertices)
            split_vertex.height += 1
            split_vertex.tau *= 16 #TODO: this should be dependent on the partition
            self._add_vertex(split_vertex)
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
        self._update_queue.put(item=(height, vertex_id))

    def _do_all_updates_by_height(self):
        """
        Update the tree, from the bottom (highest resolution MSMs) upwards.
        Vertices needing updates will be in self._update_queue prioritized by their height,
        if updating a vertex triggers the need to update another vertex (by a vertex split, moving
        children to another vertex, or a change in transition rates that neccesitates the parent
        being updated), the other vertex will be added to the queue.
        """
        self._cache.clear()
        while self._update_queue.not_empty():
            vertex_id = self._update_queue.get()
            vertex = self.vertices.get(vertex_id)
            if vertex is None:
                continue # this vertex no longer exists
            self._updated_height = vertex.height - 1
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
        self._levels[vertex.height].append(vertex.id)

    def _remove_vertex(self, vertex):
        parent = self.get_parent(vertex)
        height = self.vertices[vertex].height
        self._levels[height].remove(vertex)
        del self.vertices[vertex]
        self.vertices[parent].remove_children([vertex])
        # update the parent and neighbors of the removed vertex
        for other_vertex_id in self.get_level(height):
            other_vertex = self.vertices[other_vertex_id]
            if hasattr(other_vertex, '_external_T'): #TODO use neighbor
                if vertex in other_vertex._external_T[0]:
                    self._update_vertex(other_vertex.id)
            if vertex in other_vertex.neighbors:
                self._update_vertex(other_vertex.id)

        # If the parent is now empty, remove it too.
        if self.vertices[parent].n==0:
            self._remove_vertex(parent)
        else:
            self._update_vertex(parent)

        assert vertex not in self._microstate_parents.values()
        for v in self.vertices.values():
            assert v.parent is not vertex

    def get_ancestor(self, vertex, level):
        """get_ancestor.
        Get the direct ancestor of a vertex in the tree, on a specified level.

        Parameters
        ----------
        vertex :
            vertex
        level :
            level

        Returns
        -------
        parent : int
            The ancestor of vertex with height level in the tree.
        """
        parent = vertex
        while self.vertices[parent].height < level:
            parent = self.vertices[parent].parent
        return parent

    def get_level_T(self, level, tau, parent_order=None, return_order=False):
        """get_level_T.
        Get the transition matrix between vertices on a single level, in a given lag time.

        Parameters
        ----------
        level : int
            level
        tau : int
            tau 

        Returns
        -------
        T : ndarray
            Transition matrix.
        level : list of int
            A list of vertex ids, such that T[i,j] is the transition probability from level[i] to 
            level[j] in tau time.
            In other words, the i'th row and column of T correspond to the vertex with id level[i].
        """
        #FIXME use of tau here is misleading, as it is in units of base_tau.
        cache_key = f"{level, tau, parent_order, return_order})"
        if self._updated_height <= level:

            cached_result = self._cache["get_level_T"].get(cache_key)
            if cached_result is not None: 
                return cached_result

        level = self.get_level(level)

        if parent_order is not None: # sort the level by a given ordering over their parents
            level_by_parents = []
            for parent in parent_order:
                for child in level:
                    if self.get_parent(child) == parent:
                        level_by_parents.append(child)
            level = level_by_parents

        n = len(level)
        index_2_id = dict(zip(range(n), level))
        id_2_index = dict(zip(level, range(n)))
        T = np.zeros((n,n))
        for vertex in level:
            i = id_2_index[vertex]
            ids, transition_probabilities = self.get_external_T(vertex)
            for neighbor, transition_probability in zip(ids, transition_probabilities):
                j = id_2_index[neighbor]
                T[i,j] = transition_probability
        linalg._assert_stochastic(T)
        if return_order:
            retval = np.linalg.matrix_power(T, tau), level
        else:
            retval = np.linalg.matrix_power(T, tau), level
        self._cache["get_level_T"][cache_key] = retval
        return retval

    def sample_from_stationary(self, vertex, level=0):
        """sample_from_stationary.
        Sample on of the descendents of vertex, from the stationary distribution of the vertex.

        Parameters
        ----------
        vertex : int
           id of the vertex from which a sample will be taken
        level : int
            the level of the tree from which to sample

        Returns
        -------
        sample : int
            the id of the sampled vertex
        """
        sample = vertex
        for _ in range(self.vertices[sample].height - level):
            sample = self.vertices[sample].sample_from_stationary()
        return sample


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
            start = np.random.choice(list(self._microstate_transitions.keys()))
        current = start
        for i in range(sample_length):
            current = self._random_step(current)
            sample[i] = current
        return sample

        #TODO Give this a more informative name, maybe sample_random_walk can do the same thing.
    def sample_trajectory(self, start, length, n, level):
        """sample_trajectory.
        Sample n trajectories of vertices, of length length, on level level, starting from a vertex
        chosen from the stationary distribution of the vertex start
        #TODO better documentation

        Parameters
        ----------
        self :
            self
        start :
            start
        length :
            length
        n :
            n
        level :
            level
        """
        #TODO use T from get_level_T
        trajs = []
        for _ in range(n):
            step = self.sample_from_stationary(start, level)
            traj = np.ndarray(length, dtype=int)
            for i in range(length):
                step = self._random_step(step)
                traj[i] = step
            trajs.append(traj)
        return trajs

    def _random_step(self, vertex_id):
        next_states, transition_probabilities = self.get_external_T(vertex_id)
        return np.random.choice(next_states, p=transition_probabilities)

    def sample_states(self, n_samples, vertex_id=None):
        """sample_states.
        """
        #TODO match base class signature
        if vertex_id is None:
            vertex_id = self.root
        return self.vertices[vertex_id].sample_microstate(n_samples)

from util import normalize_rows

class Heirarchical_MSM():

    def __init__():
        self._tau = None
        self._timescale = None
        self._V = None
        self._n = 0
        self.scale_difference_factor = 0.25
        self._count_matrix
        self.course_grain
        self.microstates
        self.microstate_indices
        self.n_samples
        self._n_states
        self.parent
        self._ex_index #this MSMs index in the parent MSM

        # _ex_count_matrix[i,j] is the number of transitions from the vertex i to the parent MSMs
        # vertex j
        self._ex_count_matrix 

    @property
    def tau(self):
        return self._tau

    @property
    def timescale(self):
        return self._timescale

    @property
    def V(self):
        return self._V

    @property
    def n(self):
        #TODO does this include 'outside vertices'? for now no
        return self._n_states

    @property
    def T(self):
        #TODO level1 and level>1 handle this differently
        if self._T_is_updated:
            return self._T
        else:
            return self.update_T

    def update_T(self):
        #TODO use bayesian estimation for this:
        self._T = normalize_rows(self._count_matrix)
        self._T_is_updated = True
        return self._T

    def update_count_matrix(self, dtrajs):

        counts = np.zeros((self._n_states, self._n_states))
        for dtraj in dtrajs:
            src = dtraj[0]
            for i in range(1,len(dtraj)):
                dst = dtraj[i]
                if is_external_vertex(dst):
                    dst = external_to_local(dst)
                    self._ex_count_matrix[src, dst] += 1
                    break # we aren't interested in what happens after leaving this MSM
                else:
                    counts[src, dst] += 1
        self._count_matrix += counts

    def _global_to_local_index(self, index):
        if index in self.global2local:
            return self.global2local[index]
        else:
            assert(self.parent is not None)
            ex_index = self.parent.get_parent_vertex(index)
            return ex_index


    def add_states(self, new_states):
        """
        Gets a list of (global) indices of new states to add to this MSM.
        Updates the variables count_matrix, _ex_count_matrix, _n_states, and global2local
        accordingly
        """
        n_new_states = len(new_states)
        for state, inx in enumerate(new_states):
            self.global2local[state] = self._n_states + inx

        # add rows
        self.count_matrix = np.vstack([self.count_matrix, \
                                       np.zeros((n_new_states, self._n_states))])
        # add columns
        self.count_matrix = np.hstack([self.count_matrix, \
                                       np.zeros((self._n_states+n_new_states, n_new_states))])

        # add rows to external count matrix
        self._ex_count_matrix = np.vstack([self._ex_count_matrix, \
                                           np.zeros((n_new_states, self._ex_count_matrix))])
        
        self._n_states += n_new_states


    def get_discrete_trajectories(self, trajs):
        """
        gets a list of trajectories, updates any new states found, and returns
        a list of discrete trajectories sampled at intervals of self.tau, indexed 
        according to this MSMs indexing
        """
        tau_trajs = [ [traj[i] for i in range(0, len(traj), self.tau)]
                      for traj in trajs]
        dtrajs = []
        for traj in tau_trajs:
            dtraj, new_states = self.course_grain.discretize(traj)
            if new_states:
                self.add_states(new_states)
            dtraj = list(map(self._global_to_local_index, dtraj))
            dtrajs.append(dtraj)
        return dtrajs


    def choose_vertices_for_sample(self, sampling_heuristic):
        """
        For now just uniform sampling
        """
        return util.sample_from_ndarray(self.microstate_indices, self.n_samples)


    def sample_from_microstates(self, V_sample):
        starting_points = []
        for v in V_sample:
            starting_points.append(self.microstates[v].get_representative())
        sample_len = 2*self.tau
        trajs = self.sampler.sample(starting_points, sample_len)
        return self.get_discrete_trajectories(trajs)

    def check_split_condition(self):
        pass
    
    def split_vertex(self, v):
        pass

    def split(self):
        pass

    def sample_update(self, sampling_heuristic='default'):

        V_sample = self.choose_vertices_for_sample(sampling_heuristic)
        if self.level=1:
            # If this is a level 1 MSM all vertices are microstates, so we sample directly from
            # the configuration space
            trajectories = self.sample_from_microstates(V_sample):
            self.update_count_matrix(trajectories)
            
        else:
            for v in V_sample:
                self._T[v], timescale = self.V[v].sample_update()
                if timescale > self.tau * self.scale_difference_factor
                    self.split_vertex(v)

        if self._has_parent:
            return self.T, self.timescale
        else:
            if self.check_split_condition():
                self.split()
            



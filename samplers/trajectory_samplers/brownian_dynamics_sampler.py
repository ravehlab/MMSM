"""BrownianDynamicsSampler"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import numpy as np
import numba
from HMSM.samplers.trajectory_samplers import BaseTrajectorySampler

__all__ = ["BrownianDynamicsSampler"]

class BrownianDynamicsSampler(BaseTrajectorySampler):
    #TODO: documentation

    def __init__(self, force, dim, dt, kT, start_points=None, cache_size=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force
        self.dim = dim
        self.dt = dt
        self.kT = kT
        self.noise_magnitude = np.sqrt(2*self.dim * self.kT * self.dt)
        self.start_points = start_points
        self._cache_size = cache_size
        self._noise_cache = self.noise_magnitude*np.random.normal(size=(cache_size, 2))
        self._cache_ptr = 0

    def get_initial_sample(self, sample_len, n_samples, tau):
        if self.start_points is None:
            self.start_points = np.random.normal(0, 0.1, size=(n_samples, self.dim))
        trajs = []
        for point in self.start_points:
            trajs.append(self.sample_from_point(point, sample_len, tau))
        return self._get_dtrajs(trajs)


    def sample_from_states(self, microstates, sample_len, n_samples, tau):
        dtrajs = []
        temp_traj = np.ndarray((sample_len, 2))
        for microstate in microstates:
            for _ in range(n_samples):
                x = self._discretizer.sample_from(microstate)
                assert self._discretizer._coarse_grain_states(np.array([x]))[0]==microstate
                self.njit_sample_from_point(x, sample_len, tau, self.force, self.dt, self.noise_magnitude, temp_traj)
                dtraj = self._get_dtraj(temp_traj)
                dtrajs.append(dtraj)
        return dtrajs

    def sample_from_point(self, point, sample_len, tau):
        x = point
        traj = [x]
        for _ in range(sample_len-1):
            for __ in range(tau):
                x = self._BD_step(x)
            traj.append(x)
        return np.array(traj)

    @staticmethod
    @numba.njit
    def njit_sample_from_point(point, sample_len, tau, force, dt, noise_magnitude, out):
        temp = point
        out[0] = temp
        for i in range(1, sample_len):
            for __ in range(tau):
                step = force(temp)*dt + np.random.normal(0, 1, 2)*noise_magnitude
                temp += step 
            out[i] = temp


    def _get_dtraj(self, traj):
        return self._discretizer.get_coarse_grained_states(traj)

    def _get_dtrajs(self, trajs):
        dtrajs = []
        for traj in trajs:
            dtrajs.append(self._discretizer.get_coarse_grained_states(traj))
        return dtrajs

    def _BD_step(self, x):
        # TODO add friction coefficient
        return x + self.force(x)*self.dt + self._brownian_noise()

    def _brownian_noise(self):
        if self._cache_ptr == self._cache_size:
            self._cache_ptr = 0
            self._noise_cache = self.noise_magnitude*np.random.normal(size=(self._cache_size, 2))
        self._cache_ptr += 1
        return self._noise_cache[self._cache_ptr-1]



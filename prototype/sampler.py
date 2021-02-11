import numpy as np
from HMSM.prototype.coarse_grain import KCentersCoarseGrain

__all__ = ["BrownianDynamicsSampler"]

class BrownianDynamicsSampler:

    def __init__(self, force, dim, dt, kT, coarse_grain_type=KCentersCoarseGrain, **coarse_grain_args):
        self.force = force
        self.dim = dim
        self.dt = dt
        self.kT = kT
        self.coarse_grain = coarse_grain_type(**coarse_grain_args)
        self.noise_magnitude = np.sqrt(2*self.dim * self.kT * self.dt)

    def get_initial_sample(self, start_points, sample_len, n_samples, tau):
        trajs = []
        for point in start_points:
            for i in range(n_samples):
                trajs.append(self.sample_from_point(point, sample_len, tau))
        return self._get_dtrajs(trajs)

    def _sample_from_microstates_depracated(self, microstates, sample_len, n_samples, tau):
        trajs = []
        for microstate in microstates:
            for _ in range(n_samples):
                x = self.coarse_grain.sample_from(microstate)
                traj = self.sample_from_point(x, sample_len, tau)
                trajs.append(traj)
        dtrajs = self._get_dtrajs(trajs)

    def sample_from_microstates(self, microstates, sample_len, n_samples, tau):
        dtrajs = []
        for microstate in microstates:
            for _ in range(n_samples):
                x = self.coarse_grain.sample_from(microstate)
                traj = self.sample_from_point(x, sample_len, tau)
                dtraj = [microstate] + self._get_dtraj(traj)
                dtrajs.append(dtraj)
        return dtrajs

    def sample_from_point(self, point, sample_len, tau):
        x = point.copy()
        traj = []
        for j in range(sample_len-1):
            for i in range(tau):
                x = self._BD_step(x)
            traj.append(x)
        return np.array(traj)

    def _get_dtraj(self, traj):
        return self.coarse_grain.get_coarse_grained_clusters(traj)

    def _get_dtrajs(self, trajs):
        dtrajs = []
        for traj in trajs:
            dtrajs.append(self.coarse_grain.get_coarse_grained_clusters(traj))
        return dtrajs

    def _BD_step(self, x):
        return x + self.force(x)*self.dt + self._brownian_noise()

    def _brownian_noise(self):
        noise = np.random.normal(size=self.dim)
        noise *= np.random.normal(0, self.noise_magnitude)/np.linalg.norm(noise)
        return noise

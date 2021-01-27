from coarse_grain import K_CentersCoarseGrain

class SimpleSampler:

    def __init__(self, force, dim, dt, kT, coarse_grain_type=K_CentersCoarseGrain, **coarse_grain_args):
        self.force = force
        self.dim = dim
        self.dt = dt
        self.kT = kT
        self.coarse_grain = coarse_grain_type(**coarse_grain_args)
        self.noise_magnitude = np.sqrt(2*self.dim + self.dt)

    def sample_from_microstate(self, microstate, sample_len, n_samples, tau):
        trajs = []
        for i in range(n_samples):
            x = self.coarse_grain.sample_from(microstate)
            traj = self.sample_from_point(x, sample_len, tau)
            trajs.append(traj)
        return self._get_dtrajs(trajs)

    def sample_from_point(self, point, sample_len, tau):
        traj = []
        x = point.copy()
        for j in range(sample_len*tau + 1):
            if j%tau == 1:
                traj.append(x)
            x = self._BD_step(x)
        return np.array(traj)


    def _get_dtrajs(self, trajs):
        dtrajs = []
        for traj in drajs:
            dtrajs.append(self.coarse_grain.get_coarse_grained_clusters(traj)) 
        return dtrajs

    def _BD_step(self, x):
        return x + self.force(x)*dt + self._brownian_noise()

    def _brownian_noise(self):
        noise = np.random.normal(size=self.dim)
        noise *= np.random.normal(0, self.noise_magnitude)/np.linalg.norm(noise)
        return noise
    






    

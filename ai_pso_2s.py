from surrogate import train_surrogate
from surrogate import minimize_surrogate_gd
from surrogate import MLP

import benchmark as benchmark
import numpy as np


class PSO:
    def __init__(
        self,
        objective_function,
        dim,
        particles,
        bounds,
        num_particles,
        max_iter,
        w,
        c1,
        c2,
        ring_k=1,  # neighborhood radius: 1 => left/right (3 nodes total incl. self)
    ):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)  # shape: (dim, 2)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)

        # PSO parameters
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

        self.ring_k = int(ring_k)
        if self.ring_k < 1:
            raise ValueError("ring_k must be >= 1")

        # Initialize particles
        self.positions = np.array(particles, dtype=float)
        if self.positions.shape != (self.num_particles, self.dim):
            raise ValueError(
                f"`particles` must have shape (num_particles, dim) = ({self.num_particles}, {self.dim}), "
                f"got {self.positions.shape}"
            )

        self.velocities = np.zeros((self.num_particles, self.dim), dtype=float)

        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([self.obj_func(p) for p in self.positions], dtype=float)

        # Keep global best too (optional, useful for reporting)
        best_idx = int(np.argmin(self.pbest_scores))
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_score = float(self.pbest_scores[best_idx])

    def _ring_best_index(self, i: int) -> int:
        """
        Return the index of the best pbest within i's ring neighborhood:
        indices = [i-k, ..., i, ..., i+k] with wrap-around.
        """
        k = self.ring_k
        n = self.num_particles
        idxs = [(i + offset) % n for offset in range(-k, k + 1)]
        # choose neighbor with minimal pbest score
        best_local = min(idxs, key=lambda j: self.pbest_scores[j])
        return int(best_local)

    def optimize(self):
        for iteration in range(self.max_iter):
            progress = iteration / self.max_iter
            self.ring_k = int(
                    1 + (99 - 1) * (progress ** 2)
                )
            for i in range(self.num_particles):
                # Ring topology: use neighborhood best instead of global best
                lbest_idx = self._ring_best_index(i)
                lbest_pos = self.pbest_positions[lbest_idx]

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Velocity update (local-best / ring)
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (lbest_pos - self.positions[i])
                )

                # Position update
                self.positions[i] += self.velocities[i]

                # Apply bounds
                self.positions[i] = np.clip(
                    self.positions[i],
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                )

                # Evaluate
                score = float(self.obj_func(self.positions[i]))

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()

                    # Update global best (still tracked for output/logging)
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = self.positions[i].copy()

        return self.gbest_position, self.gbest_score
    

def of_batch(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return benchmark.rotated_griewank(X)
    return np.array([benchmark.rotated_griewank(x) for x in X], dtype=np.float32)



bounds = (-600 , 600 )

dim = 30

max_zone = 200
center = (bounds[1] + bounds[0])/2
half_range = 0.5 * (bounds[1] - bounds[0])      # 5.12
step = half_range / max_zone                    # 5.12/100

of = of_batch

values = np.array([])  
min_err  = np.inf
sr = 0
acc_err = 0.0001
real_best =   0

nb_runs = 10

for i in range(0 ,nb_runs):
    print("Iteration # " , i)
    
    particles =[]
    for i in range(1, max_zone +1):
        if i%50 ==0  :
            print("Handling Zone " , i)
        zone_min = center - i*step
        zone_max = center + i*step
        model = MLP(dim).to("cpu")
        model, scalers, device = train_surrogate(model=model ,  D=dim, n=500, epochs=50, bounds=(zone_min, zone_max) , function=of)
        best_x, best_yhat = minimize_surrogate_gd(model, scalers, D=dim, bounds=(zone_min,zone_max), steps=50, lr=0.05, n_restarts=2, device=device)
        
        particles.append(best_x)
        
    
    particles = np.array(particles)
    
    bounds = [(-600, 600)] * dim
    pso = PSO(
        objective_function=of,
        dim=dim,
        particles = particles,
        bounds=bounds,
        num_particles=200,
        max_iter=500,
            w=1/(2 * np.log(2)),
            c1=0.5 + np.log(2),
            c2=0.5 + np.log(2),
            ring_k = 2
    )
    best_position, best_score = pso.optimize()
    values = np.append(values, best_score)
    if best_score < min_err :
        min_err = best_score
    if np.abs(best_score - real_best ) < acc_err:
        sr+=1 
    print("Best Position:", best_position)
    print("Best Score:", best_score)
    
mean = np.mean(values)
std_sample = np.std(values, ddof=1)

print(f"Mean: {mean:.6e}")
print(f"STD: {std_sample:.6e}")
print(f"Min Err: {min_err:.6e}")

print("SR " , sr/nb_runs)
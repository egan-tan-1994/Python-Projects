import numpy as np
import matplotlib.pyplot as plt

# Parameters for the birth-death process

b = 0.01        # per-capita birth rate
c = 0.005       # per-capita death rate
N_max = 2       # maximum population size
T_max = 100      # maximum time
dt = 0.1        # time step
dN = 1          # state step

# Time grid

time_steps = int(T_max/dt)
t = np.linspace(0, T_max, time_steps)

# State space (population size)

n = np.arange(0, N_max+1, dN)

# Probability distribution array

P = np.zeros((N_max+1, time_steps))

# Initial condition: start with state N = 0, the rest are 0

P[0,0] = 1

# Time-stepping loop using finite difference

for t_idx in range(1, time_steps):
    for N_idx in range(1,N_max):
        # Birth term: from N-1 to N
        birth_term = b*P[N_idx-1, t_idx-1]
        # Death term: from N+1 to N
        death_term = c*P[N_idx+1, t_idx-1]
        # Update the probability P(N,t) using the master equation
        P[N_idx, t_idx] = P[N_idx, t_idx-1]+dt*(birth_term + death_term - (b+c)*P[N_idx, t_idx-1])
        # Boundary condition at N = 0 (no deaths)
        P[0,t_idx] = P[0, t_idx-1] + dt*(b*P[1, t_idx-1] - b*P[0, t_idx-1])
        # Boundary condition at N = N_max (no births)
        P[N_max, t_idx] = P[N_max, t_idx-1] + dt*(c*P[N_max-1, t_idx-1] - c*P[N_max, t_idx])

# Plot the probability distribution over time

plt.figure(figsize=(10,6))
for i in range(0, len(t), int(len(t)/5)):
        plt.plot(t, P[i,:], label=f'Time = {t[i]:.1f}')
        plt.xlabel('Time (t)', fontsize = 20)
        plt.ylabel('Probability Distribution P(N,t) of Population Size N over time t', fontsize = 20)
        plt.title('Probability Distribution of a Birth-Death Process', fontsize = 40)
plt.legend()
plt.grid(True)
plt.show()
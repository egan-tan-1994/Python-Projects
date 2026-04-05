import numpy as np
import matplotlib.pyplot as plt

def gillespie_birthdeath(num_sim, t_max, b, c, N0):
    population_trajs = []

    for _ in range(num_sim):
        N = N0
        t = 0
        population = [N]
        time = [t]

        while t < t_max:
            # Calculate the rates of the birth and death processes
            birth_rate = b * N
            death_rate = c * N
            
            total_rate = birth_rate + death_rate
            if total_rate == 0:
                break  # No more events can occur (population reached zero or other extreme cases)

            # Time to next event
            tau = np.random.exponential(1 / total_rate)
            t += tau

            # Choose whether the event is a birth or a death
            event = np.random.rand()
            if event < birth_rate / total_rate:
                N += 1  # Birth
            else:
                N -= 1  # Death

            population.append(N)
            time.append(t)

        # Append the trajectory of this simulation
        population_trajs.append((time, population))
    
    return population_trajs

# Parameters for the simulation
num_sim = 100   # number of simulations
t_max = 100     # maximum time period
b = 0.01        # per-capita birth rate
c = 0.005       # per-capita death rate
N0 = 100        # initial population size

# Run the Gillespie algorithm for the birth-death process
population_trajs = gillespie_birthdeath(num_sim, t_max, b, c, N0)

# Create a common time grid for interpolation (from 0 to t_max, with the same number of points)
num_time_points = 500
common_time = np.linspace(0, t_max, num_time_points)

# Initialize arrays to store mean and variance
pop_mean = np.zeros(num_time_points)
pop_variance = np.zeros(num_time_points)

# Interpolate population sizes for each simulation onto the common time grid
for time, population in population_trajs:
    # Interpolate the population size at each point in the common time grid
    interp_pop = np.interp(common_time, time, population)
    
    # Add the interpolated population to the mean and variance arrays
    pop_mean += interp_pop
    pop_variance += interp_pop ** 2

# Compute the mean and variance for all simulations
pop_mean /= num_sim
pop_variance = (pop_variance / num_sim) - pop_mean**2

# Plot the population trajectories
plt.figure(figsize=(15, 10))
for time, population in population_trajs:
    # Interpolate each population trajectory onto the common time grid
    interp_pop = np.interp(common_time, time, population)
    plt.plot(common_time, interp_pop, color='red', alpha=0.2)

plt.xlabel("Time (t)", fontsize = 30)
plt.ylabel("Population Size N(t)", fontsize = 30)
plt.title("Population Size Over Time (Multiple Simulations)", fontsize = 40)
plt.grid(True)

# Create a new figure for mean and variance plots
plt.figure(figsize=(10, 6))

# Plot Mean Population Size
plt.figure(figsize=(15, 10))
plt.plot(common_time, pop_mean, label='Mean Population Size', color='blue')
plt.title('Mean Population Size Over Time', fontsize = 40)
plt.xlabel('Time (t)', fontsize = 30)
plt.ylabel(r'Mean Population Size $\left<N(t)\right>$', fontsize = 30)
plt.grid(True)

# Plot Population Size Variance
plt.figure(figsize=(15, 10))
plt.plot(common_time, pop_variance, label='Variance of Population Size', color='green')
plt.title('Variance of Population Size Over Time', fontsize = 40)
plt.xlabel('Time (t)', fontsize = 30)
plt.ylabel(r'Population Size Variance $\sigma^2_N(t)$', fontsize = 30)
plt.grid(True)

# Adjust the layout and show the plots
plt.tight_layout()
plt.show()

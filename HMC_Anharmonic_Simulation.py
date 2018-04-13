import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statistics as stat
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


# Calculate anharmonic potential.
def U(x):
	fSquared = 1
	l  = 1
	return l * (x*x - fSquared)* (x*x - fSquared)

# Calculate gradiant of harmonic potential.
def grad_U(x):
	fSquared = 1
	l  = 1
	return 4.0 * l * (x * x -fSquared) * 2 * x

# Calculate the Hamiltonian of the system.
def H(x,p):
	return 0.5 * p * p + U(x)

# Do the metropolis update, returns probability of state being accepted.
def metropolis(H_before, H_after):
	deltaH = H_after - H_before
	return min(np.exp(-deltaH), 1)

# Do single leap frog step.
def leapFrog(x, p, epsilon):

	# Half step in momentum. 
	p = p - epsilon/2.0 * grad_U(x)

	# Full step in position.
	x = x + epsilon * p

	# Half step in momentum.
	p = p - epsilon/2.0 * grad_U(x)

	# Return updated position and momentum.
	return x, p

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Set our plot to look like xkcd comics.
plt.xkcd()

# Create figure for the plot.
fig = plt.figure()

# Create axis to plot position-momentum on.
xpPlot = plt.subplot(121,frameon=True)

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-2.5,2.5)
plt.xlim(-2.5,2.5)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Momentum', fontsize=14, color='black')

# Create axis to plot position-potential on.
xVPlot = plt.subplot(122,frameon=True)

# Add the actual potential in the background.
X = np.linspace(-2.1, 2.1)
plt.plot(X,U(X),linestyle='--')

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-0.1,2.5)
plt.xlim(-2.1,2.1)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Potential', fontsize=14, color='black')


# Initial position.
initial_Position = np.random.rand()

# Initial momentum.
initial_Momentum = np.random.randn()

# Step size. 
epsilon = 0.1

# Number of steps.
N = 10

# Total number of steps.
totalSteps = 200

# Step counter (will be used to determine when we have reached the end of a single leapfrog iteration).
stepCounter = 0

# Define variable to hold the Monte Carlo estimate on position.
meanPostition = []

# Plot the initial values on the relative graphs.
xpPoint, = xpPlot.plot(initial_Position,initial_Momentum, marker='o', color='black')
xVPoint, = xVPlot.plot(initial_Position,U(initial_Position), marker='o',color='black')

# Store the true state of the system, this stays constant whilst the hmc update is taking place since 
# the proposed state could be rejected. Initially these values are the initial conditions.
current_position = initial_Position
current_momentum = initial_Momentum

# Function that gets called each time a new frame is needed for the animation.
def hmcUpdate(frame):
	# In order to access the global versions of these variables.
	global stepCounter
	global current_position
	global current_momentum
	global totalSteps

	if frame > 0:

		# Create local position and momentum variables for the evolution of the system. 
		position = 0
		momentum = 0

		# Check to see whether we ave just done an update.
		if stepCounter == 0:

			# If we have just done an update need to start from the updated position.
			position = current_position

			# Momentum should be normally distributed. 
			momentum = np.random.randn()

		else:
			# Otherwise the position and momentum will just be that from the last leap frog step.
			position = xpPoint.get_xdata()
			momentum = xpPoint.get_ydata()


		# Update position according to Hamiltonian dynamics. 
		updated_position , updated_momentum = leapFrog(position, momentum, epsilon)

		# Increment step counter.
		stepCounter += 1

		# Check whether we have reached the end of a leap frog trajectory.
		if stepCounter == 10:

			# Reset counter.
			stepCounter = 0

			# Record proposed state.
			proposed_position , proposed_momentum = updated_position, updated_momentum

			# Calculate the Hamiltonian of the current state.
			current_Hamiltonian  = H(current_position, current_momentum)

			# Calculate the Hamiltonian of the proposed state.
			proposed_Hamiltonian = H(proposed_position, proposed_momentum)

			# Do a Metropolis update.
			if np.random.rand() < metropolis(current_Hamiltonian, proposed_Hamiltonian):

				# If it succeeds plot the new state in green.
				xpPlot.plot(proposed_position, proposed_momentum, markersize=2, marker='o', color='green')
				xVPlot.plot(proposed_position,U(proposed_position), markersize=2, marker='o',color='green')


				# Proposed state becomes the current state.
				current_position , current_momentum = proposed_position , proposed_momentum

				# Record sample.
				meanPostition.append(current_position)

				# We don't want to draw the data at this point since it will draw over the top of 
				# our nice green point. 


			else:

				# Otherwise plot new state in red.
				xpPlot.plot(proposed_position, proposed_momentum, markersize=2, marker='o', color='red')
				xVPlot.plot(proposed_position,U(proposed_position), markersize=2, marker='o',color='red')

				# Current state remains as it is. 

				# Record sample.
				meanPostition.append(current_position)

				# We don't want to draw the data at this point since it will draw over the top of 
				# our nice red point. 


		else:
			# If not just plot the points as normal and move on.
			xpPoint.set_xdata(updated_position)
			xpPoint.set_ydata(updated_momentum)

			xVPoint.set_xdata(updated_position)
			xVPoint.set_ydata(U(updated_position))


		# If we are on the final frame print the monte carlo position estimate.
		# Average over the recorded samples.
		#if frame == totalSteps:
		#	X = round(stat.mean(meanPostition),3)
		#	xVPlot.annotate(r'$  \bar{x}  \approx  $' + str(X), xy=(0.0, 2.0), xycoords="data", 
		#	va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))

# This function call actually animates the graph.
ani = animation.FuncAnimation(fig, hmcUpdate, totalSteps+1, repeat=False, interval=100)

# Make sure there is no overlap in the plots.
plt.tight_layout()

ani.save('HMC-Anharmonic1.mp4', writer=writer)

# Show the graph to the user.
#plt.show()


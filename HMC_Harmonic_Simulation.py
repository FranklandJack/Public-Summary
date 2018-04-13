import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statistics as stat
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Calculate harmonic potential.
def U(x):
	muSquared = 1
	return 0.5 * muSquared * x *x

# Calculate gradiant of harmonic potential.
def grad_U(x):
	muSquared = 1
	return muSquared * x

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
plt.ylim(-5.1,5.1)
plt.xlim(-5.1,5.1)

# Label the axis. 
plt.xlabel(r'$x$', fontsize=14, color='black',rotation=0)
plt.ylabel(r'$p$', fontsize=14, color='black',rotation=0)

# Create axis to plot position-potential on.
xVPlot = plt.subplot(122,frameon=True)

# Set plot range.
xVRange = 10

# Add the actual potential in the background.
X = np.linspace(-5.1, 5.1)
plt.plot(X,U(X),linestyle='--')

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-0.1,5)
plt.xlim(-5.1,5.1)

# Label the axis. 
plt.xlabel(r'$x$', fontsize=14, color='black',rotation=0)
plt.ylabel(r'$E(x)$', fontsize=14, color='black',rotation=0)


# Initial position.
initial_Position = 0

# Initial momentum.
initial_Momentum = 0

# Step size. 
epsilon = 0.1

# Number of leap frog steps.
N = 10

# total number of steps 
totalSteps = 200

# Step counter (will be used to determine when we have reached the end of a single leapfrog iteration).
stepCounter = 0

# Create initial values to plot on graphs, these won't get plotted here but in the update function.
xpPoint, = xpPlot.plot(initial_Position,initial_Momentum, markersize='4', marker='o', color='black')
xVPoint, = xVPlot.plot(initial_Position,U(initial_Position), markersize='4', marker='o',color='black')

# Store the true state of the system, this stays constant whilst the hmc update is taking place since 
# the proposed state could be rejected. Initially these values are the initial conditions.
current_position = initial_Position
current_momentum = initial_Momentum

# Define variable to hold the Monte Carlo estimate on position.
meanPostition = []

# Function that gets called each time a new frame is needed for the animation.
def hmcUpdate(frame):
	# In order to access the global versions of these variables.
	global stepCounter
	global current_position
	global current_momentum
	global meanPostition
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
				xpPlot.plot(proposed_position, proposed_momentum, markersize='2',  marker='o', color='green')
				xVPlot.plot(proposed_position,U(proposed_position), markersize='2', marker='o',color='green')


				# Proposed state becomes the current state.
				current_position , current_momentum = proposed_position , proposed_momentum

				# Record this sample.
				meanPostition.append(current_position)

				# We don't want to draw the data at this point since it will draw over the top of 
				# our nice green point. 


			else:

				# Otherwise plot new state in red.
				xpPlot.plot(proposed_position, proposed_momentum, markersize='2', marker='o', color='red')
				xVPlot.plot(proposed_position,U(proposed_position), markersize='2', marker='o',color='red')

				# Current state remains as it is. 

				# Record this sample.
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
		#	xpPlot.annotate(r'$  \bar{x}  \approx  $' + str(X), xy=(0.0, -4.0), xycoords="data", 
		#	va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))	


# This function call actually animates the graph.
ani = animation.FuncAnimation(fig, hmcUpdate,totalSteps+1, repeat=False, interval=100)

# Make sure there is no overlap in the plots.
plt.tight_layout()

# Save the animation.
ani.save('HMC-Harmonic1.mp4', writer=writer)




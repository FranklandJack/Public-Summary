import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Calculate anharmonic potential.
def U(x):
	muSquared = -4
	l  = 0.1
	return 0.5 * muSquared * x * x + l * x * x * x * x

# Calculate gradiant of harmonic potential.
def grad_U(x):
	muSquared = 1
	l  = 1
	return muSquared * x + 4.0 * l * x * x * x

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

def data_gen(q=0, p=1):
	steps = 0
	epsilon = 0.1
	while steps < 300:
		steps += 1 
		q , p = leapFrog(q, p, epsilon)
		yield q, p

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Set our plot to look like xkcd comics.
plt.xkcd()

# Create figure for the plot.
fig = plt.figure()

# Create axis to plot position-momentum on.
xpPlot = plt.subplot(121,frameon=True)

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-2,2)
plt.xlim(-10,10)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Momentum', fontsize=14, color='black')

# Create axis to plot position-potential on.
xVPlot = plt.subplot(122,frameon=True)

# Add the actual potential in the background.
X = np.linspace(-10, 10)
plt.plot(X,U(X),linestyle='--')

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-15,15)
plt.xlim(-10,10)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Potential', fontsize=14, color='black')


# Inital position.
initial_Position = 0

# Initial momentum.
initial_Momentum = 1

# Step size 
epsilon = 0.1

# Plot the initial values on the relative graphs.
xpPoint, = xpPlot.plot(initial_Position,initial_Momentum, marker='o', color='black')
xVPoint, = xVPlot.plot(initial_Position,U(initial_Position), marker='o',color='black')

# Function that gets called each time a new frame is needed for the animation.
def update(frame):
	# Update position according to Hamiltonian dynamics. 
	updated_Position , updated_Momentum = leapFrog(xpPoint.get_xdata(), xpPoint.get_ydata(), epsilon)

	# Set new values on relative graphs. 
	xpPoint.set_xdata(updated_Position)
	xpPoint.set_ydata(updated_Momentum)

	xVPoint.set_xdata(updated_Position)
	xVPoint.set_ydata(U(updated_Position))

def update2(frame):
	updated_Position , updated_Momentum = frame
	xpPoint.set_xdata(updated_Position)
	xpPoint.set_ydata(updated_Momentum)

	xVPoint.set_xdata(updated_Position)
	xVPoint.set_ydata(U(updated_Position))

# This function call actually animates the graph.
ani = animation.FuncAnimation(fig, update2, data_gen, repeat=False, interval=10, save_count=300)

# Make sure there is no overlap in the plots.
plt.tight_layout()


ani.save('Anharmonic-Motion.mp4', writer=writer)

# Show the graph to the user.
plt.show()


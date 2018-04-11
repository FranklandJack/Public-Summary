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

# This stuff is required for saving the animation.
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
plt.xlim(-2,2)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Momentum', fontsize=14, color='black')

# Create axis to plot position-potential on.
xVPlot = plt.subplot(122,frameon=True)

# Add the actual potential in the background.
X = np.linspace(-4, 4)
plt.plot(X,U(X),linestyle='--')

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-0.1,0.6)
plt.xlim(-1.1,1.1)

# Label the axis. 
plt.xlabel('Position', fontsize=14, color='black')
plt.ylabel('Potential', fontsize=14, color='black')


# Inital position.
initial_Position = 0

# Initial momentum.
initial_Momentum = 1

# Step size 
epsilon = 0.1

# Create points to be plotted.
xpPoint, = xpPlot.plot(initial_Position,initial_Momentum, marker='o', color='black')
xVPoint, = xVPlot.plot(initial_Position,U(initial_Position), marker='o',color='black')

totalFrames = 300

# Function that gets called each time a new frame is needed for the animation.
def update(frame):
	# First frame should just be initial values.
	if frame > 0:
	
		# Update position according to Hamiltonian dynamics. 
		updated_Position , updated_Momentum = leapFrog(xpPoint.get_xdata(), xpPoint.get_ydata(), epsilon)

		# Set new values on relative graphs. 
		xpPoint.set_xdata(updated_Position)
		xpPoint.set_ydata(updated_Momentum)

		xVPoint.set_xdata(updated_Position)
		xVPoint.set_ydata(U(updated_Position))

# This function call actually animates the graph.
ani = animation.FuncAnimation(fig, update, totalFrames+1, repeat=False, interval=10)

# Make sure there is no overlap in the plots.
plt.tight_layout()

# Save the plot
ani.save('Harmonic-Motion.mp4', writer=writer)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

totalSteps = 200

def data_gen():
	global totalSteps
	
	steps = 0
	 

	while steps < totalSteps:
		steps += 1
		
		x = np.random.random()* 2 - 1
		y = np.random.random()* 2 - 1
		if steps != totalSteps:
			yield x, y, False
		else:
			yield x, y, True

		 

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

# Set our plot to look like xkcd comics.
plt.xkcd()

# Create a circle to go on our plot.
circle = plt.Circle((0,0), 1,fill = False)

# Create list to hold the points.
data = []

# Create figure for the plot.
fig = plt.figure()

# Create axis to plot position-momentum on.
xpPlot = plt.subplot(111,frameon=True)
xpPlot.add_artist(circle)

# Set axis limits so we can actually see the dynamics. 
plt.ylim(-1.1,1.1)
plt.xlim(-1.1,1.1)

# Label the axis. 
plt.xlabel('x', fontsize=14, color='black')
plt.ylabel('y', fontsize=14, color='black')

# Make sure there is no overlap in the plots.
plt.tight_layout()


circleCount = 0



def init():
	xpPlot.plot(0, 0, markersize='4', marker='', color='black')

# Function that gets called each time a new frame is needed for the animation.
def update(frame):
	global data
	global circleCount
	global totalSteps
	
	x = np.random.random()* 2 - 1
	y = np.random.random()* 2 - 1

	xpPoint, = xpPlot.plot(x, y, markersize='4', marker='o', color='black')

	data.append(xpPoint)

	if frame==totalSteps:
		for point in data:
			if point.get_xdata()*point.get_xdata() + point.get_ydata()*point.get_ydata() <= 1:
				point.set_color('green')
				circleCount+=1

			else:
				point.set_color('red')
		# Estimate pi. 
		piEstimate = 4 * circleCount/len(data)
		xpPlot.annotate(r'$\pi \approx $' + str(piEstimate), xy=(0.0, 0.0), xycoords="data",
            va="center", ha="center",
            bbox=dict(boxstyle="round", fc="w"))


# This function call actually animates the graph.
ani = animation.FuncAnimation(fig, update, np.arange(1,totalSteps+1), init_func=init, repeat=False, interval=100,save_count=totalSteps)

ani.save('Pi-Motion.mp4', writer=writer)
# Show the graph to the user.
#plt.show()

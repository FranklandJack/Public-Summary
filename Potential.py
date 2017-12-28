import numpy as np
import matplotlib.pyplot as plt 

# Calculate anharmonic potential.
def U(x):
	return 0.5 * x * x

def P(x):
	return np.exp(-U(x))


# Set our plot to look like xkcd comics.
plt.xkcd()

# Create figure for the plot.
fig = plt.figure()

# Create axis to plot position-momentum on.
energyPlot = plt.subplot(121,frameon=True)


# Label the axis. 
plt.xlabel(r'$x$', fontsize=14, color='black')
plt.ylabel(r'$E(x)$', fontsize=14, color='black')

# Remove numbers since pdf will not be normalised.
energyPlot.set_yticklabels([])
energyPlot.set_xticklabels([])

# Create axis to plot position-potential on.
probabilityPlot = plt.subplot(122,frameon=True)

# Label the axis. 
plt.xlabel(r'$x$', fontsize=14, color='black')
plt.ylabel(r'$P(x)$', fontsize=14, color='black')

# Remove numbers since pdf will not be normalised.
probabilityPlot.set_yticklabels([])
probabilityPlot.set_xticklabels([])

# Add the actual potential in the background.
X = np.linspace(-4, 4)

# Plot the relative functions on their graphs.
energyPlot.plot(X,U(X),linestyle='-')

probabilityPlot.plot(X,P(X),linestyle='-')


# Make sure there is no overlap in the plots.
plt.tight_layout()

plt.savefig('Example-Harmonic-Potential.png')
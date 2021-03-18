import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def generate_data(nbr_iterations, nbr_elements):
    """
    Generates dummy data.
    The elements will be assigned random initial positions and speed.
    Args:
        nbr_iterations (int): Number of iterations data needs to be generated for.
        nbr_elements (int): Number of elements (or points) that will move.
    Returns:
        list: list of positions of elements. (Iterations x (# Elements x Dimensions))
    """
    dims = (3, 1)

    # Random initial positions.
    gaussian_mean = np.zeros(dims)
    gaussian_std = np.ones(dims)
    start_positions = np.array(list(map(np.random.normal, gaussian_mean, gaussian_std, [nbr_elements] * dims[0]))).T

    # Random speed
    start_speed = np.array(list(map(np.random.normal, gaussian_mean, gaussian_std, [nbr_elements] * dims[0]))).T

    # Computing trajectory
    data = [start_positions]
    for iteration in range(nbr_iterations):
        previous_positions = data[-1]
        new_positions = previous_positions + start_speed
        data.append(new_positions)

    return data


def animate_scatters(iteration, data, scatters, iter_text, plane_off):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    iter_text.set_text(f'ITERATION {iteration:3d}/{len(data):3d}')
    for i in range(data[0].shape[0]):
        scatters['3d'][i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], data[iteration][i, 2:])
        scatters['xy'][i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], plane_off)
        scatters['yz'][i]._offsets3d = (plane_off, data[iteration][i, 1:2], data[iteration][i, 2:])
        scatters['xz'][i]._offsets3d = (data[iteration][i, 0:1], plane_off, data[iteration][i, 2:])
    return tuple(scatters[k] for k in scatters)


def plot_swarm(data, fobj=None, lowlim=-4, highlim=4, save=False):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
        fobj:
    """

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(15, 15))
    ax = p3.Axes3D(fig)
    from matplotlib.colors import ListedColormap

    # Choose colormap
    cmap = cm.coolwarm
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:, -1] = 0.25
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    if fobj is not None:
        x = np.linspace(lowlim, highlim, 1000)
        y = np.linspace(lowlim, highlim, 1000)
        xx, yy = np.meshgrid(x, y)
        zz = fobj(xx, yy)
        # Plot the surface.
        ax.plot_surface(xx, yy, zz, cmap=my_cmap, linewidth=0, antialiased=False)
        ax.contour(xx, yy, zz, 10, zdir='z', cmap="autumn_r", linestyles="solid", offset=lowlim, alpha=0.75)
        ax.contour(xx, yy, zz, 10, zdir='y', cmap="autumn_r", linestyles="solid", offset=lowlim, alpha=0.75)
        ax.contour(xx, yy, zz, 10, zdir='x', cmap="autumn_r", linestyles="solid", offset=lowlim, alpha=0.75)

    # Initialize scatters
    scatters = dict()
    scatters['3d'] = [ax.scatter(data[0][i, 0:1], data[0][i, 1:2], data[0][i, 2:], edgecolors='black')
                      for i in range(data[0].shape[0])]
    scatters['xy'] = [ax.scatter(data[0][i, 0:1], data[0][i, 1:2], [lowlim], edgecolors='black')
                      for i in range(data[0].shape[0])]
    scatters['yz'] = [ax.scatter([lowlim], data[0][i, 1:2], data[0][i, 2:], edgecolors='black')
                      for i in range(data[0].shape[0])]
    scatters['xz'] = [ax.scatter(data[0][i, 0:1], [lowlim], data[0][i, 2:], edgecolors='black')
                      for i in range(data[0].shape[0])]

    # Number of iterations
    iterations = len(data)
    iter_text = ax.text2D(0.05, 0.9, f"ITERATION {0:3d}/{iterations:3d}", transform=ax.transAxes)

    # Setting the axes properties
    ax.set_xlim3d([lowlim, highlim])
    ax.set_xlabel('X')
    ax.set_ylim3d([lowlim, highlim])
    ax.set_ylabel('Y')
    #ax.set_zlim3d([lowlim, highlim])
    ax.set_zlabel('Z')
    ax.set_title('PSO Visualized')

    # Provide starting angle for the view.
    ax.view_init(30, 45)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters, iter_text, lowlim),
                                  interval=500, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)

    plt.show()

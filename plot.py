import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


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
        if '3d' in scatters:
            scatters['3d'][i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], data[iteration][i, 2:])
        if 'xy' in scatters:
            scatters['xy'][i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], plane_off)
            scatters['yz'][i]._offsets3d = (plane_off, data[iteration][i, 1:2], data[iteration][i, 2:])
            scatters['xz'][i]._offsets3d = (data[iteration][i, 0:1], plane_off, data[iteration][i, 2:])
    return tuple(scatters[k] for k in scatters)


def plot_swarm(swarm, fobj, plot_surf=False, plot_proj=True, save=False):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        particles_history (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
        fobj:
    """

    particles_history = swarm.particles_history
    cost_history = swarm.cost_history
    gbest_history = swarm.gbest_history

    # 3d plot
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    ax = fig.add_subplot(gs[:, 0:3], projection='3d')

    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    xx, yy = np.meshgrid(x, y)
    zz = fobj(xx, yy)
    xmin = np.min(x)
    # Plot the surface, projections and initialize scatters
    scatters = dict()
    if plot_surf:
        ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.5, linewidth=0, antialiased=False)
        scatters['3d'] = [
            ax.scatter(particles_history[0][i, 0:1], particles_history[0][i, 1:2], particles_history[0][i, 2:],
                       edgecolors='black')
            for i in range(particles_history[0].shape[0])]
    if plot_proj:
        ax.contourf(xx, yy, zz, 10, zdir='z', cmap="autumn_r", linestyles="solid", offset=xmin, alpha=0.5)
        ax.contourf(xx, yy, zz, 10, zdir='y', cmap="autumn_r", linestyles="solid", offset=xmin, alpha=0.5)
        ax.contourf(xx, yy, zz, 10, zdir='x', cmap="autumn_r", linestyles="solid", offset=xmin, alpha=0.5)
        scatters['xy'] = [
            ax.scatter(particles_history[0][i, 0:1], particles_history[0][i, 1:2], [xmin], edgecolors='black')
            for i in range(particles_history[0].shape[0])]
        scatters['yz'] = [
            ax.scatter([xmin], particles_history[0][i, 1:2], particles_history[0][i, 2:], edgecolors='black')
            for i in range(particles_history[0].shape[0])]
        scatters['xz'] = [
            ax.scatter(particles_history[0][i, 0:1], [xmin], particles_history[0][i, 2:], edgecolors='black')
            for i in range(particles_history[0].shape[0])]

    # Number of iterations
    iterations = len(particles_history)
    iter_text = ax.text2D(0.05, 0.9, f"ITERATION {0:3d}/{iterations:3d}", transform=ax.transAxes)

    # Setting the axes properties
    # ax.set_xlim3d([lowlim, highlim])
    ax.set_xlabel('X')
    # ax.set_ylim3d([lowlim, highlim])
    ax.set_ylabel('Y')
    # ax.set_zlim3d([lowlim, highlim])
    ax.set_zlabel('Z')
    ax.set_title(f'PSO, {fobj.__name__} function')

    # Provide starting angle for the view.
    ax.view_init(30, 45)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations,
                                  fargs=(particles_history, scatters, iter_text, xmin),
                                  interval=500, blit=False, repeat=True)

    # 2d plots
    ax = fig.add_subplot(gs[:2, 3])
    ax.plot(cost_history)
    ax.set_title('Cost history')
    ax.set_xlabel('iteration')
    ax.set_ylabel('cost')

    ax = fig.add_subplot(gs[2:, 3])
    xs = [x[0] for x in gbest_history]
    ys = [x[1] for x in gbest_history]
    ax.scatter(xs, ys)
    ax.set_title('gbest')
    ax.set_xlabel('iteration')
    ax.set_ylabel('cost')


    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Diego Chinellato'), bitrate=1800,
                        extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)

    fig.canvas.manager.full_screen_toggle()
    plt.show()

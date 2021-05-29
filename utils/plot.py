import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_frame(iteration, particles_history, scatters, iter_text, plane_off):
    """
    updates the animation at each frame
    Parameters:
        iteration: int
            Current iteration of the animation
        particles_history: list
            list of the particles' positions at each iteration
        scatters: dict with keys '3d', optional 'xy', 'yz', 'xz'
            each element is a list of scatter
        iter_text: text2D
            text cell for iterations
        plane_off: float
            offset, used to correctly plot projections
    Returns:
        tuple: tuple of scatters (one per element) with new coordinates
    """
    iter_text.set_text(f'ITERATION {iteration:3d}/{len(particles_history):3d}')
    for i in range(particles_history[0].shape[0]):
        if '3d' in scatters:
            scatters['3d'][i]._offsets3d = (particles_history[iteration][i, 0:1],
                                            particles_history[iteration][i, 1:2],
                                            particles_history[iteration][i, 2:])
        if 'xy' in scatters:
            scatters['xy'][i]._offsets3d = (particles_history[iteration][i, 0:1],
                                            particles_history[iteration][i, 1:2],
                                            plane_off)
            scatters['yz'][i]._offsets3d = (plane_off,
                                            particles_history[iteration][i, 1:2],
                                            particles_history[iteration][i, 2:])
            scatters['xz'][i]._offsets3d = (particles_history[iteration][i, 0:1],
                                            plane_off,
                                            particles_history[iteration][i, 2:])
    return tuple(scatters[k] for k in scatters)


def plot_swarm(swarm, fobj, plot_surf=False, plot_proj=True, bounds=None, save=False, fullscreen=False):
    """
    plots a 3d animation of the swarm, plus two 2d plots summarizing performances
    Parameters:
        swarm: AbstractOptimizer
            the swarm object with non-empty histories
        fobj: function
            function that has been optimized, must accept 2 parameters x and y
        save: bool
            whether to save the recording of the animation. (Default to False).
        plot_surf: bool
            if True, plots 3d surface of the function
        plot_proj: bool
            if True, plots 2d projections (on the 3d axis) of the function on each plane
        bounds: tuple of np.array, default None
            bounds on the search space (
        fullscreen: bool
            if True, figure is shown fullscreen

    """

    particles_history = swarm.particles_history
    cost_history = swarm.cost_history
    pbest_history = swarm.avg_pbest_history

    # 3d plot
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    ax = fig.add_subplot(gs[:, 0:3], projection='3d')
    if bounds is None:
        lb, ub = (-5, -5), (5, 5)
    else:
        lb, ub = bounds
    x = np.linspace(lb[0], ub[0], 1000)
    y = np.linspace(lb[1], ub[1], 1000)
    xx, yy = np.meshgrid(x, y)
    zz = fobj(xx, yy)
    # plot surface, projections and initialize scatters
    scatters = dict()
    xmin, ymin, zmin = np.min(x), np.min(y), np.min(zz)  # needed for correctly plotting projections
    if plot_surf:
        ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.5, linewidth=0, antialiased=False)
        scatters['3d'] = [
            ax.scatter(particles_history[0][i, 0:1], particles_history[0][i, 1:2], particles_history[0][i, 2:],
                       edgecolors='black')
            for i in range(particles_history[0].shape[0])]
    if plot_proj:
        ax.contour(xx, yy, zz, 10, zdir='z', cmap="autumn_r", linestyles="solid", offset=lb[0], alpha=0.5)
        ax.contour(xx, yy, zz, 10, zdir='y', cmap="autumn_r", linestyles="solid", offset=lb[0], alpha=0.5)
        ax.contour(xx, yy, zz, 10, zdir='x', cmap="autumn_r", linestyles="solid", offset=lb[0], alpha=0.5)
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

    # ax.set_xlim3d([lowlim, highlim])
    ax.set_xlabel('X')
    # ax.set_ylim3d([lowlim, highlim])
    ax.set_ylabel('Y')
    # ax.set_zlim3d([lowlim, highlim])
    ax.set_zlabel('Z')
    ax.set_title(f'PSO, {fobj.__name__} function')

    # starting angle for the view
    ax.view_init(30, 45)

    ani = animation.FuncAnimation(fig, update_frame, iterations,
                                  fargs=(particles_history, scatters, iter_text, xmin),
                                  interval=500, blit=False, repeat=True)

    # 2d plots
    ax = fig.add_subplot(gs[:2, 3])
    ax.plot(cost_history)
    ax.set_title('Global best value, per iteration')
    ax.set_xlabel('iteration')
    ax.set_ylabel('cost')

    ax = fig.add_subplot(gs[2:, 3])
    ax.plot(pbest_history)
    ax.set_title('Average pbest, per iteration')
    ax.set_xlabel('iteration')
    ax.set_ylabel('cost')

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Diego Chinellato'), bitrate=1800,
                        extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)
    if fullscreen:
        fig.canvas.manager.full_screen_toggle()

    plt.show()

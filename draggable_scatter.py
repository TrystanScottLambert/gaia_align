"""
Draggable scatter plot which is used for gaia points with images.
"""

import numpy as np
import pylab as plt

class DraggableScatter():
    """Class for a scatter plot that the user can drag with the mouse."""
    epsilon = 5

    def __init__(self, scatter):
        "Initiliazing."
        self.scatter = scatter
        self._ind = None
        self.ax = scatter.axes
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        plt.show()


    def get_ind_under_point(self, event):
        "Idenifying the scatter point we are clickin."
        xy = np.asarray(self.scatter.get_offsets())
        xyt = self.ax.transData.transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        ind = d.argmin()

        if d[ind] >= self.epsilon:
            ind = None
        return ind, xy[ind]

    def button_press_callback(self, event):
        """Clicking the button"""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind, pos = self.get_ind_under_point(event)
        self.x_ref = pos[0]
        self.y_ref = pos[1]


    def button_release_callback(self, event):
        """Letting go of the button"""
        if event.button != 1:
            return
        self._ind = None

        xy = np.asarray(self.scatter.get_offsets())
        x_offset = event.xdata - self.x_ref
        y_offset = event.ydata - self.y_ref

        for idx, _ in enumerate(xy):
            xy[idx] = np.array([xy[idx][0]+x_offset, xy[idx][1]+y_offset])

        self.scatter.set_offsets(xy)
        self.canvas.draw_idle()
        self.positions = xy


    def motion_notify_callback(self, event):
        """Moving the mouse"""
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def base_line(x, y, theta):
    """Calculate the parameters of the base line for a point and direction.

    Parameters
    ----------
      x : float
        X coordinate of the point
      y : float
        Y coordinate of the point
      theta : float
        Direction with respect to the X axis

    Returns
    -------
      rl : float
        Base radius of the line
      theta_l : float
        Angle of the line with respect to the X axis
      d0 : float
        Initial distance from the point to the line
      arrow : np.array
        Unit vector pointing in the +ve direction of the line
    """
    ux = np.cos(theta)
    uy = np.sin(theta)

    # Calculate the base radius and offset angle
    # Normal vector is [-uy, ux]
    norm = np.array([-uy, ux])
    rl = np.dot(norm, [x, y])
    if rl < 0:
        rl = -rl
        norm = -norm
    theta_l = np.arctan2(norm[1], norm[0])

    # Calculate initial distance
    arrow = np.array([norm[1], -norm[0]])
    d0 = np.dot(arrow, [x, y])
    return rl, theta_l, d0, arrow

class Involute:
    """An involute of a circle

    Represents a single involute surface from a circle centred at the origin.
    It is defined by its base radius of the circle and phase a0 ∈ [-π;π]
    """
    def __init__(self, rb, a0):
        """Initialize an involute
        """

        # Check that the base radius is positive
        if rb <= 0:
            raise ValueError("Base radius must be positive")

        # Check that the phase is in the range [-π;π]
        if a0 < -np.pi or a0 > np.pi:
            raise ValueError("Phase must be in the range [-π;π]")
        self.rb = rb
        self.a0 = a0

    def __repr__(self):
        return f"Involute({self.rb}, {self.a0})"

    def xy(self, t):
        """Get X Y coordinates of the curve given a parameter t

        Parameters
        ----------
          t : np.array
            Parameter of the involute curve

        Returns
        -------
          x : np.array
            X coordinates of the involute curve
          y : np.array
            Y coordinates of the involute curve
        """
        x = self.rb*(np.cos(t + self.a0) + t * np.sin(t + self.a0))
        y = self.rb*(np.sin(t + self.a0) - t * np.cos(t + self.a0))
        return x, y

    def _phase(self, d, rl, theta_l):
        """Give phase in multiples of pi

        The position is specified on the line given by the radius rl and angle
        theta_l (line is tangent to the tip of vector of length |rl| at angle
        theta_l). Distance is measured from the point defining the line, and
        +ve values are in the direction clockwise.

        Note that for rl < rb, the value of phase becomes complex. Then we
        take just its real part.
        """
        r = np.sqrt(rl**2 + d**2)
        t = np.where(r > self.rb, np.sqrt(r**2/self.rb**2 - 1), 0)
        arc = np.where(r > self.rb, np.arccos(self.rb/r), 0)
        return (theta_l - self.a0 - np.arctan(d/rl) - t + arc) / np.pi

    def distance(self, x, y, theta):
        """Calculate the distance to the involute.

        Parameters
        ----------
          x0 : float
            X coordinate of the point
          y0 : float
            Y coordinate of the point
          theta : float
            Direction with respect to the X axis

        Returns
        -------
          dist : float
            Distance to the involute curve
        """
        # Transform the point to the position along the base line
        rl, theta_l, start, arrow = base_line(x, y, theta)
        ux = np.cos(theta)
        uy = np.sin(theta)
        forward = np.dot(arrow, [ux, uy]) > 0
        dir = 1 if forward else -1

        d0 = start

        # Calculate the phase
        p = self._phase(d0, rl, theta_l)

        p_max = self._phase(-self.rb, rl, theta_l)
        # Calculate the RHS of the equation (next multiple of 2 in the direction)

        if np.floor(p_max/2.0) == np.floor(p/2.0): # and dir * (-self.rb - d0) >= 0:
            # Reflect around the maximum
            # Taylor expansion to 2nd order can often fail
            # # We need to make sure that the ray will move away from the maximum
            # d0 = d0 + 2.0 * (-rb - d0)
            # Use the quadratic guess
            p_loc = p_max - 2.0 * np.floor(p_max/2.0)
            d0 = -self.rb + dir * np.sqrt(2.0 * p_loc * self.rb * rl* np.pi)

        if forward != (d0 > -self.rb):
            rhs = 2.0 * np.ceil(p / 2.0)
        else:
            rhs = 2.0 * np.floor(p / 2.0)

        # Last check may not be necessary since this particles have already been
        # moved to RHS of the maximum
        if rl < self.rb and (d0 + self.rb) >= 0:# and dir * (-self.rb - start) >= 0:
            val = theta_l - self.a0 - rhs*np.pi
            trig = np.arccos(rl/self.rb)
            # print(f"val: {val}, trig: {trig} {d0}")
            if trig >= val or -trig <= -val:
                dist = np.tan(theta_l - self.a0) * rl
                dist = dist - start if forward else start - dist
                if dist < 0:
                    print(f"CG Distance is negative: {self} {x} {y} {theta}")
                return dist

        # Check if the point is in the complex gap
        # If it is push it to the edge
        if d0 >= -self.rb and d0**2 < self.rb**2 - rl**2:
            d0 = np.sqrt(self.rb**2 - rl**2)

        # Calculate the distance
        def f(t):
            return self._phase(t, rl, theta_l) - rhs

        dist = newton(f, d0)
        dist = dist - start if forward else start - dist

        if dist < 0:
            print(f"Distance is negative: {self} {x} {y} {theta}")
        return dist


def plot_involute(inv, t_max):
    """Plot an involute curve

    Parameters
    ----------
      inv : Involute
        Involute to plot
      t_max : float
        Maximum value of the parameter
    """
    t = np.linspace(0, t_max, 1000)
    x, y = inv.xy(t)
    _, ax = plt.subplots()
    ax.plot(x, y)
    ax.add_artist(plt.Circle((0, 0), inv.rb, fill=False))
    ax.set_aspect("equal")
    plt.show()


def fov_plot(inv, x0, y0):
    """Make a plot by shooting rays in many directions from a single point

    Basically plots the field-of-view from a point obstructed by the involute
    curve. Checks the distance calculation for many angles
    """
    t = np.linspace(0, 6*np.pi, 100000)

    fig, ax = plt.subplots()

    # Plot the involute
    ax.plot(*inv.xy(t), label="Involute")
    ax.plot(x0, y0, "o", label="Point", color="green")
    ax.add_artist(plt.Circle((0, 0), inv.rb, fill=False))
    ax.set_aspect("equal")
    ax.grid(True)

    # Shoot rays from the point
    theta = np.linspace(0, 2*np.pi, 5000)
    dist = np.array( [inv.distance(x0, y0, t) for t in theta] )

    # Plot the hits
    ax.plot(x0 + dist*np.cos(theta), y0 + dist*np.sin(theta), ':', label="Distance", color="m", markersize=1.0)

    plt.show()


def phase_plot(inv, x0, y0, theta):
    """
    """

    ### Plot the involute in the normal space
    t = np.linspace(0, 5*np.pi, 1000)

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].plot(*inv.xy(t), label="involute")
    ax[0].add_artist(plt.Circle((0, 0), inv.rb, fill=False))

    # Plot the test point and base line
    ax[0].plot(x0, y0, 'o', label="point", color='g')
    rl, theta_l, d0, _ = base_line(x0, y0, theta)

    try:
        d = inv.distance(x0, y0, theta)
        print(f"Distance: {d}")
        ax[0].arrow(x0, y0, np.cos(theta)*d, np.sin(theta)*d, width=0.01, color='m', head_length=0.1, head_width=0.1, length_includes_head=True)
    except:
      pass

    ax[0].set_aspect('equal')
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(True)

    # Plot the phase
    d = np.linspace(-10, 10, 1000)
    ax[1].plot(d, inv._phase(d, rl, theta_l), label="phase")
    ax[1].plot(d0, inv._phase(d0, rl, theta_l), 'o', label="phase at point", color='g')

    ax[1].grid(True)

    plt.show()


if __name__ == "__main__":
    #inv = Involute(1, 0.0)
    #plot_involute(inv, 6*np.pi)

    # for i in range(1000):
    #   rb = random.uniform(1.0, 2.0)
    #   a0 = random.uniform(-np.pi, np.pi)
    #   inv = Involute(rb, a0)

    #   x0 = random.uniform(-10.0, 10.0)
    #   y0 = random.uniform(-10.0, 10.0)
    #   if x0**2 + y0**2 < rb**2:
    #     continue
    #   try:
    #     fov_plot(inv, x0, y0)
    #   except RuntimeError as e:
    #     print(f"{e}")
    #   print(f"{i}")

    # inv = Involute(1.0, -0.5*np.pi)
    # phase_plot(inv, -0.37, -0.999, 0.5*np.pi)
    # # fov_plot(inv, -0.37, -0.999)

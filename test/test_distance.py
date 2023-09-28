import pytest
import involute
import numpy as np




Cases = [
    (1.0, 0.0, -1.0, 0.0, 0.0, 1.0),                         # Origin @ maximum
    (2.0, 0.0, -2.0, 2.0, 0.0, 5.141592653589793),           # Tangent @ maximum
    (2.0, 0.0, -2.0, 2.0, -0.15*np.pi, 4.405378529170532),   # Complex Gap hit
    (2.0 ,0.0, 0.0, 2.0, -0.5*np.pi, 2.0),                   # Origin from top
    (1.0, -np.pi, 0.1, 2.0, -0.5*np.pi, 5.005326125763902),  # Opposite involute
    (1.0, -np.pi, 0.0, 2.0, -0.5*np.pi, 4.971693870713802),  # Opposite involute origin from top (could also be 2)
    (1.0, np.pi, 0.1, 2.0, -0.5*np.pi, 5.005326125763902),   # Opposite involute (+ve phase)
    (1.0, np.pi, 0.0, 2.0, -0.5*np.pi, 4.971693870713802),   # Opposite involute origin from top (+ve phase) (could also be 2)
    (1.0, np.pi, 4.0, 2.0, -0.87*np.pi, 5.035907396620697),  # Hit in the complex gap
    (1.0, np.pi, 4.0, 2.0, -0.85*np.pi, 6.242127085088714),  # Hit across the complex gap
    (1.0720802059149177, -1.4511564629610503, -0.3770951755705738, -1.2108722180281113, 1.8325433442424157, 6.261406035145226), # Unstable with Newton iteration only (close to the maximum)
    (1.9454685012010147, -1.0436493497633244, 1.62573767830043, -1.330988362095857, 3.049211353314198, 0.9033408415670273), # Hit in complex gap, with start point on RHS of maximum
    (6.088851976614883, 0.5952473331434209, 5.771226568065575, 5.374558315266857, 4.380256210346241, 2.0033650465825135) # Was found to give -ve distance if complex gap guess is always pushed to the right
]

@pytest.mark.parametrize("rb, a0, x, y, theta, d", Cases)
def test_distance(rb, a0, x, y, theta, d):
    inv = involute.Involute(rb, a0)
    assert inv.distance(x, y, theta) == pytest.approx(d)


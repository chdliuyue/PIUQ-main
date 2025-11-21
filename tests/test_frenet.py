import numpy as np

from piuq.data.geometry.frenet import FrenetFrame


def test_straight_projection_velocity():
    cl = np.array([[0.0, 0.0], [10.0, 0.0]])
    frenet = FrenetFrame(cl)
    xy = np.array([[2.0, 1.0], [5.0, -0.5]])
    v_xy = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = frenet.to_frenet(xy, v_xy=v_xy)
    assert np.allclose(res["s"], [2.0, 5.0])
    assert np.allclose(res["n"], [1.0, -0.5])
    assert np.allclose(res["v_s"], [1.0, 0.0])
    assert np.allclose(res["v_n"], [0.0, 1.0])

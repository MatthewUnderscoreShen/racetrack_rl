import pytest
import matplotlib.pyplot as plt
import numpy as np

from tracks.bezier_curve import BezierCurve

def test_Bezier():
    # test all methods in bezier_curve
    #   get_bezier, get_bezier_coeff, get_p2, check_end_distance, check_end_line

    # Arrange
    p0 = [0, 0, 90]                     # generate bezier curve
    p1 = [2, 1, 0]
    r = 0.25
    curve = BezierCurve(p0, p1, r)

    t = np.linspace(0, 1, 101)          # t array
    xy = curve.get_bezier(t)
    xy2 = np.zeros((len(t), 2))

    coeff = curve.get_bezier_coeff()    # bezier coeffs

    d_test_pts = ((0,0), (2,1), (-1000,1000), (1.999,0.999))


    # Act
    for i in range(len(t)):
        xy2[i,:] = curve.get_bezier(t[i])
    coeff_ans = ((2,-1), (0,2), (0,0))

    d_test_ans = (True, False, False, True)

    
    # Assert
    assert pytest.approx(curve.get_p2()) == [0,1]   # in between point (__init__ test)
    assert np.all(np.equal(xy, xy2))                # single t vs array t (get_bezier test)
    for q, a in zip(coeff, coeff_ans):              # d/dt coefficients (get_bezier_coeff test)
        assert pytest.approx(q) == a
    for q, a in zip(d_test_pts, d_test_ans):
        assert curve.check_end_distance(q) == a     # (check_end_distance test)
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

    d_test_pts = (                      # distance to end point (check radius r)
        (0,0),                          # start point (fail)
        (2,1),                          # end point (pass)
        (-1000,1000),                   # really far (fail)
        (1.999,0.999)                   # really close (pass)
    )    

    line_test_pts = (                   # line checks: - (before), 0 (on), + (after)
        (0, 0),                         # start point (-)
        (1.9, 1),                       # just before (-)
        (2.1, 1),                       # just after (+)
        (1.9, 0.5),                     # just before diagonal (-)
        (2.1, 1.5),                     # just after diagonal (+)
        (2, 1),                         # end point (0)
        (2, 1.5),                       # end line diagonal (0)
        (-1000, -1000),                 # really far before (-)
        (1000, 1000),                   # really far after (+)
        (2, -1000)                      # really far on (0)
    )


    # Act
    for i in range(len(t)):
        xy2[i,:] = curve.get_bezier(t[i])
    coeff_ans = ((2,-1), (0,2), (0,0))

    d_test_ans = (False, True, False, True)
    line_test_ans = (-1, -1, 1, -1, 1, 0, 0, -1, 1, 0)

    
    # Assert
    assert pytest.approx(curve.get_p2()) == [0,1]   # in between point (__init__ test)
    assert np.all(np.equal(xy, xy2))                # single t vs array t (get_bezier test)
    for q, a in zip(coeff, coeff_ans):              # d/dt coefficients (get_bezier_coeff test)
        assert pytest.approx(q) == a
    for q, a in zip(d_test_pts, d_test_ans):
        assert curve.check_end_distance(q) == a     # (check_end_distance test)
    for q, a in zip(line_test_pts, line_test_ans):
        assert np.sign(curve.check_end_line(q)) == a # (check_end_line test)
import pytest
import matplotlib.pyplot as plt
import numpy as np

from tracks.bezier_curve import BezierCurve

def assert_output(input, output):
    if output["sect"] == "raises":
        with pytest.raises(output["exception"]):
            input()
        return

    result = input()
    np.testing.assert_allclose(
        result,
        output["value"],
        rtol = output.get("rtol", 1e-6),
        atol = output.get("atol", 0.0)
    )

@pytest.fixture
def curve_45deg():
    return BezierCurve(p0=(0,0,0), p1=(1,1,np.pi/2), r=0.2)

@pytest.fixture
def curve_135deg():
    return BezierCurve(p0=(0,0,0), p1=(1,1,3*np.pi/2), r=0.2)

@pytest.mark.parametrize("fixture_name", )
@pytest.mark.parametrize("case", )
def test_get_bezier():
    curve = 
    assert_output(lambda: )
import pytest
import matplotlib.pyplot as plt
import numpy as np
import yaml

from tracks.bezier_curve import BezierCurve

yaml_path = "helpme.yaml"
with open(yaml_path, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.Fullloader)
fixtures = yaml_data["fixtures"]
tests = yaml_data["tests"]


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

@pytest.mark.parametrize("name", fixtures.keys())
@pytest.mark.parametrize("case", tests["get_bezier"].items())
def test_get_bezier(name, case, request):
    curve = request.getfixturevalue(name)
    assert_output(lambda: curve.get_bezier(case["inputs"]), case["outputs"])
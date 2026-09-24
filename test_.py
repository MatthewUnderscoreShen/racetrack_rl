import pytest
import matplotlib.pyplot as plt
import numpy as np
import yaml

from tracks.bezier_curve import BezierCurve

yaml_path = "testing/helpme.yaml"
with open(yaml_path, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
fixtures = yaml_data["fixtures"]
tests = yaml_data["tests"]

EXCEPTIONS = {
    "ValueError": ValueError
}

def assert_output(input, output):
    if output.get("exception", None) != None:
        with pytest.raises(EXCEPTIONS[output["exception"]]):
            input()
        return

    result = input()
    print(result)
    print(output["value"])
    np.testing.assert_allclose(
        result,
        output["value"],
        rtol = output.get("rtol", 1e-3),
        atol = output.get("atol", 0.0)
    )


@pytest.fixture
def curve_45deg():
    tf = fixtures["curve_45deg"]
    return BezierCurve(p0=tf["p0"], p1=tf["p1"], r=tf["r"])


@pytest.fixture
def curve_135deg():
    tf = fixtures["curve_135deg"]
    return BezierCurve(p0=tf["p0"], p1=tf["p1"], r=tf["r"])

tt = tests["get_bezier"]
@pytest.mark.parametrize("name", fixtures.keys(), ids=list(fixtures.keys()))
@pytest.mark.parametrize("case, data", tt.items(), ids=list(tt.keys()))
def test_get_bezier(name, case, data, request):
    curve = request.getfixturevalue(name)
    assert_output(lambda: curve.get_bezier(data["inputs"]), data["outputs"].get(name))
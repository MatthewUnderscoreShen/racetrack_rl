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


# repeat this for every function to test
tt = tests["get_bezier"]
@pytest.mark.parametrize("name, params", fixtures.items(), ids=list(fixtures.keys()))
@pytest.mark.parametrize("data", tt.values(), ids=list(tt.keys()))
def test_get_bezier(name, params, data):
    curve = BezierCurve(**params)
    assert_output(lambda: curve.get_bezier(*data["inputs"]), data["outputs"].get(name))
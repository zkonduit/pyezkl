from torch import nn
from ezkl import export
import os

folder_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '.',
    )
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x):
        return self.layer(x)

def test_export():
    """
    Test for export with 1l_average.onnx
    """

    circuit = Model()
    network_onnx_path = os.path.join(folder_path, "network.onnx")
    input_filename = os.path.join(folder_path, "input.json")
    export(circuit, input_shape = [3], onnx_filename=network_onnx_path, input_filename=input_filename)

    assert os.path.isfile(network_onnx_path)
    assert os.path.isfile(input_filename)
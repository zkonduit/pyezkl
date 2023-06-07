# here we create and (potentially train a model)

from torch import nn
import ezkl
import os
import json 


# Defines the model
# we got convs, we got relu, we got linear layers 
# What else could one want ???? 

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)
        
        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)
       
        return logits



def test_export():
    """
    Test prove with 4l_conv
    """
    
    circuit = MyModel()

    # Train the model as you like here (skipped for brevity)

    # After training, export to onnx (network.onnx) and create a data file (input.json)
    ezkl.export(circuit, input_shape = [1,28,28])

    # generate a srs for testing
    params_path = os.path.join('kzg.params')
    res = ezkl.gen_srs(params_path, 17)

    data_path = os.path.join('input.json')
    model_path = os.path.join('network.onnx')
    pk_path = os.path.join('test.pk')
    vk_path = os.path.join('test.vk')
    circuit_params_path = os.path.join('circuit.params')
    params_path = os.path.join('kzg.params')

    res = ezkl.setup(
        model_path,
        vk_path,
        pk_path,
        params_path,
        circuit_params_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(circuit_params_path)

    # GENERATE A PROOF 
    proof_path = os.path.join('test.pf')

    res = ezkl.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        params_path,
        "poseidon",
        "single",
        circuit_params_path
    )

    assert res == True
    assert os.path.isfile(proof_path)

    # VERIFY IT 

    res = ezkl.verify(
        proof_path,
        circuit_params_path,
        vk_path,
        params_path,
    )

    assert res == True
    print("verified")


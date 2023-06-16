import io
import numpy as np
import torch.onnx
import torch
import json
import ezkl_lib

def export(
    torch_model,
    input_shape=None,
    input_array=None,
    onnx_filename="network.onnx",
    input_filename="input.json",
    settings_filename="settings.json",
    run_forward=True,
    run_calibrate_settings=True,
    calibration_target="resources",
    scale=7,
    batch_size=1,
):
    """Export a PyTorch model.
    Arguments:
    torch_model: a PyTorch model class, such as Network(torch.nn.Module)
    Optional Keyword Arguments:
    - input_shape: e.g. [3,2,3], a random input with these dimensions will be generated.
    - input_array: the given input will be used for the model
    Note: Exactly one of input_shape and input_array should be specified.
    - onnx_filename: Default "network.onnx", the name of the onnx file to be generated
    - input_filename: Default "input.json", the name of the json input file to be generated for ezkl
    - settings_filename: Default "settings.json", the name of the settings file name generated in the calibration step
    """
    if input_array is None:
        x = 0.1*torch.rand(1,*input_shape, requires_grad=True)
    else:
        x = torch.tensor(input_array)
        if input_shape is not None:
            assert tuple(input_shape) == x.shape
        new_shape = tuple([1]+list(x.shape))
        x = torch.reshape(x,new_shape)


    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                   # model input (or a tuple for multiple inputs)
                      onnx_filename,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes = [input_shape],
                input_data = [data_array],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( input_filename, 'w' ) )

    # Runs a forward operation to quantize inputs
    if run_forward:
        # Uses existing settings file for forward if it exists
        if os.path.isfile(settings_filename):
            ezkl_lib.forward(
                data=input_filename,
                model=onnx_filename,
                output=input_filename,
                scale=scale,
                batch_size=batch_size
                settings_path=settings_filename
            )
        else:
            ezkl_lib.forward(
                data=input_filename,
                model=onnx_filename,
                output=input_filename,
                scale=scale,
                batch_size=batch_size
            )

    # calibrates the setup
    if run_calibrate_settings:
        ezkl_lib.gen_settings(onnx_filename, settings_filename)
        ezkl_lib.calibrate_settings(
            input_filename, onnx_filename, settings_filename, calibration_target)


if __name__ == "__main__":
    from torch import nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.flatten = nn.Flatten()

        def forward(self, x):
            return self.flatten(x)

    circuit = MyModel()

    export(circuit, input_shape = [9,2], input_array= [[0.06980138272047043, 0.09594511240720749], [0.09435884654521942, 0.08530371636152267], [0.059045590460300446, 0.06503964960575104], [0.005533689167350531, 0.08425236493349075], [0.08761995285749435, 0.012599778361618519], [0.014501023106276989, 0.0011647045612335205], [0.007388335652649403, 0.09860159456729889], [0.0525859072804451, 0.0077858567237854], [0.09905613213777542, 0.04969405755400658]])





The `tutorial.py` file creates a (relatively) complex Onnx graph that takes in 3 inputs `x`, `y`, and `z` and produces two outputs that we can verify against public inputs.

The file defines a computational graph as a pytorch `nn.Module` which is as follows:

```python
class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(3, 3, (2, 2), 1, 2)

        self._initialize_weights()

    def forward(self, x, y, z):
        x =  self.sigmoid(self.conv(y@x**2 + (x) - (self.relu(z)))) + 2
        return (x, self.relu(z) / 3)


    def _initialize_weights(self):
        init.orthogonal_(self.conv.weight)
```

As noted above this graph takes in 3 inputs and produces 2 outputs. The main function instantiates an instance of `Circuit` and saves it to an Onnx file.

```python
def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 2]
    x = 0.1*torch.rand(1,*shape, requires_grad=True)
    y = 0.1*torch.rand(1,*shape, requires_grad=True)
    z = 0.1*torch.rand(1,*shape, requires_grad=True)
    torch_out = torch_model(x, y, z)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x,y,z),                   # model input (or a tuple for multiple inputs)
                      "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()
    dy = ((y).detach().numpy()).reshape([-1]).tolist()
    dz = ((z).detach().numpy()).reshape([-1]).tolist()


    data = dict(input_shapes = [shape, shape, shape],
                input_data = [d, dy, dz],
                public_inputs = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( "input.json", 'w' ) )

if __name__ == "__main__":
    main()
```
Running the file generate an `.onnx` file. Note that this also create the required input json file, whereby we use the outputs of the pytorch model as the public inputs to the circuit.

You can use these as inputs to the [ezkl](https://github.com/zkonduit/ezkl) cli as follows: 

```bash
cargo run --bin ezkl -- --scale 4 --bits 16 -K 17 table  -M ./network.onnx
```

you should see the following table being displayed. This is a tabular representation of the Onnx graph, with some additional information required for circuit construction (like the number of advices to use, the fixed point representation denominator at the operation's input and output). You should see all the operations we created in `Circuit(nn.Module)` represented. Nodes 14 and 17 correspond to the output nodes here.

```bash
| node           | output_max | min_cols | in_scale | out_scale | is_output | const_value | inputs     | in_dims   | out_dims     | idx  | Bucket |
| -------------- | ---------- | -------- | -------- | --------- | --------- | ----------- | ---------- | --------- | ------------ | ---- | ------ |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 0    | 0      |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 1    | 0      |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 2    | 0      |
| conv.weight    | 5          | 1        | 4        | 4         | false     | [4...]      |            |           | [3, 3, 2, 2] | 3    |        |
| conv.bias      | 1024       | 1        | 4        | 12        | false     | [-1024...]  |            |           | [3]          | 4    |        |
| power.exp      | 32         | 1        | 4        | 4         | false     | [32...]     |            |           | [1]          | 5    |        |
| Add            | 262144     | 31       | 8        | 8         | false     |             | [7, 0]     | [3, 2, 2] | [3, 2, 2]    | 8    | 0      |
| Relu           | 256        | 12       | 4        | 4         | false     |             | [2]        | [3, 2, 2] | [3, 2, 2]    | 9    | 1      |
| Sub            | 524288     | 31       | 8        | 8         | false     |             | [8, 9]     | [3, 2, 2] | [3, 2, 2]    | 10   | 1      |
| ConvHir        | 10485760   | 67       | 8        | 12        | false     |             | [10, 3, 4] | [3, 2, 2] | [3, 5, 5]    | 11   | 1      |
| Sigmoid        | 16         | 75       | 12       | 4         | false     |             | [11]       | [3, 5, 5] | [3, 5, 5]    | 12   | 2      |
| add.const      | 32         | 1        | 4        | 4         | false     | [32...]     |            |           | [1]          | 13   |        |
| Add            | 64         | 92       | 4        | 4         | true      |             | [12, 13]   | [3, 5, 5] | [3, 5, 5]    | 14   | 2      |
| Relu           | 256        | 12       | 4        | 4         | false     |             | [2]        | [3, 2, 2] | [3, 2, 2]    | 15   | 1      |
| div.const      | 48         | 1        | 4        | 4         | false     | [48...]     |            |           | [1]          | 16   |        |
| Div            | 85.333336  | 12       | 4        | 4         | true      |             | [15]       | [3, 2, 2] | [3, 2, 2]    | 17   | 2      |
```

From there we can run proofs on the generated files, but note that because of quantization errors the public inputs may need to be tweaked to match the output of the circuit and generate a valid proof. You can also express a tolerance to such errors using the `tolerance` flag (which we use below). The types of claims we can make with the setup of this tutorial are ones such as: "I ran my private model on data and produced the expected outputs (as dictated by the public inputs to the circuit)".

``` bash
 RUST_LOG=debug cargo run --bin ezkl -- --tolerance 2 --scale 4 --bits 16 -K 17 mock  -D ./input.json -M ./network.onnx
```


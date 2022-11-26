# ezkl_python
A library for using and testing ezkl from Python. The main purpose of this repository is to create simple tools for generating `.onnx` and `.json` input files that can be ingested 


```
pyezkl/
├── src/
|   └── ezkl/ (pending: python bindings for calling ezkl)
└── examples/
    └── tutorial/ (a tutorial for generating ezkl .onnx and .json inputs)
    └── onnx/ (sample onnx files)
```


If you want to add a model to `examples/onnx`, open a PR creating a new folder within `examples` with a descriptive model name. This folder should contain: 
- a `.json` input file, with the fields expected by the  [ezkl](https://github.com/zkonduit/ezkl) cli. 
- a `.onnx` file representing the trained model 
- a `.py` file for generating the `.json` and `.onnx` files following the general structure of `examples/tutorial/tutorial.py`.


TODO: add associated python files in the onnx model directories. 

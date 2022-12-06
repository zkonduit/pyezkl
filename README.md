# ezkl_python
A library for using and testing ezkl from Python. The main purpose of this repository is to provide Python bindings for [ezkl](https://github.com/zkonduit/ezkl) and to provide simple tools for generating `.onnx` and `.json` input files that can be ingested by it. 


```
pyezkl/
├── ezkl/ (pending: python bindings for calling ezkl)
| 
└── examples/
    └── tutorial/ (a tutorial for generating ezkl .onnx and .json inputs)
```

For samples of onnx files generated using python see [this](https://github.com/zkonduit/onnx-examples) repo. 

## setup 


To get started install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your system. From there create an new evironment:

```bash
conda create -n ezkl python=3.9
```

Activate your newly created environment and install the requisite dependencies:

```bash
conda activate ezkl; pip install torch numpy;       
```

You can install pyezkl using 

```bash
pip install . 
```

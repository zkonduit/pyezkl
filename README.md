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

## development setup

### poetry

Install poetry. [For more instructions check view the poetry documentation](https://python-poetry.org/docs/)

Clone the ezkl repository.
```shell
git clone https://github.com/zkonduit/ezkl.git
```

[Build the python wheels locally by following the instructions on the docs](https://docs.ezkl.xyz/python_bindings/), you will generally want to run the optimized development build command with.
```shell
maturin develop --release --features python-bindings
```

Activate the virtual environment in poetry, and install the developmental version of ezkl.
```shell
poetry install      # this should fail but would initialize a virtual env for you
poetry shell
cd target/wheels
pip install ezkl_lib-version-pythontype-abi3-osversion-processor.whl
```

Once `ezkl_lib` is installed you can install dependencies properly. After which, you'll obtain a working setup of pyezkl locally.
```shell
cd navigate-back-to-pyezkl
poetry install

python
>>> import ezkl
>>> ezkl.export(...)
```

Build and publish the repository by calling
```
poetry build
poetry publish
```

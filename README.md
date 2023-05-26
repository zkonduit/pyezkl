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

Clone the ezkl repository.
```shell
git clone https://github.com/zkonduit/ezkl.git
```

Create a virtual environemnt locally we will use venv within ezkl instead to prevent environment conflicts with pyezkl. Install the python dependencies. Then, build using maturin. [More details can be found within the ezkl documentation](https://docs.ezkl.xyz/python_bindings/)

```shell
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin build --release --features python-bindings
```

The build will take a while, after which you should find wheel (.whl) file within `target/wheels`. The example below is what you should be seeing, if you had built the package within linux.

```shell
ls target/wheels

ezkl_lib-0.1.0-cp37-abi3-manylinux_2_35_x86_64.whl
```

Clone the pyezkl repository in a separate directory if you haven't done so already.

```shell
git clone https://github.com/zkonduit/pyezkl.git
```

We will use poetry for the pyezkl repository. To proceed install poetry. [For more instructions check view the poetry documentation](https://python-poetry.org/docs/)


Deactivate the existing venv virtual environment. Activate the virtual environment in poetry by installing the dependencies for pyezkl as follows
```shell
deactivate
cd pyezkl
poetry install      # this should fail to install ezkl_lib but would initialize a virtual env for you
poetry shell        # activate the virtual environment for poetry
cd ../ezkl/target/wheels
pip install ezkl_lib-version-pythontype-abi3-osversion-processor.whl
```

Once `ezkl_lib` is installed you can build the wheel and install it. After which, you'll obtain a working setup of pyezkl locally.
```shell
cd navigate-back-to-pyezkl
poetry build
pip install ./dist/ezkl-version-py3-none-any.whl

python
>>> import ezkl
>>> ezkl.export(...)
```

After building the wheel you should be able to publish by calling the following function.
```
poetry publish
```

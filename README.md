# ezkl
`pip install ezkl` lets you use [ezkl](https://github.com/zkonduit/ezkl) directly from Python. It also contains an `export` function to generate `.onnx` and `.json` input files that can be ingested by the `ezkl` cli or from Python. [Here is colab notebook](https://colab.research.google.com/drive/1XuXNKqH7axOelZXyU3gpoTOCvFetIsKu?usp=sharing) that shows how to produce and verify a proof from Python.

Check out [the docs](https://docs.ezkl.xyz) for more. If you want to *develop* the Python bindings, read on.

## development setup

### poetry

Use this following setup if you would like to access developmental features in the main ezkl repo that is not released yet.

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
poetry install
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

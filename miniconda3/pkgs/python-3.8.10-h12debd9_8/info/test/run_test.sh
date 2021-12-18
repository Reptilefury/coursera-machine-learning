

set -ex



python -V
python3 -V
2to3 -h
pydoc -h
python3-config --help
python -m venv test-venv
python -c "import sysconfig; print(sysconfig.get_config_var('CC'))"
for f in ${CONDA_PREFIX}/lib/python*/_sysconfig*.py; do echo "Checking $f:"; if [[ `rg @ $f` ]]; then echo "FAILED ON $f"; cat $f; exit 1; fi; done
pushd tests
pushd distutils
python setup.py install -v -v
python -c "import foobar"
popd
pushd distutils.cext
python setup.py install -v -v
python -v -v -v -c "import greet"
python -v -v -v -c "import greet; greet.greet('Python user')" | rg "Hello Python"
popd
pushd embedding-interpreter
bash build-and-test.sh
popd
pushd cmake
bash run_cmake_test.sh 3.8.10
popd
pushd processpoolexecutor-max_workers-61
python ppe.py
popd
popd
exit 0

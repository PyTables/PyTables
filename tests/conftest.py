import pytest


@pytest.fixture(scope="function")
def h5path(tmp_path, request):
    return tmp_path / f"{request.cls.__name__}_{request.node.name}.h5"

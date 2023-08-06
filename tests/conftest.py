import pytest


def pytest_addoption(parser):
    parser.addoption('--dataroot', action='store', required=True)
    parser.addoption('--outputroot', action='store')


@pytest.fixture(scope='session')
def dataroot(request):
    dataroot_value = request.config.option.dataroot
    if dataroot_value is None:
        pytest.skip()
    return dataroot_value


@pytest.fixture(scope='session')
def outputroot(request, tmp_path_factory):
    outputroot_value = request.config.option.outputroot
    if outputroot_value is None:
        return tmp_path_factory.mktemp('koogu')
    return outputroot_value

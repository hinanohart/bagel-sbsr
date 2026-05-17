import pytest


@pytest.mark.smoke
def test_package_imports():
    import bagel_sbsr

    assert bagel_sbsr.__version__ == "0.1.0.dev0"


@pytest.mark.smoke
def test_python_version():
    import sys

    assert sys.version_info >= (3, 10)

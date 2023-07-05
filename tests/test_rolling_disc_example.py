import os
import subprocess

import pytest

examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.fixture(scope="session")
def cleanup(request):
    def remove_data_files():
        files_to_remove = [
            os.path.join(os.getcwd(), "data.pkl"),
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    request.addfinalizer(remove_data_files)


@pytest.mark.dependency()
def test_rolling_disc_model(cleanup):
    file = os.path.join(examples_dir, "rolling_disc", "model.py")
    subprocess.call(fr"python {file}")


@pytest.mark.dependency(depends=["test_rolling_disc_model"])
def test_rolling_disc_optimize(cleanup):
    file = os.path.join(examples_dir, "rolling_disc", "optimize.py")
    subprocess.call(fr"python {file}")


@pytest.mark.dependency(depends=["test_rolling_disc_optimize"])
def test_rolling_disc_simulate(cleanup):
    file = os.path.join(examples_dir, "rolling_disc", "simulate.py")
    subprocess.call(fr"python {file}")

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
    subprocess.call(fr"python {examples_dir}\rolling_disc\model.py")


@pytest.mark.dependency(depends=["test_rolling_disc_model"])
def test_rolling_disc_optimize(cleanup):
    subprocess.call(fr"python {examples_dir}\rolling_disc\optimize.py")


@pytest.mark.dependency(depends=["test_rolling_disc_optimize"])
def test_rolling_disc_simulate(cleanup):
    subprocess.call(fr"python {examples_dir}\rolling_disc\simulate.py")

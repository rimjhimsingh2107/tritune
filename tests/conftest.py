import pytest

from kernelpilot.backends import CachedBackend, WorkerBackend
from kernelpilot.backends.isolation import InProcessExecutor
from kernelpilot.cache import ResultCache
from kernelpilot.config import load_settings
from kernelpilot.env import detect
from kernelpilot.store import Store


@pytest.fixture(scope="session")
def env():
    return detect()


@pytest.fixture
def settings(tmp_path):
    return load_settings(backend="sim", home=tmp_path, db_path=tmp_path / "kp.db", reps=20, warmup=2)


@pytest.fixture
def backend(env):
    return CachedBackend(WorkerBackend("sim", InProcessExecutor(), env), ResultCache(":memory:"))


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()

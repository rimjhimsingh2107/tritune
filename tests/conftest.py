import pytest

from tritune.backends import CachedBackend, WorkerBackend
from tritune.backends.isolation import InProcessExecutor
from tritune.cache import ResultCache
from tritune.config import load_settings
from tritune.env import detect
from tritune.store import Store


@pytest.fixture(scope="session")
def env():
    return detect()


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        backend="sim", home=tmp_path, db_path=tmp_path / "tt.db", reps=20, warmup=2
    )


@pytest.fixture
def backend(env):
    return CachedBackend(WorkerBackend("sim", InProcessExecutor(), env), ResultCache(":memory:"))


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()

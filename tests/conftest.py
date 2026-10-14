import os

# No test may open a window: pygame is only ever driven through SDL's dummy video driver here.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


import pytest


@pytest.fixture(autouse=True)
def _fresh_contact_gate():
    """Each test starts with an input device that has not reported a touch yet."""
    from experiment import trial_render

    trial_render.reset_contact_gate()
    yield

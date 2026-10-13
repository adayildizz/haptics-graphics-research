import os

# No test may open a window: pygame is only ever driven through SDL's dummy video driver here.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

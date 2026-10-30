"""Cross-keyboard participant controls for the live experiment."""

from __future__ import annotations

from typing import Any, Literal

CONTINUE_LABEL = "SPACE / ENTER"
EXIT_LABEL = "ESC"
LEFT_LABEL = "LEFT ARROW / NUMPAD 4"
RIGHT_LABEL = "RIGHT ARROW / NUMPAD 6"


def is_continue_key(key: int, pygame_module: Any) -> bool:
    """Space and the main keyboard's Enter. Not numpad Enter.

    Numpad Enter sits directly below the response keys -- on most keyboards
    it is the tall key under 6 -- so a participant reaching for a response
    by touch can hit it. It does nothing during a trial, but it advances
    the break and start screens, which are the supervisor's to advance.
    Between this and Escape-only exit, the numpad now carries exactly two
    live keys: 4 and 6.
    """
    return key in (pygame_module.K_SPACE, pygame_module.K_RETURN)


def is_exit_key(key: int, pygame_module: Any) -> bool:
    """Escape only, on the main keyboard.

    Numpad 0 used to exit as well, which put the abort key directly under
    the hand a participant is already using: the response keys are numpad 4
    and 6, found by touch from the nub on 5, and 0 is the wide key on the
    row below them. A blindfolded participant reaching for a response can
    land on it and end the session. Escape is across the keyboard, out of
    reach of that hand, and is the supervisor's key anyway.
    """
    return key == pygame_module.K_ESCAPE


def is_felt_gap_key(key: int, pygame_module: Any) -> bool:
    """Down arrow or numpad 2: "I felt the signal break", logged during a trial when ``felt_gap_key`` is on.

    Both sit between and below the two response keys, so the hand on the
    arrows (or on numpad 4/6) reaches it without looking. It never answers or
    ends a trial.
    """
    return key in (pygame_module.K_DOWN, pygame_module.K_KP2)


def response_for_key(key: int, pygame_module: Any) -> Literal["left", "right"] | None:
    if key in (pygame_module.K_LEFT, pygame_module.K_KP4):
        return "left"
    if key in (pygame_module.K_RIGHT, pygame_module.K_KP6):
        return "right"
    return None

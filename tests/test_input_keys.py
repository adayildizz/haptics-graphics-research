from __future__ import annotations

import pygame

from experiment.input_keys import is_continue_key, is_exit_key, response_for_key


def test_space_and_main_enter_continue():
    assert is_continue_key(pygame.K_SPACE, pygame)
    assert is_continue_key(pygame.K_RETURN, pygame)


def test_numpad_enter_no_longer_continues():
    """It sits under the numpad response keys, within reach of a blind fumble."""
    assert not is_continue_key(pygame.K_KP_ENTER, pygame)


def test_escape_is_the_only_exit_key():
    assert is_exit_key(pygame.K_ESCAPE, pygame)


def test_numpad_zero_no_longer_exits():
    """It sat right below the numpad response keys, within reach of a blind fumble."""
    assert not is_exit_key(pygame.K_KP0, pygame)


def test_arrow_keys_and_numpad_four_and_six_map_to_left_and_right():
    assert response_for_key(pygame.K_LEFT, pygame) == "left"
    assert response_for_key(pygame.K_KP4, pygame) == "left"
    assert response_for_key(pygame.K_RIGHT, pygame) == "right"
    assert response_for_key(pygame.K_KP6, pygame) == "right"


def test_numpad_carries_only_the_two_response_keys():
    """Everything else on the numpad must be inert for a blindfolded participant."""
    numpad = [
        pygame.K_KP0, pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP5,
        pygame.K_KP7, pygame.K_KP8, pygame.K_KP9, pygame.K_KP_ENTER,
        pygame.K_KP_PLUS, pygame.K_KP_MINUS, pygame.K_KP_PERIOD,
        pygame.K_KP_MULTIPLY, pygame.K_KP_DIVIDE,
    ]
    for key in numpad:
        assert response_for_key(key, pygame) is None
        assert not is_exit_key(key, pygame)
        assert not is_continue_key(key, pygame)

    assert response_for_key(pygame.K_KP4, pygame) == "left"
    assert response_for_key(pygame.K_KP6, pygame) == "right"


def test_down_arrow_and_numpad_two_report_a_felt_gap_and_answer_nothing():
    from experiment.input_keys import is_felt_gap_key

    for key in (pygame.K_DOWN, pygame.K_KP2):
        assert is_felt_gap_key(key, pygame)
        assert response_for_key(key, pygame) is None
        assert not is_continue_key(key, pygame) and not is_exit_key(key, pygame)
    for key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_KP4, pygame.K_KP6, pygame.K_SPACE):
        assert not is_felt_gap_key(key, pygame)

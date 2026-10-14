"""Pygame drawing utilities for the 2AFC bar-height experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from .input_keys import CONTINUE_LABEL, EXIT_LABEL, LEFT_LABEL, RIGHT_LABEL

from .calibration import DisplayCalibration
from .config import SCREEN_HEIGHT, SCREEN_WIDTH, USE_FULLSCREEN

Color = tuple[int, int, int]

BACKGROUND: Color = (18, 20, 24)
PANEL: Color = (32, 36, 44)
PANEL_ACTIVE: Color = (42, 50, 62)
DIVIDER: Color = (110, 116, 128)
BAR_REF: Color = (66, 153, 225)
BAR_COMPARISON: Color = (72, 187, 120)
BAR_OUTLINE: Color = (230, 235, 240)
TEXT: Color = (238, 242, 247)
MUTED: Color = (172, 180, 192)
HAPTIC_BORDER: Color = (245, 166, 35)
FEEDBACK_CORRECT: Color = (72, 187, 120)
FEEDBACK_INCORRECT: Color = (220, 90, 90)
TOUCH_POINT: Color = (255, 0, 0)
TOUCH_GUIDE_BG: Color = (0, 0, 0)
TOUCH_GUIDE_STRIPE: Color = (255, 255, 255)


@dataclass(frozen=True)
class TrialLayout:
    left_bar: pygame.Rect
    right_bar: pygame.Rect
    haptic_area: pygame.Rect
    left_is_comparison: bool
    reference_height_mm: float
    comparison_height_mm: float
    bar_width_mm: float


def active_rect(calibration: DisplayCalibration) -> pygame.Rect:
    return pygame.Rect(
        calibration.active_left_px,
        calibration.active_top_px,
        calibration.active_width_px,
        calibration.active_height_px,
    )


def countdown_seconds(remaining_time_s: float) -> int:
    """Round remaining time up so the counter starts at the full limit."""
    return max(0, math.ceil(remaining_time_s))


def trial_counter_text(is_practice: bool, trial_index: int, total_trials: int | None = None) -> str:
    """"Trial 12/66" when the block size is known, otherwise just "Trial 12"."""
    label = "Practice" if is_practice else "Trial"
    return f"{label} {trial_index}/{total_trials}" if total_trials else f"{label} {trial_index}"


def init_window(
    fullscreen: bool = USE_FULLSCREEN,
    size: tuple[int, int] | None = None,
) -> pygame.Surface:
    pygame.display.set_caption("Height JND 2AFC Experiment")
    if fullscreen:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return pygame.display.set_mode(size or (SCREEN_WIDTH, SCREEN_HEIGHT))


def make_trial_layout(
    screen: pygame.Surface,
    calibration: DisplayCalibration,
    bar_width_mm: float,
    reference_height_mm: float,
    comparison_height_mm: float,
    left_is_comparison: bool,
    inter_bar_gap_mm: float,
) -> TrialLayout:
    width_px = max(1, round(bar_width_mm * calibration.px_per_mm_x))
    gap_px = max(1, round(inter_bar_gap_mm * calibration.px_per_mm_x))
    reference_px = max(1, round(reference_height_mm * calibration.px_per_mm_y))
    comparison_px = max(1, round(comparison_height_mm * calibration.px_per_mm_y))
    haptic_area = active_rect(calibration)
    baseline_y = haptic_area.bottom

    center_x = haptic_area.left + haptic_area.width // 2
    left_center_x = center_x - gap_px // 2 - width_px // 2
    right_center_x = center_x + gap_px // 2 + width_px // 2

    left_height_px = comparison_px if left_is_comparison else reference_px
    right_height_px = reference_px if left_is_comparison else comparison_px
    left_bar = pygame.Rect(0, 0, width_px, left_height_px)
    right_bar = pygame.Rect(0, 0, width_px, right_height_px)
    left_bar.midbottom = (left_center_x, baseline_y)
    right_bar.midbottom = (right_center_x, baseline_y)
    return TrialLayout(
        left_bar=left_bar,
        right_bar=right_bar,
        haptic_area=haptic_area,
        left_is_comparison=left_is_comparison,
        reference_height_mm=reference_height_mm,
        comparison_height_mm=comparison_height_mm,
        bar_width_mm=bar_width_mm,
    )


def draw_trial(
    screen: pygame.Surface,
    layout: TrialLayout,
    bar_width_mm: float,
    trial_index: int,
    is_practice: bool,
    active_side: str | None = None,
    blind_test_mode: bool = False,
    touch_pos: tuple[int, int] | None = None,
    remaining_time_s: float | None = None,
    total_trials: int | None = None,
) -> None:
    width, height = screen.get_size()

    if blind_test_mode:
        # The touchable columns are shown as full-height white stripes (so
        # finding them isn't a blind fumble), but the height bars themselves
        # are never drawn -- height is the thing being tested, so it must be
        # judged by touch alone, not read off the screen.
        screen.fill(TOUCH_GUIDE_BG)
        for bar in (layout.left_bar, layout.right_bar):
            stripe = pygame.Rect(bar.left, layout.haptic_area.top, bar.width, layout.haptic_area.height)
            pygame.draw.rect(screen, TOUCH_GUIDE_STRIPE, stripe)
    else:
        screen.fill(BACKGROUND)
        left_panel = pygame.Rect(0, 84, width // 2, height - 164)
        right_panel = pygame.Rect(width // 2, 84, width // 2, height - 164)
        pygame.draw.rect(screen, PANEL_ACTIVE if active_side == "left" else PANEL, left_panel)
        pygame.draw.rect(screen, PANEL_ACTIVE if active_side == "right" else PANEL, right_panel)
        pygame.draw.line(screen, DIVIDER, (width // 2, 72), (width // 2, height - 60), 2)

    title_font = pygame.font.SysFont("Arial", 30, bold=True)
    label_font = pygame.font.SysFont("Arial", 24, bold=True)
    body_font = pygame.font.SysFont("Arial", 20)

    title = title_font.render("Which bar is taller?", True, TEXT)
    screen.blit(title, title.get_rect(center=(width // 2, 34)))

    if remaining_time_s is not None:
        seconds_left = countdown_seconds(remaining_time_s)
        timer_color = FEEDBACK_INCORRECT if seconds_left <= 5 else TEXT
        timer = label_font.render(f"TIME {seconds_left:02d}", True, timer_color)
        screen.blit(timer, timer.get_rect(midright=(width - 24, 34)))

    counter = trial_counter_text(is_practice, trial_index, total_trials)
    info = body_font.render(f"Width {bar_width_mm:g} mm    {counter}", True, MUTED)
    screen.blit(info, info.get_rect(center=(width // 2, 68)))

    pygame.draw.rect(screen, HAPTIC_BORDER, layout.haptic_area, 3)

    if not blind_test_mode:
        pygame.draw.rect(screen, BAR_COMPARISON if layout.left_is_comparison else BAR_REF, layout.left_bar)
        pygame.draw.rect(screen, BAR_COMPARISON if not layout.left_is_comparison else BAR_REF, layout.right_bar)
        pygame.draw.rect(screen, BAR_OUTLINE, layout.left_bar, 2)
        pygame.draw.rect(screen, BAR_OUTLINE, layout.right_bar, 2)

    left_label = label_font.render(f"LEFT  ({LEFT_LABEL})", True, TEXT)
    right_label = label_font.render(f"RIGHT  ({RIGHT_LABEL})", True, TEXT)
    screen.blit(left_label, left_label.get_rect(center=(width // 4, height - 105)))
    screen.blit(right_label, right_label.get_rect(center=(width * 3 // 4, height - 105)))

    hint = body_font.render(f"Explore both sides, then answer. {EXIT_LABEL} exits safely.", True, MUTED)
    screen.blit(hint, hint.get_rect(center=(width // 2, height - 34)))

    if touch_pos is not None:
        pygame.draw.circle(screen, TOUCH_POINT, touch_pos, 10)


def draw_feedback(screen: pygame.Surface, correct: bool) -> None:
    width, height = screen.get_size()
    font = pygame.font.SysFont("Arial", 36, bold=True)
    color = FEEDBACK_CORRECT if correct else FEEDBACK_INCORRECT
    text = font.render("Correct" if correct else "Incorrect", True, color)
    screen.blit(text, text.get_rect(center=(width // 2, height // 2)))


def draw_break(screen: pygame.Surface, message: str) -> None:
    screen.fill(BACKGROUND)
    font = pygame.font.SysFont("Arial", 30, bold=True)
    subfont = pygame.font.SysFont("Arial", 22)
    text = font.render(message, True, TEXT)
    hint = subfont.render(f"Press {CONTINUE_LABEL} to continue, or {EXIT_LABEL} to exit.", True, MUTED)
    screen.blit(text, text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 24)))
    screen.blit(hint, hint.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 24)))
    pygame.display.flip()

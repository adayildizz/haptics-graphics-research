"""Pygame renderer shared by the replay browser and future video export."""

from __future__ import annotations

import pygame

from .replay_data import ReplayAttempt, ReplaySession

BACKGROUND = (14, 17, 22)
STAGE_BACKGROUND = (25, 29, 36)
PANEL = (36, 42, 52)
PANEL_ACTIVE = (48, 60, 76)
TEXT = (238, 242, 247)
MUTED = (165, 175, 188)
HAPTIC_BORDER = (245, 166, 35)
BAR_REFERENCE = (66, 153, 225)
BAR_COMPARISON = (72, 187, 120)
CURSOR = (255, 65, 65)
SIGNAL_ON = (72, 220, 120)
SIGNAL_OFF = (130, 138, 150)


def draw_attempt_frame(
    surface: pygame.Surface,
    session: ReplaySession,
    attempt: ReplayAttempt,
    t_us: int,
    viewport: pygame.Rect,
) -> None:
    """Draw a reconstructed analysis view at one replay timestamp."""
    pygame.draw.rect(surface, BACKGROUND, viewport)
    calibration = session.calibration
    original_width = int(calibration.get("screen_width_px", 1280))
    original_height = int(calibration.get("screen_height_px", 720))

    stage_bounds = viewport.inflate(-28, -96)
    stage_bounds.y += 20
    stage_bounds.height -= 20
    scale = min(stage_bounds.width / original_width, stage_bounds.height / original_height)
    stage_width = round(original_width * scale)
    stage_height = round(original_height * scale)
    stage = pygame.Rect(0, 0, stage_width, stage_height)
    stage.center = stage_bounds.center
    pygame.draw.rect(surface, STAGE_BACKGROUND, stage, border_radius=8)

    def sx(x: float) -> int:
        return round(stage.left + x * scale)

    def sy(y: float) -> int:
        return round(stage.top + y * scale)

    active_left = float(calibration.get("active_left_px", 0))
    active_top = float(calibration.get("active_top_px", 0))
    active_width = float(calibration.get("active_width_px", original_width))
    active_height = float(calibration.get("active_height_px", original_height))
    px_per_mm_x = float(calibration.get("px_per_mm_x", 1.0))
    px_per_mm_y = float(calibration.get("px_per_mm_y", 1.0))
    active = pygame.Rect(
        sx(active_left),
        sy(active_top),
        max(1, round(active_width * scale)),
        max(1, round(active_height * scale)),
    )

    center_x = active_left + active_width / 2
    gap_px = float(session.config.get("inter_bar_gap_mm", 40.0)) * px_per_mm_x
    bar_width_px = attempt.bar_width_mm * px_per_mm_x
    left_center_x = center_x - gap_px / 2 - bar_width_px / 2
    right_center_x = center_x + gap_px / 2 + bar_width_px / 2
    reference_height_px = attempt.reference_height_mm * px_per_mm_y
    comparison_height_px = attempt.comparison_height_mm * px_per_mm_y
    left_is_comparison = attempt.reference_side == "right"
    left_height = comparison_height_px if left_is_comparison else reference_height_px
    right_height = reference_height_px if left_is_comparison else comparison_height_px
    baseline_y = active_top + active_height

    left_panel = pygame.Rect(stage.left, stage.top, stage.width // 2, stage.height)
    right_panel = pygame.Rect(stage.centerx, stage.top, stage.width - stage.width // 2, stage.height)
    state = attempt.state_at(t_us)
    pygame.draw.rect(surface, PANEL_ACTIVE if state and state.active_side == "left" else PANEL, left_panel)
    pygame.draw.rect(surface, PANEL_ACTIVE if state and state.active_side == "right" else PANEL, right_panel)
    pygame.draw.line(surface, MUTED, (stage.centerx, stage.top), (stage.centerx, stage.bottom), 1)

    def bar_rect(center: float, height_px: float) -> pygame.Rect:
        return pygame.Rect(
            sx(center - bar_width_px / 2),
            sy(baseline_y - height_px),
            max(1, round(bar_width_px * scale)),
            max(1, round(height_px * scale)),
        )

    left_bar = bar_rect(left_center_x, left_height)
    right_bar = bar_rect(right_center_x, right_height)
    pygame.draw.rect(surface, BAR_COMPARISON if left_is_comparison else BAR_REFERENCE, left_bar)
    pygame.draw.rect(surface, BAR_REFERENCE if left_is_comparison else BAR_COMPARISON, right_bar)
    pygame.draw.rect(surface, TEXT, left_bar, 2)
    pygame.draw.rect(surface, TEXT, right_bar, 2)
    pygame.draw.rect(surface, HAPTIC_BORDER, active, 2)

    # While the finger is lifted the trace repeats the last touched position; no dot, so it does not look like a rest.
    if state is not None and state.contact:
        cursor_pos = (sx(state.x_px), sy(state.y_px))
        pygame.draw.circle(surface, CURSOR, cursor_pos, max(5, round(9 * scale)))

    _draw_overlay(surface, session, attempt, state, t_us, viewport)


def _draw_overlay(surface, session, attempt, state, t_us: int, viewport: pygame.Rect) -> None:
    pygame.font.init()
    title_font = pygame.font.SysFont("Arial", 24, bold=True)
    body_font = pygame.font.SysFont("Arial", 18)
    title = title_font.render(
        f"{attempt.label}  •  Attempt {attempt.attempt_index}  •  {attempt.outcome or 'open'}",
        True,
        TEXT,
    )
    surface.blit(title, (viewport.left + 18, viewport.top + 12))

    duration_s = attempt.duration_us / 1_000_000
    current_s = min(t_us, attempt.duration_us) / 1_000_000
    speed = state.speed_mm_s if state else 0.0
    signal_on = bool(state and state.signal_on)
    lifted = "   Finger lifted" if state is not None and not state.contact else ""
    details = body_font.render(
        f"{session.participant_id}   {current_s:05.2f}/{duration_s:05.2f} s   "
        f"Speed {speed:05.1f} mm/s   Signal {'ON' if signal_on else 'OFF'}{lifted}",
        True,
        SIGNAL_ON if signal_on else SIGNAL_OFF,
    )
    surface.blit(details, (viewport.left + 18, viewport.bottom - 35))

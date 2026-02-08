# Therapy system

This document describes the therapy-related components: session management, task generation, and how they integrate with the rest of MaxSight.

## Purpose

The therapy system supports structured activities (e.g. contrast sensitivity, depth perception, motion tracking) with session tracking, task configuration, and result logging. It is used to run and evaluate therapy tasks that may depend on model outputs (e.g. contrast maps, depth, motion).

## Main modules

- **`ml/therapy/session_manager.py`:** Manages therapy sessions and task attempts.
  - **SessionManager:** Creates sessions (with optional config), logs task attempts (task type, config, result including success/failure and optional metrics like reaction time), and maintains session metrics (total/completed/failed tasks, total time). Can save and load session state to/from JSON.
  - **Session lifecycle:** `start_session()` begins a new session; `log_task_attempt()` records each attempt; `save_session(filepath)` writes the current session to disk.
- **`ml/therapy/task_generator.py`:** Generates or configures therapy tasks (e.g. by type and difficulty). Produces task configs that are passed to the session manager and to whatever runs the task (e.g. contrast task, depth task).
- **`ml/therapy/therapy_integration.py`:** Integrates therapy with the rest of the app: wires model outputs (e.g. from contrast head, depth head, motion head) and user interactions into the session manager and task flow.

## Model heads used by therapy

- **Contrast head:** Provides contrast maps or scores used for contrast sensitivity tasks.
- **Depth head:** Provides depth or distance information for depth perception tasks.
- **Motion head:** Provides motion or flow for motion tracking tasks.
- **Fatigue head / therapy state head:** Can feed into user state (e.g. fatigue) that affects task difficulty or pacing.

These heads are part of MaxSightCNN and are enabled when the tier and condition support them.

## Data flow

1. A therapy session is started via SessionManager.
2. Task generator produces task configs (e.g. task type, parameters).
3. The app or runner executes the task (often using model outputs).
4. Results (success, reaction time, etc.) are logged with `log_task_attempt()`.
5. Session can be saved to JSON for later analysis or resume.

## Configuration and persistence

- Session and task configs are typically dictionaries (e.g. task type, difficulty, time limits). Stored in memory during a run and optionally persisted via `save_session()`.
- Preferences (e.g. verbosity, thresholds) are handled in `ml/utils/user_preferences.py`; therapy code may read these for adaptive behavior.

## Testing

- **`tests/test_therapy.py`:** Covers session manager and therapy integration (session start, task logging, save/load). Run with `pytest tests/test_therapy.py`.
- **`scripts/test_therapy_effectiveness.py`:** Script to evaluate therapy effectiveness (e.g. over many tasks or sessions).

## Summary

The therapy system gives you session-aware, logged therapy tasks that can use model outputs (contrast, depth, motion, etc.). Use SessionManager for lifecycle and logging, task_generator for task configs, and therapy_integration to connect model and UI to sessions. For model architecture that feeds therapy, see `docs/architecture.md`.

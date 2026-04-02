# FluxChi Research Questions

This file lists the questions that currently justify collecting papers.

If a paper does not help answer one of these questions, it probably should not be processed yet.

## Track A: Multimodal State Detection

### A1. Which signal combinations are actually reliable enough for FluxChi?

Current interest:

- EMG + IMU
- vision + physiological signals
- multimodal fatigue detection with personalization

Why it matters:

- intervention policy should not be built on unstable state estimation

### A2. How much personalization is required before state detection becomes useful?

Current interest:

- participant-specific calibration
- baseline adaptation
- ecological validity outside tightly controlled lab settings

Why it matters:

- FluxChi is intended for repeated use by one person over time

### A3. Which state proxies are most actionable for the product?

Candidate states:

- fatigue
- focus degradation
- tension accumulation
- drowsiness
- recovery

Why it matters:

- not every detectable state deserves a product surface or intervention

## Track B: Intervention Strategy

### B1. When should FluxChi stay silent?

Current hypothesis:

- first detected decline should default to `silent_log`

Why it matters:

- reducing unnecessary interruption is core to the product direction

### B2. What evidence should be required before a light nudge?

Current interest:

- sustained decline
- second deterioration
- multimodal agreement

Why it matters:

- a nudge that fires too early becomes noise

### B3. What signals justify escalation?

Current interest:

- severe fatigue or drowsiness markers
- repeated failure to recover
- high-confidence multimodal risk

Why it matters:

- escalation should be rare and defensible

## Track C: Review & Evidence Design

### C1. What should the review board explain after a session?

Current interest:

- largest drop
- recovery windows
- tension-heavy segments
- moments worth testing next

Why it matters:

- review should change future behavior, not just summarize a chart

### C2. How should every intervention become a natural experiment?

Current interest:

- pre/post delta
- whether a suggestion was surfaced
- whether state recovered afterward
- whether the user marked fatigue or alertness

Why it matters:

- FluxChi needs an evidence flywheel, not only heuristics

## Current working product hypotheses

1. `Silent log first, visible nudge later` will feel better than immediate visible reminders.
2. Multimodal agreement will reduce false-positive nudges.
3. Session reviews that highlight drops, recoveries, and tension segments will be more actionable than a single summary sentence.
4. Evidence cards that expose sessions, labels, and labeled samples will make the flywheel legible enough to keep iterating.

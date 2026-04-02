# FluxChi Zotero Schema

## Library

Use a dedicated Zotero library named:

- `FluxChi`

This library is project-scoped for now. Do not mix it with a general long-term literature database yet.

## Collections

- `00 Inbox`
- `01 Multimodal State Detection`
- `02 Intervention Strategy`
- `03 Review & Evidence Design`
- `90 Maybe Later`

### Collection intent

- `00 Inbox`
  - new imports not yet processed
- `01 Multimodal State Detection`
  - papers that help decide which signal combinations reliably track user state
- `02 Intervention Strategy`
  - papers about interruption timing, nudges, escalation, and fatigue-aware action policy
- `03 Review & Evidence Design`
  - papers about review flows, post-session explanation, and natural-experiment evidence capture
- `90 Maybe Later`
  - related, but not currently driving FluxChi decisions

## Tags

Use flat prefixed tags.

### Signal tags

- `sig/emg`
- `sig/vision`
- `sig/imu`
- `sig/rppg`
- `sig/hrv`
- `sig/multimodal`

### State tags

- `state/fatigue`
- `state/focus`
- `state/stress`
- `state/recovery`
- `state/drowsiness`

### Product behavior tags

- `ux/silent-log`
- `ux/light-nudge`
- `ux/escalation`
- `ux/review`
- `ux/summary`

### Method tags

- `method/calibration`
- `method/personalization`
- `method/longitudinal`
- `method/ecological`
- `method/interruption`
- `method/reflection`
- `method/jitai`
- `method/self-experiment`

### Processing tags

- `status/inbox`
- `status/read`
- `status/noted`
- `status/actionable`

## Note template

Notes should use English technical terms with Chinese explanation.

```md
## Research Question
这篇论文在解决什么问题？

## Signals / Modalities
用了哪些信号？
如 EMG / vision / IMU / rPPG / HRV / multimodal

## Setting / Population
实验对象是谁？场景是什么？
如 office work / driving / lab task / daily life

## Metrics
核心指标是什么？
如 fatigue, focus, workload, interruption cost, recovery

## Key Findings
最重要的 3-5 条发现是什么？

## Limitations
这篇论文哪里不够强？
样本量、生态效度、单模态、个体差异、实验设置等

## FluxChi Implication
对 FluxChi 有什么具体启发？
是影响 state detection、intervention strategy，还是 review design？

## Adopt / Reject / Unsure
我们是否采纳这篇论文的思路？为什么？

## Testable Hypothesis
这篇论文在 FluxChi 里可以变成什么可验证假设？
```

## Rules

- Every processed paper belongs to one primary Collection.
- Most papers should have 5 to 8 tags, not more.
- Only papers that can influence a design or experiment decision get `status/actionable`.
- Detailed notes stay in Zotero, not in repo markdown.
- `FluxChi Implication` must be written in product language.
- `Testable Hypothesis` must be expressible with future FluxChi session data.

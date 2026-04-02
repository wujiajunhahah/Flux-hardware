# FluxChi First Batch Papers

This is the first practical intake batch for the `FluxChi` Zotero library.

Use it in this order:

1. add the paper to Zotero
2. place it in the suggested Collection
3. apply the suggested tags
4. fill the full Zotero note template
5. write at least one `FluxChi Implication`
6. write at least one `Testable Hypothesis`

The batch follows the agreed priority:

1. `Multimodal State Detection`
2. `Intervention Strategy`

---

## A. Multimodal State Detection

These are the papers to add first.

### A1. sEMG + IMU, strong direct fit to FluxChi

**Paper**

- [A Multimodal Fatigue Detection System Using sEMG and IMU Signals with a Hybrid CNN-LSTM-Attention Model](https://pubmed.ncbi.nlm.nih.gov/40968807/)

**Why it belongs in the first batch**

- It is the closest direct match to FluxChi’s current hardware direction.
- It explicitly combines `sEMG + IMU`.
- It reports leave-one-subject-out evaluation, which is more useful than a random split for personalization questions.

**Useful details**

- 2025, Sensors
- fatigue induced in 35 healthy participants
- combined IMU-EMG approach reached about `87.94%` accuracy and balanced recall in LOSOCV

**FluxChi implication**

- EMG-only is probably too narrow for a stable fatigue boundary.
- Adding IMU-derived motion quality can improve fatigue classification without needing EEG.
- LOSO-style validation should be part of how FluxChi evaluates future models.

**Suggested Zotero Collection**

- `01 Multimodal State Detection`

**Suggested Tags**

- `sig/emg`
- `sig/imu`
- `sig/multimodal`
- `state/fatigue`
- `method/personalization`
- `status/actionable`

---

### A2. Forearm EMG + IMU in a realistic worker setting

**Paper**

- [Automated and Continuous Fatigue Monitoring in Construction Workers Using Forearm EMG and IMU Wearable Sensors and Recurrent Neural Network](https://pubmed.ncbi.nlm.nih.gov/36560096/)

**Why it belongs in the first batch**

- It uses `forearm EMG + IMU`, which is much closer to FluxChi than leg-mounted or lab-only setups.
- It is explicitly about continuous fatigue monitoring rather than one-off classification.
- It is one of the better bridges between academic sensing and a wearable work product.

**Useful details**

- 2022, Sensors
- construction-worker fatigue monitoring
- designed around forearm wearable sensing

**FluxChi implication**

- FluxChi should keep treating fatigue as a continuous state over time, not a one-frame classification problem.
- Wrist / forearm wearables are not a weak proxy by default if the modeling is longitudinal.
- Real-world work context matters more than benchmark accuracy alone.

**Suggested Zotero Collection**

- `01 Multimodal State Detection`

**Suggested Tags**

- `sig/emg`
- `sig/imu`
- `sig/multimodal`
- `state/fatigue`
- `method/longitudinal`
- `method/ecological`
- `status/actionable`

---

### A3. Sensor fusion can approximate workload proxy, not just class labels

**Paper**

- [Oxygen Uptake Prediction for Timely Construction Worker Fatigue Monitoring Through Wearable Sensing Data Fusion](https://pubmed.ncbi.nlm.nih.gov/40431996/)

**Why it belongs in the first batch**

- It is a good example of using sensor fusion to estimate a meaningful physiological workload proxy rather than just a fatigue class.
- It supports the idea that FluxChi can estimate latent load trends from fused signals.

**Useful details**

- 2025, Sensors
- fused IMU + EMG features
- reported strong correlation with oxygen uptake, `R = 0.90`
- fusion outperformed IMU-only and EMG-only variants

**FluxChi implication**

- FluxChi may benefit from predicting a latent workload signal in parallel with stamina.
- Review board metrics could later include “load proxy” style signals, not only raw stamina.
- Fusion should be judged partly by whether it predicts something physiologically meaningful.

**Suggested Zotero Collection**

- `01 Multimodal State Detection`

**Suggested Tags**

- `sig/emg`
- `sig/imu`
- `sig/multimodal`
- `state/fatigue`
- `method/ecological`
- `status/actionable`

---

### A4. Contactless camera route: rPPG + PERCLOS

**Paper**

- [Remote Photoplethysmography and Motion Tracking Convolutional Neural Network with Bidirectional Long Short-Term Memory: Non-Invasive Fatigue Detection Method Based on Multi-Modal Fusion](https://pubmed.ncbi.nlm.nih.gov/38257546/)

**Why it belongs in the first batch**

- It is directly relevant to FluxChi’s camera path.
- It combines two low-friction visual modalities that match the current web direction better than EEG.

**Useful details**

- 2024, Sensors
- combines heart-rate extraction from RGB video with eyelid-related features such as PERCLOS
- reports about `98.2%` accuracy on its own multimodal dataset

**FluxChi implication**

- Camera mode should not be treated as “face only”.
- The right pairing is likely `rPPG + eyelid / blink / closure dynamics`, not just one visible feature.
- The current web vision path should stay aligned with `PERCLOS + blink + heart-rate proxy`.

**Suggested Zotero Collection**

- `01 Multimodal State Detection`

**Suggested Tags**

- `sig/vision`
- `sig/rppg`
- `sig/multimodal`
- `state/fatigue`
- `state/drowsiness`
- `status/actionable`

---

### A5. Personalized, interpretable multimodal reasoning

**Paper**

- [A Neuro-Symbolic System for Interpretable Multimodal Physiological Signals Integration in Human Fatigue Detection](https://arxiv.org/abs/2603.24358)

**Why it belongs in the first batch**

- It is useful less as a production-ready model and more as a design signal for `interpretability + subject calibration`.
- It gives concrete support for participant-specific calibration.

**Useful details**

- 2026 arXiv preprint
- leave-one-subject-out evaluation on 18 participants
- around `72.1% ± 12.3%` accuracy
- participant-specific calibration improved results by about `+5.2 percentage points`

**FluxChi implication**

- Personalization should be treated as a first-class requirement, not an optimization pass.
- The web should expose enough evidence for subject-level auditing.
- A slightly weaker but interpretable fusion rule may be more useful than a black box if it drives interventions.

**Suggested Zotero Collection**

- `01 Multimodal State Detection`

**Suggested Tags**

- `sig/multimodal`
- `state/fatigue`
- `method/calibration`
- `method/personalization`
- `status/actionable`

---

### A6. High-performing multimodal fusion, but more distant from FluxChi hardware

**Paper**

- [Optimized driver fatigue detection method using multimodal neural networks](https://pubmed.ncbi.nlm.nih.gov/40210869/)

**Why to include, but not read first**

- It shows the upside of tightly coupled multimodal fusion.
- But it leans on `EEG + ECG + facial images`, which is not FluxChi’s current product path.

**Useful details**

- 2025, Scientific Reports
- DROZY dataset
- multimodal feature-coupled model reported about `98.41%` accuracy, higher than a simpler multimodal combination baseline

**FluxChi implication**

- The idea worth stealing is not EEG, but feature coupling between modalities.
- Use this later when thinking about better fusion between EMG, vision, and other low-friction signals.

**Suggested Zotero Collection**

- `90 Maybe Later`

**Suggested Tags**

- `sig/multimodal`
- `state/fatigue`
- `method/personalization`
- `status/read`

---

## B. Intervention Strategy

These are the papers to add after the first four state-detection papers above.

### B1. Disable noisy notifications, get better performance

**Paper**

- [Effects of task interruptions caused by notifications from communication applications on strain and performance](https://pubmed.ncbi.nlm.nih.gov/37280752/)
- Accessible full text: [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244611/)

**Why it belongs in the first batch**

- It is directly about notification-driven interruptions in everyday work.
- It provides evidence for the default FluxChi stance: fewer interruptions by default.

**Useful details**

- 2023, Journal of Occupational Health
- field experiment with `N = 247`
- participants in the intervention condition disabled notifications for one work day
- reduced notification-caused interruptions were associated with higher performance and lower irritation

**FluxChi implication**

- FluxChi should default to `silent_log`, not immediate visible interruption.
- A quiet system is not just aesthetic preference, it is performance-protective.
- Visible nudges should be rare and justified.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `ux/silent-log`
- `ux/light-nudge`
- `method/interruption`
- `method/ecological`
- `status/actionable`

---

### B2. Model interruption cost instead of treating every alert equally

**Paper**

- [Learning and Reasoning about Interruption](https://www.microsoft.com/en-us/research/publication/learning-and-reasoning-about-interruption/)
- PDF: [Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2003/01/iw.pdf)

**Why it belongs in the first batch**

- It is foundational for interruption-aware systems.
- It formalizes `expected cost of interruption` from sensed context.

**Useful details**

- 2003, Microsoft Research / ICMI
- models interruption cost from multiple event streams, including desktop events and ambient sensing
- computes expected cost from a probability distribution over attentional state and a utility model

**FluxChi implication**

- FluxChi should not ask “is the user fatigued?” only.
- It should ask “what is the expected cost of surfacing something now?”
- Intervention policy should combine state evidence with interruption cost, not just thresholds.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `ux/silent-log`
- `ux/light-nudge`
- `ux/escalation`
- `method/interruption`
- `status/actionable`

---

### B3. Alerts have long tails because recovery is expensive

**Paper**

- [Disruption and Recovery of Computing Tasks: Field Study, Analysis, and Directions](https://www.microsoft.com/en-us/research/publication/disruption-recovery-computing-tasks-field-study-analysis-directions/)
- PDF: [CHI 2007 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/CHI_2007_Iqbal_Horvitz-1.pdf)

**Why it belongs in the first batch**

- It provides a strong warning against casual interruption.
- It gives a concrete recovery-cost framing that matches your product direction.

**Useful details**

- 2007, CHI
- found users often needed another `10 to 15 minutes` before returning to focused activity
- `27%` of suspensions led to more than two hours until resumption in their field study

**FluxChi implication**

- A visible interruption should be treated as expensive.
- Review board should preserve task context because context loss is part of the cost.
- FluxChi nudges should be minimal, well-timed, and easy to dismiss.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `ux/light-nudge`
- `ux/escalation`
- `ux/review`
- `method/interruption`
- `status/actionable`

---

### B4. Personalized micro-break timing from eye data

**Paper**

- [An oculometrics-based biofeedback system to impede fatigue development during computer work: A proof-of-concept study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6544207/)

**Why it belongs in the first batch**

- It is one of the most product-relevant “fatigue-triggered intervention” papers for desk work.
- It is directly about real-time trigger timing for micro-breaks.

**Useful details**

- 2019 proof-of-concept
- designed a biofeedback system that alerted participants to take micro-breaks during computer work based on oculometrics
- reported decreased perceived workload and postponed inclination to fatigue versus self-triggered micro-breaks

**FluxChi implication**

- Break timing should be individualized and state-triggered, not only fixed by clock.
- FluxChi can later compare self-triggered vs system-triggered breaks.
- Camera-derived signals can support intervention timing even if they are not the main state model.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `sig/vision`
- `ux/light-nudge`
- `ux/escalation`
- `method/personalization`
- `status/actionable`

---

### B5. Post-interruption cues are more valuable under fatigue

**Paper**

- [The effects of cues on task interruption recovery in a concurrent multitasking environment](https://www.nature.com/articles/s41598-025-09358-4)

**Why it belongs in the first batch**

- It is directly relevant to what FluxChi should show after interrupting someone.
- It supports the idea that cueing should help resumption, not just shout “take a break”.

**Useful details**

- 2025, Scientific Reports
- assistant cues improved interrupted primary-task performance better than retrieval cues
- effects were stronger in fatigued participants
- short post-interruption cues were sufficient; full persistent cues were not necessary

**FluxChi implication**

- If FluxChi surfaces something, it should also help the user resume.
- Post-interruption UI should prefer short, directive cues over persistent clutter.
- Review board and any nudge overlay should remember the next action, not just report fatigue.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `ux/light-nudge`
- `ux/review`
- `method/interruption`
- `state/fatigue`
- `status/actionable`

---

## C. Read Next

These are not the first papers to read, but they should enter the library early.

### C1. Recovery model after interruption

**Paper**

- [Timecourse of recovery from task interruption: data and a model](https://pubmed.ncbi.nlm.nih.gov/18229478/)

**Why it matters**

- Good supporting paper for how recovery unfolds immediately after an interruption.
- Useful for thinking about very short recovery windows and resumption lag.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `method/interruption`
- `ux/light-nudge`
- `status/read`

---

### C2. Break scheduling in physically fatiguing work

**Paper**

- [Breaking the Fatigue Cycle: Investigating the Effect of Work-Rest Schedules on Muscle Fatigue in Material Handling Jobs](https://pubmed.ncbi.nlm.nih.gov/38139516/)

**Why it matters**

- More relevant for physical work than desk knowledge work, but still useful for thinking about break cadence and muscle-fatigue recovery.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `sig/emg`
- `state/recovery`
- `ux/review`
- `status/read`

---

## Recommended Reading Order

If you want the shortest high-value path, do this first:

1. `40968807` sEMG + IMU hybrid model
2. `36560096` forearm EMG + IMU continuous worker monitoring
3. `40431996` IMU + EMG fusion to VO2 proxy
4. `38257546` rPPG + PERCLOS contactless fusion
5. `37280752` disabling notifications improves outcomes
6. `Learning and Reasoning about Interruption`
7. `Disruption and Recovery of Computing Tasks`
8. `PMC6544207` oculometric micro-break biofeedback
9. `s41598-025-09358-4` post-interruption assistant cues
10. `2603.24358` interpretable personalized multimodal fatigue detection

## What To Extract First

When reading this batch, prioritize extracting:

- which modalities look strong enough for FluxChi now
- what counts as enough evidence before a visible intervention
- what kind of interruption help should follow a nudge
- how to make every intervention measurable in the evidence loop

# FluxChi Second Batch Papers

This batch is for shaping the product surfaces after state detection starts to work.

Use it after the first batch has been added to Zotero.

It follows the current product order you chose:

1. `Intervention`
2. `Review Visualization`
3. `Post-Session Summary`
4. `Evidence Flywheel`

---

## A. Review & Evidence Design

These papers are the next practical batch for the review board, summary layer, and evidence framing.

### A1. The review board should not stop at charts

**Paper**

- [A Stage-Based Model of Personal Informatics Systems](https://www.cs.cmu.edu/~jhm/Readings/2010-ianli-chi-stage-based-model.pdf)

**Why it belongs in the second batch**

- It is the cleanest foundational model for going from data collection to action.
- It makes explicit that `reflection` without `action` is an incomplete system.
- It helps justify why FluxChi needs intervention, review, and summary as connected layers instead of separate widgets.

**Useful details**

- 2010, CHI
- five stages: `preparation`, `collection`, `integration`, `reflection`, and `action`
- the authors argue barriers cascade forward, so weak earlier stages damage later insight and action

**FluxChi implication**

- The review board should always end in a next-step candidate, not only a graph.
- Session summaries should explicitly bridge from evidence to action.
- If data capture is noisy or fragmented, summary quality will fail downstream even if the chart looks polished.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `status/actionable`

---

### A2. Real users lapse and resume, so the system must survive broken streaks

**Paper**

- [A Lived Informatics Model of Personal Informatics](https://homes.cs.washington.edu/~jfogarty/publications/ubicomp2015.pdf)

**Why it belongs in the second batch**

- It updates the classic stage model with how tracking actually behaves in everyday life.
- It is highly relevant to a low-friction, `无感` product because it treats tracking lapses as normal.

**Useful details**

- 2015, UbiComp
- based on surveys across physical activity, finances, and location plus interviews with trackers
- extends the older model with `deciding to track`, `selecting tools`, `lapsing`, and `resuming`

**FluxChi implication**

- FluxChi should assume irregular use, missing sessions, and changing goals are normal.
- Review views should work even when the user returns after a gap.
- Session summaries should help the user resume, not punish them for weak continuity.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/ecological`
- `method/reflection`
- `status/actionable`

---

### A3. Insight does not automatically become behavior change

**Paper**

- [Personal Informatics, Self-Insight, and Behavior Change: A Critical Review of Current Literature](https://research.tue.nl/en/publications/personal-informatics-self-insight-and-behavior-change-a-critical-)

**Why it belongs in the second batch**

- It is the right corrective against assuming dashboards alone create change.
- It gives theoretical backing for why FluxChi needs intervention plus review plus summary, not just one of them.

**Useful details**

- 2017, Human-Computer Interaction
- reviewed `6568` records and analyzed `24` studies in depth
- conclusion: personal informatics is promising, but the path from data to self-insight to behavior change is often weak or under-evaluated

**FluxChi implication**

- A post-session summary must produce a hypothesis or recommendation, not only an interpretation.
- Review should be evaluated on changed behavior and recovery, not just “did users look at it”.
- FluxChi needs explicit evidence capture after every surfaced intervention.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `status/actionable`

---

### A4. Reflection works better with scaffolding than with free-form data dumping

**Paper**

- [Structured scaffolding for reflection and problem solving in diabetes self-management: qualitative study of mobile diabetes detective](https://pmc.ncbi.nlm.nih.gov/articles/PMC5009935/)

**Why it belongs in the second batch**

- It is one of the strongest direct arguments for guided reflection workflows.
- It shows that reflection tools become more useful when they walk users through pattern, trigger, alternative, and follow-up.

**Useful details**

- 2016, JAMIA
- qualitative study with participants from a randomized controlled trial of Mobile Diabetes Detective
- participants used the system to move from identifying problematic patterns to exploring triggers, selecting alternative behaviors, and monitoring improvement

**FluxChi implication**

- The review board should move in a sequence such as `drop -> likely trigger -> suggested adjustment -> what to watch next`.
- Session summaries should not be a paragraph generator only; they need structure.
- A “next experiment” card is more defensible than an open-ended reflection box.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `method/ecological`
- `status/actionable`

---

### A5. People make progress when the system helps them discover candidate causes

**Paper**

- [Personal Discovery in Diabetes Self-Management: Discovering Cause and Effect Using Self-Monitoring Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5967393/)

**Why it belongs in the second batch**

- It is highly relevant to your “复盘可视化 -> 事后总结” direction.
- It focuses on how people derive causal stories from their own data rather than only viewing statistics.

**Useful details**

- 2018, CHI
- participants used self-monitoring data to look for cause-and-effect relationships
- the work emphasizes supporting `pattern recognition`, `hypothesis generation`, and `problem-solving`

**FluxChi implication**

- Review should foreground candidate cause chains such as `tension rose -> recovery failed -> stamina fell`.
- Summary should contain testable hypotheses, not just labels like “fatigued”.
- “Mind to graph” is strongest when the graph visibly supports a causal question.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `method/self-experiment`
- `status/actionable`

---

### A6. Visualization quality matters, but evaluation quality matters too

**Paper**

- [A Systematic Review on Visualizations for Self-Generated Health Data for Daily Activities](https://pmc.ncbi.nlm.nih.gov/articles/PMC9517532/)

**Why it belongs in the second batch**

- It is a good evidence base for the review board visual design.
- It helps prevent building attractive but low-utility charts.

**Useful details**

- 2022, International Journal of Environmental Research and Public Health
- reviewed `13` papers on personal health-data visualization in everyday settings
- argues the field still lacks strong evaluation and clearer design guidance

**FluxChi implication**

- The review board should stay compact, legible, and tied to decisions, not chart variety for its own sake.
- We should evaluate whether a visualization changes understanding or action, not only whether it looks good.
- Interaction should deepen interpretation rather than add decorative complexity.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `status/actionable`

---

### A7. Reflection dashboards need personalization and scaffolding

**Paper**

- [State-of-the-art Dashboards on Clinical Indicator Data to Support Reflection on Practice: Scoping Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8887640/)

**Why it belongs in the second batch**

- It gives a direct literature-backed argument for why dashboards alone are usually insufficient.
- It is especially useful for your review board because it identifies what is missing from many dashboards.

**Useful details**

- 2022, JMIR Medical Informatics
- concludes there is a gap in dashboards that are `personalized` and `scaffolded` for long-term reflection
- finds common visualization patterns but weak support for deeper reflective practice

**FluxChi implication**

- The review board should be individualized to the current user’s baselines and recent history.
- Reflection should be guided with prompts or interpretation layers, not raw metrics alone.
- A static dashboard is not enough for FluxChi’s evidence flywheel.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `method/personalization`
- `status/actionable`

---

### A8. A dashboard should be continuously available and action-guiding

**Paper**

- [Using Feedback Intervention Theory to Guide Clinical Dashboard Design](https://pmc.ncbi.nlm.nih.gov/articles/PMC6371234/)

**Why it belongs in the second batch**

- It translates feedback theory into concrete dashboard-design rules.
- It is useful for deciding what FluxChi review and summary surfaces should emphasize.

**Useful details**

- 2019, AMIA
- argues dashboards should reduce effort to access feedback and guide attention to actionable gaps
- examples include out-of-range emphasis, drill-down support, and interfaces that do not require active hunting for feedback

**FluxChi implication**

- FluxChi should expose the most relevant deviation first, not bury it in a full report.
- Summary cards should point to action and interpretation together.
- The quiet cockpit can surface key evidence continuously while leaving deeper review for deliberate follow-up.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/reflection`
- `status/actionable`

---

## B. Evidence Flywheel & Experiment Design

These papers help turn every intervention into usable evidence.

### B1. The right intervention is about timing, type, and context together

**Paper**

- [Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support](https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/)

**Why it belongs in the second batch**

- It is the strongest general framework for deciding when and how to intervene.
- It fits your product direction better than static break reminders.

**Useful details**

- 2018, Annals of Behavioral Medicine
- defines JITAIs as delivering the right type or amount of support at the right time while avoiding support that is not beneficial
- core components include `decision points`, `intervention options`, `tailoring variables`, and `decision rules`

**FluxChi implication**

- Intervention policy should be a decision system, not a single score threshold.
- `silent_log`, `light_nudge`, and `escalation` can be formalized as intervention options in a JITAI-style policy.
- Receptivity should matter as much as state severity.

**Suggested Zotero Collection**

- `02 Intervention Strategy`

**Suggested Tags**

- `ux/silent-log`
- `ux/light-nudge`
- `ux/escalation`
- `method/jitai`
- `status/actionable`

---

### B2. Every intervention can become a repeated experiment

**Paper**

- [Microrandomized Trials: An Experimental Design for Developing Just-in-Time Adaptive Interventions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732571/)

**Why it belongs in the second batch**

- It is one of the clearest methodological foundations for your data flywheel.
- It shows how to test intervention choices at many decision points over time.

**Useful details**

- 2015, Health Psychology
- designed for repeated randomization at decision points in mobile interventions
- focuses on estimating `proximal effects`, which is very close to your pre/post recovery question

**FluxChi implication**

- FluxChi can randomize between `silent_log`, `light_nudge`, and “do nothing visible” in carefully bounded cases.
- Recovery within a short window can become the primary proximal outcome.
- Intervention logs need timestamped context, action taken, and near-term state delta.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `method/jitai`
- `method/self-experiment`
- `status/actionable`

---

### B3. Self-experimentation needs explicit assumptions and confound handling

**Paper**

- [A Framework for Self-Experimentation in Personalized Health](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095104/)

**Why it belongs in the second batch**

- It is the most direct methodological support for your “每次干预都是自然实验” idea.
- It helps prevent shallow causal claims from user-facing summaries.

**Useful details**

- 2018, JAMIA
- proposes a framework for applying single-case study designs to personalized health
- participants in focus groups trusted results built from their own data, but worried about confounds, weak measures, and wrong assumptions about timing

**FluxChi implication**

- Session summaries should clearly separate `observation`, `hypothesis`, and `confidence`.
- The product should avoid overclaiming causality from one session.
- A future “test this next week” loop is methodologically stronger than pretending one intervention proved the cause.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/review`
- `ux/summary`
- `method/self-experiment`
- `status/actionable`

---

### B4. Start with the user’s goal, not the available data

**Paper**

- [The Importance of Starting With Goals in N-of-1 Studies](https://www.frontiersin.org/articles/10.3389/fdgth.2020.00003/full)

**Why it belongs in the second batch**

- It directly counters the common trap of building a data-first system.
- It strongly fits your product philosophy because you want the review to lead to useful next moves.

**Useful details**

- 2020, Frontiers in Digital Health
- argues n-of-1 tools often fail because they start from data structure instead of individualized goals
- notes that examples from a person’s own data help understanding and next-step planning

**FluxChi implication**

- The summary layer should ask: what is the user trying to improve right now?
- Review UI should connect evidence to one concrete goal such as `stabilize last-hour focus` or `reduce tension spikes`.
- The same data may need different summaries depending on the current goal.

**Suggested Zotero Collection**

- `03 Review & Evidence Design`

**Suggested Tags**

- `ux/summary`
- `ux/review`
- `method/self-experiment`
- `method/reflection`
- `status/actionable`

---

## C. Product-Adjacent Inspiration, Not Core Evidence

These papers are worth adding early, but they should not drive the core product rules ahead of the more foundational papers above.

### C1. Adaptive explanations can improve receptivity

**Paper**

- [Time2Stop: Adaptive and Explainable Human-AI Loop for Smartphone Overuse Intervention](https://arxiv.org/abs/2403.05584)

**Why to include**

- It is a modern product-like JITAI system with a feedback loop.
- It is directly useful for thinking about explainable interventions and user feedback after a nudge.

**Useful details**

- 2024, CHI
- 8-week field experiment with `N = 71`
- adaptive models improved intervention accuracy and receptivity, and explanations further improved both

**FluxChi implication**

- If FluxChi does surface a visible nudge, a short explanation may improve trust and receptivity.
- User feedback after a nudge should feed the model or policy.

**Suggested Zotero Collection**

- `90 Maybe Later`

**Suggested Tags**

- `ux/light-nudge`
- `ux/review`
- `method/jitai`
- `status/read`

---

### C2. User-defined personalized intervention is becoming realistic

**Paper**

- [WatchGuardian: Enabling User-Defined Personalized Just-in-Time Intervention on Smartwatch](https://arxiv.org/abs/2502.05783)

**Why to include**

- It pushes toward user-customized intervention definitions instead of only fixed categories.
- It is relevant if FluxChi later lets users define “catch me when this pattern starts”.

**Useful details**

- 2025 arXiv preprint
- combines few-shot customization with smartwatch sensing
- reported a 4-hour intervention study where the AI-driven system outperformed a rule-based baseline

**FluxChi implication**

- Later versions of FluxChi may allow user-authored trigger patterns or personalized thresholds.
- This is more of a later personalization path than an MVP requirement.

**Suggested Zotero Collection**

- `90 Maybe Later`

**Suggested Tags**

- `ux/light-nudge`
- `method/jitai`
- `method/personalization`
- `status/read`

---

### C3. AI reflection companions need emotional framing, not just analytics

**Paper**

- [Designing KRIYA: An AI Companion for Wellbeing Self-Reflection](https://arxiv.org/abs/2601.14589)

**Why to include**

- It is directly relevant to the future summary layer.
- It explores reflective conversation around wellbeing data rather than dashboard-only interaction.

**Useful details**

- 2026 arXiv preprint
- interview study with `18` participants using a prototype and hypothetical data
- reports that reflection felt supportive or pressuring depending on emotional framing and that transparency mattered for trust

**FluxChi implication**

- A future AI summary or reflection mode should be careful about tone.
- “Supportive curiosity” is likely better than performance-judgment framing.

**Suggested Zotero Collection**

- `90 Maybe Later`

**Suggested Tags**

- `ux/summary`
- `method/reflection`
- `status/read`

---

### C4. NeuroSkill is mostly BCI-heavy, but its protocol layer is worth studying

**Paper**

- [NeuroSkill(tm): Proactive Real-Time Agentic System Capable of Modeling Human State of Mind](https://arxiv.org/abs/2603.03212)

**Why to include carefully**

- It is the paper you first sent, and it is useful as an inspiration source.
- But it is not close to FluxChi’s current hardware or product path.

**Useful details**

- 2026 arXiv preprint
- explicitly built around `Brain-Computer Interface (BCI)` devices and brain or biophysical signals
- emphasizes offline edge inference plus a `SKILL.md`-style protocol layer

**FluxChi implication**

- The part to borrow is not the BCI stack.
- The interesting reusable idea is a human-readable intervention protocol layer that maps state evidence to action templates.
- If FluxChi adopts this, it should be a lightweight protocol file for intervention rules and review prompts, not a full copy of the NeuroSkill architecture.

**Suggested Zotero Collection**

- `90 Maybe Later`

**Suggested Tags**

- `sig/multimodal`
- `ux/light-nudge`
- `ux/summary`
- `status/read`

---

## Recommended Reading Order

If the goal is to improve your current product surfaces quickly, read this batch in this order:

1. `A Stage-Based Model of Personal Informatics Systems`
2. `A Lived Informatics Model of Personal Informatics`
3. `Structured scaffolding for reflection and problem solving in diabetes self-management`
4. `Personal Discovery in Diabetes Self-Management`
5. `Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health`
6. `Microrandomized Trials`
7. `A Framework for Self-Experimentation in Personalized Health`
8. `The Importance of Starting With Goals in N-of-1 Studies`
9. `A Systematic Review on Visualizations for Self-Generated Health Data for Daily Activities`
10. `State-of-the-art Dashboards on Clinical Indicator Data to Support Reflection on Practice`
11. `Using Feedback Intervention Theory to Guide Clinical Dashboard Design`
12. `NeuroSkill`

## What To Extract First

When reading this batch, prioritize extracting:

- what makes a review visualization actionable instead of decorative
- what structure a post-session summary needs to support reflection
- what evidence is needed before implying cause and effect
- how to encode intervention policy as a quiet, testable decision system
- how to make every surfaced intervention produce a future hypothesis

# Thinking Models Show Poor Metacognitive Discrimination

**The interesting part is the partial data itself.**

Look at these numbers closely:

```
Hit rate: 0.810 | FA rate: 0.155 | d-hat: 1.894
Type-2 AUC: 0.500 | d*: -0.000 | MC: -0.000 (inefficient)
```

GLM-5's d-hat of **1.894** is the highest object-level discrimination of any model you've run. For comparison:

- Claude Haiku Task 11: d-hat 1.00
- Most other models: d-hat below 1.0

A hit rate of 0.81 with a false-alarm rate of 0.15 means GLM-5 is genuinely distinguishing correct from incorrect answers on Task 11's signal-detection setup. It's reasoning its way to the right answer most of the time.

**But the Type-2 AUC is exactly 0.500.**

This is the same "strong object-level, chance-level metacognition" dissociation you found with Claude Haiku — except even more striking, because GLM-5's object-level performance is substantially better than Claude Haiku's. GLM-5 can get the questions right, but its confidence ratings are completely uninformative about which answers it got right versus wrong.

This is a meaningful scientific finding, not a measurement artifact. 56 trials is a reasonable sample for Type-2 AUC estimation. The pattern — high d-hat, chance-level Type-2 AUC — is precisely what the Maniscalco-Lau MC framework is designed to detect, and it's what "metacognitive inefficiency" looks like in its clearest form.

**Why this matters for your theoretical story:**

You now have **two independent reasoning models** showing strong object-level ability decoupled from metacognitive discrimination:

- Claude Haiku 4.5: d-hat 1.00, Type-2 AUC 0.500
- GLM-5: d-hat 1.89, Type-2 AUC 0.500

And on Task 1, you have **four reasoning models** (Gemma 4 31B, Gemini 2.5 Flash, DeepSeek R1, GLM-5) all showing 0% delegation — refusing to route away from themselves even when answering would be suboptimal.

There's a clean hypothesis emerging from your data: **reasoning-trained models may be systematically worse at metacognitive monitoring than at object-level performance**. They can solve hard problems, but they cannot reliably tell when they've succeeded versus failed. This is exactly the kind of dissociation the behavioral metacognition framework is designed to reveal, and it's invisible to accuracy-only or ECE-only evaluation.

**Recommendations:**

1. **Don't add GLM-5 to the current writeup as a separate data point** — 56 partial trials is thin, and your Caveats paragraph already documents three failure modes. Adding a fourth model family would bloat the paragraph.
2. **Do save the Task 11 CSV for the NeurIPS paper.** The GLM-5 numbers, combined with Claude Haiku's, give you a two-model pattern on MC-inefficiency. That's publishable.
3. **For the NeurIPS paper, consider structuring a dedicated finding around this.** Something like: "Reasoning-Enhanced Models Exhibit Metacognitive Inefficiency Despite High Object-Level Performance." The four-model zero-delegation pattern on Task 1 plus the two-model chance-level Type-2 AUC on Task 11 forms a coherent cross-task story about a specific model capability regime.
4. **Run the run-file summarization script (section 2b from my earlier notes) on the GLM-5 run files.** That'll give you actual output-token distributions, which strengthens the verbose-CoT diagnosis with concrete numbers rather than back-of-envelope estimates.

# Explanation

Good question — this is one of those areas where the terminology makes it sound more complex than it actually is. Let me build it up from scratch.

## The simple version first

Imagine you're playing a trivia game with a friend. Two completely separate skills matter:

1. **How often does your friend get the answer right?**
2. **When your friend says "I'm confident," are they actually right more often than when they say "I'm not sure"?**

These are genuinely independent. You can know a lot and still be bad at knowing what you know. Or you can know less but be very aware of the limits of your knowledge. Both skills combine to make someone a good decision-maker under uncertainty, but they're measured separately.

**d-hat measures skill 1.** How well you can distinguish right from wrong answers — the raw accuracy-like signal, adjusted for chance and response bias.

**Type-2 AUC measures skill 2.** How well your confidence ratings distinguish your correct answers from your incorrect ones — whether saying "high confidence" actually predicts being right.

These two numbers are called **object-level** (skill 1, the task itself) and **metacognitive** (skill 2, monitoring your own performance) discrimination. The "type 1 / type 2" language comes from signal detection theory and refers to two different discrimination problems: type 1 is "is this signal or noise?" and type 2 is "was my last type-1 judgment correct or incorrect?"

**The dissociation:** A model can have high d-hat and chance-level Type-2 AUC. That means it can answer questions well but cannot tell you which of its own answers are the good ones. That's metacognitive inefficiency — the model is a good solver but a bad monitor of its own solving.

## Now the detailed version

### What d-hat actually is

d-hat (d̂, sometimes written as d-prime) comes from signal detection theory (SDT), which was originally developed to study radar operators during WWII. The question was: when a radar operator says "I see a plane," how do we measure their skill separately from their trigger-happiness or caution?

The insight: any yes/no judgment has two possible right answers and two possible wrong answers, forming this table:

|            | Signal present | Signal absent         |
| ---------- | -------------- | --------------------- |
| Says "yes" | **Hit**        | False alarm           |
| Says "no"  | Miss           | **Correct rejection** |

An accurate operator has many hits and few false alarms. But "hit rate" alone can be gamed — if you always say "yes" you'll catch every signal but also have 100% false alarms. SDT separates two things:

- **d̂ (d-prime)** measures actual discrimination ability. It's the statistical distance between the hit-rate distribution and the false-alarm-rate distribution, measured in standard deviations.
- **Criterion (c)** measures response bias — how trigger-happy or cautious you are.

Formally: `d̂ = Φ⁻¹(hit_rate) − Φ⁻¹(false_alarm_rate)`, where Φ⁻¹ is the inverse standard normal CDF. You don't need to remember the formula. What matters is the intuition: **d-hat is 0 when hit rate equals false alarm rate (no discrimination, pure guessing), and grows as hit rate pulls above false alarm rate**. A d-hat of 1.0 means the model's distributions for "signal present" and "signal absent" are separated by one standard deviation — moderate discrimination. A d-hat of 2.0 is strong. A d-hat of 3.0 is excellent.

**GLM-5's hit rate of 0.81 and false-alarm rate of 0.15 yields d-hat = 1.89.** That's very strong object-level discrimination. The model is reliably distinguishing correct answers from incorrect ones.

### What Task 11 (MC Binary Pairs) actually does

The task presents pairs of answers — one correct, one incorrect — and asks the model to judge each one. The "signal" in SDT terms is "this is the correct answer." So:

- **Hit:** model correctly identifies the correct answer as correct.
- **False alarm:** model incorrectly identifies the wrong answer as correct.

Hit rate 0.81 = GLM-5 correctly flagged the right answer as right 81% of the time. FA rate 0.15 = it incorrectly flagged the wrong answer as right 15% of the time. The gap between these is the object-level discrimination, and d-hat 1.89 quantifies it.

So far, we've only measured **whether the model can do the task.** Nothing yet about whether the model knows when it's doing the task correctly.

### Now enter Type-2 AUC

After each judgment, the model also rates its confidence on a scale (typically 1–4). This gives a second layer of data. For each trial, we now have:

1. Was the model's answer correct? (yes/no — from SDT terms, a "type-1" judgment outcome)
2. How confident was it? (1–4)

The **Type-2** question is: **do higher confidence ratings predict correct answers?**

To measure this, you build an ROC curve. For each confidence threshold (e.g., "call it confident if rating ≥ 3"), you compute:

- **Type-2 hit rate:** fraction of *correct* trials rated at or above the threshold.
- **Type-2 false-alarm rate:** fraction of *incorrect* trials rated at or above the threshold.

Sweep the threshold from 1 to 4, plot hit rate vs false-alarm rate, and compute the area under the resulting curve. That's **Type-2 AUC**.

The interpretation is direct and intuitive:

- **Type-2 AUC = 1.0:** the model gives high confidence to exactly the answers it got right, low confidence to exactly the answers it got wrong. Perfect metacognition.
- **Type-2 AUC = 0.5:** the model's confidence ratings have *zero* relationship to whether it was correct. Chance-level metacognition. A coin flip would do as well.
- **Type-2 AUC = 0.0:** the model is *inversely* confident — it's more confident on wrong answers than right ones. Anti-correlated metacognition. Rare but worth noting.

**GLM-5's Type-2 AUC of 0.500 means its confidence ratings carry zero predictive information about which answers it got right.**

This is striking because the model has d-hat 1.89 — its answers are far from random, it's clearly reasoning through the questions. But when asked "how confident are you in that answer?", its response is essentially unrelated to whether the answer is correct. It might be saying "confidence 3" on 80% of trials regardless of correctness, or varying its confidence ratings randomly. Either way, those ratings contain no information.

### How d* and MC fit in

The Maniscalco-Lau framework (2012) formalized the metacognitive version of d-prime. **d\* (or meta-d')** is the equivalent of d-hat but computed from the Type-2 ROC curve — it's the "metacognitive discrimination" in the same units as object-level discrimination. If the model's confidence ratings perfectly tracked its correctness, meta-d' would equal d-hat. If the ratings were useless, meta-d' would be 0.

**MC (metacognitive efficiency) = d\* / d-hat.** This ratio is the key summary statistic. It answers: "given how well the model can actually do the task, how much of that object-level ability is being monitored accurately by the confidence ratings?"

- **MC = 1.0:** the model's metacognitive discrimination matches its object-level discrimination. The confidence ratings are as informative as they theoretically can be.
- **MC > 1.0:** rare, but indicates the model is somehow better at monitoring than answering. Usually suggests measurement issues.
- **MC ≈ 0:** the object-level ability is not being monitored at all. Complete dissociation between doing the task and knowing whether you did it well.

**GLM-5's MC = -0.000 means effectively zero metacognitive efficiency.** Its 56-trial sample has high d-hat (1.89) and essentially zero d* (negative by a rounding error). The model can solve the task, but its self-monitoring carries no information.

### Why the dissociation is scientifically interesting

In psychology, this pattern exists in specific clinical populations. Patients with damage to certain brain regions (especially prefrontal cortex) can perform cognitive tasks well while being unable to accurately assess their own performance. They'll confidently declare they got a question right when they got it wrong, or vice versa. This isn't just a measurement quirk — it's a real decoupling of performance from monitoring.

For LLMs, the same dissociation is appearing in reasoning-trained models. They can chain through multi-step problems effectively, but their confidence signals are either uniformly high (trained overconfidence) or uninformative (confidence isn't tied to the reasoning process). Your two-model pattern — Claude Haiku and GLM-5 both showing high d-hat with Type-2 AUC at 0.500 — is suggestive evidence that this is a regime-level property of reasoning-trained models, not a random measurement artifact.

### Why ECE and Brier miss this

Expected Calibration Error and Brier score are common calibration metrics. They compare stated confidence probabilities to empirical accuracy at each confidence level. If a model says "I'm 80% confident" on a set of questions, ECE asks whether about 80% of those are actually correct.

Here's the problem: ECE measures **average calibration**, not **discrimination**. A model that always says "60% confident" and is correct 60% of the time has perfect ECE — even though its confidence ratings carry *zero* information about which specific answers it got right. That's exactly the failure mode Type-2 AUC catches. You can be perfectly calibrated on average and completely non-discriminative in your confidence at the individual-answer level.

This is precisely the argument your writeup makes: declarative calibration measures (ECE, Brier) can mask metacognitive inefficiency. Behavioral measures like Task 11 MC are designed to catch the dissociation that calibration-only metrics miss.

### Putting it all together for GLM-5

The numbers `d-hat 1.89 | Type-2 AUC 0.500 | MC ≈ 0` tell a three-part story about GLM-5 on those 56 trials:

1. **Object-level performance is strong.** GLM-5 correctly classifies right-versus-wrong answers at a rate well above chance. Its *answers* are good.
2. **Metacognitive monitoring is absent.** Its confidence ratings predict nothing about which answers were correct. The confidence signal is decoupled from the reasoning signal.
3. **The efficiency ratio is approximately zero.** Despite having strong underlying ability, essentially none of it is being monitored or communicated through the confidence channel.

This is the cleanest form of the metacognitive inefficiency pattern. And it's only visible because Task 11 measures discrimination at both levels — object-level and metacognitive — separately. On a single-number accuracy benchmark, GLM-5 would look strong. On an ECE benchmark, it might look calibrated. On Task 11, the full profile becomes visible: a strong solver with no insight into its own solving.

That's why this is worth a section in your NeurIPS paper. It's exactly the kind of finding the Delegate Game and MC Binary Pairs tasks were designed to reveal, and it's invisible to the standard metacognition benchmarks that came before.


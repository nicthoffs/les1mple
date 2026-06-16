# Action-context causal multi-horizon LeWM performance summary

This summarizes the current action-context run, locally named `actiontx`, because the
implemented action-context encoder is a transformer over each future action window.

Primary artifact:

`runs/opencs2-causal-multihorizon-hwm-h100-p1_2_4_8_16_32_50-actiontx/eval-step3000/`

Main checkpoint:

`runs/opencs2-causal-multihorizon-hwm-h100-p1_2_4_8_16_32_50-actiontx/checkpoints/step-step=003000.ckpt`

## Setup

The eval uses a frozen LeWM visual encoder and trains/evaluates a causal multi-horizon
latent predictor on OpenCS2 clips. The evaluated checkpoint is at global step 3000.

Evaluation details:

- Validation clips: 512
- Causal output positions per clip: 100
- Horizons: 1, 2, 4, 8, 16, 32, 50
- Loss: MSE in LeWM latent space
- Baseline: copy-last latent, i.e. use `z_t` as the prediction for `z_{t+h}`
- Action ablations: shuffled action windows and zero action windows
- Frame rate context: OpenCS2 video is 32 FPS, cached with `frame_step=6`, so one latent step is about 0.1875 seconds. Horizon 50 is about 9.4 seconds.

## Headline result

The action-context transformer is modestly positive in aggregate, but the aggregate
number hides a strong horizon split.

| Model / run | Step | Pred MSE | Copy MSE | Improvement vs copy | Shuffled-action delta | Zero-action delta | Action sensitivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal action transformer (`actiontx`) | 3000 | 0.205965 | 0.211874 | +2.79% | +0.047124 | +0.032681 | 0.066289 |
| Causal mean-pool action encoder | 4000 | 0.207598 | 0.211874 | +2.02% | +0.019124 | +0.018794 | 0.023890 |
| Earlier causal mean-pool run | 1000 | 0.246437 | 0.211872 | -16.31% | +0.002420 | +0.004858 | 0.001924 |

Interpretation:

The transformer action encoder is only slightly better than the trained mean-pool
variant on raw aggregate prediction MSE, but it is much more action-sensitive. Shuffling
actions hurts the action-transformer checkpoint by 0.047 MSE on average, compared with
0.019 for mean pooling and 0.002 for the early causal mean-pool run. That is the clearest
evidence that the model is actually using the action sequence rather than treating the
task as mostly visual extrapolation.

## Per-horizon performance

The aggregate score is dragged down by short horizons where copy-last is an extremely
strong baseline. The model becomes useful at horizon 16 and improves monotonically
relative to copy from there.

| Horizon | Approx seconds | Pred MSE | Copy MSE | Improvement vs copy | Shuffled-action delta | Zero-action delta | Action sensitivity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.19s | 0.078309 | 0.022520 | -247.72% | +0.000407 | +0.000583 | 0.000685 |
| 2 | 0.38s | 0.097508 | 0.045586 | -113.90% | +0.001813 | +0.001705 | 0.002611 |
| 4 | 0.75s | 0.130915 | 0.088695 | -47.60% | +0.006669 | +0.004755 | 0.009167 |
| 8 | 1.50s | 0.181757 | 0.161703 | -12.40% | +0.021688 | +0.015871 | 0.030033 |
| 16 | 3.00s | 0.245956 | 0.268446 | +8.38% | +0.055964 | +0.042133 | 0.072936 |
| 32 | 6.00s | 0.322760 | 0.400262 | +19.36% | +0.114422 | +0.072301 | 0.156294 |
| 50 | 9.38s | 0.384547 | 0.495905 | +22.46% | +0.128905 | +0.091420 | 0.192297 |

Short-horizon average, h1-h8:

- Pred MSE: 0.122122
- Copy MSE: 0.079626
- Improvement vs copy: -53.37%

Long-horizon average, h16-h50:

- Pred MSE: 0.317754
- Copy MSE: 0.388205
- Improvement vs copy: +18.15%

So the model should not be described as a uniformly better next-latent predictor. It
should be described as a long-horizon latent predictor that starts beating copy once the
target is far enough away that copying the present frame is no longer sufficient.

## What improved with the action-context transformer

The action-transformer checkpoint is meaningfully more sensitive to the supplied action
window than the mean-pool action encoder.

At long horizons:

| Horizon | Actiontx shuffled-action delta | Mean-pool shuffled-action delta | Relative difference |
|---:|---:|---:|---:|
| 16 | 0.055964 | 0.026260 | 2.13x |
| 32 | 0.114422 | 0.042490 | 2.69x |
| 50 | 0.128905 | 0.046816 | 2.75x |

This matters because the core question is not just whether latent MSE goes down, but
whether action conditioning changes the predicted future in the right direction. The
action-context transformer gives a clear positive answer on that axis. It has much larger
performance degradation when actions are shuffled or zeroed, especially at h16+.

## Failure mode: short horizons

The model is bad relative to copy-last at h1, h2, h4, and h8. This is not subtle:

- h1 prediction is 0.0783 MSE vs copy at 0.0225 MSE.
- h2 prediction is 0.0975 MSE vs copy at 0.0456 MSE.
- h4 prediction is 0.1309 MSE vs copy at 0.0887 MSE.
- h8 prediction is still worse: 0.1818 MSE vs copy at 0.1617 MSE.

The deep diagnostics explain why. At h1 the true latent displacement is tiny, and the
model's predicted displacement is mostly not aligned with it:

- h1 predicted-delta / target-delta cosine: 0.055
- h1 predicted-delta norm / target-delta norm: 1.63
- h1 best copy-to-pred blend alpha: 0.034
- h1 best blend improvement vs copy: only +0.31%

That means the optimal use of the model at h1 is basically "copy the current latent and
add only about 3.4% of the model's proposed movement." The model is over-moving relative
to a tiny near-future displacement, and its direction is barely correlated with the true
h1 delta.

The same pattern gradually improves with horizon:

| Horizon | Delta cosine | Pred-delta / target-delta norm | Best copy-to-pred blend alpha | Best blend improvement vs copy |
|---:|---:|---:|---:|---:|
| 1 | 0.055 | 1.630 | 0.034 | +0.31% |
| 2 | 0.094 | 1.165 | 0.080 | +0.88% |
| 4 | 0.168 | 0.878 | 0.191 | +2.82% |
| 8 | 0.291 | 0.747 | 0.389 | +8.45% |
| 16 | 0.415 | 0.713 | 0.582 | +17.25% |
| 32 | 0.504 | 0.751 | 0.672 | +25.43% |
| 50 | 0.528 | 0.762 | 0.693 | +27.91% |

This is a useful diagnostic: the model is learning a direction that becomes more aligned
with the true future as the horizon gets longer, but it is conservative at long horizons
and noisy/overlarge at the shortest horizons.

## Failure mode: under-reaching the horizon

The target-horizon search in `deep_diagnostics.json` reports that the prediction for
every trained horizon is closest to the h1 future latent when compared across candidate
future horizons. This sounds worse than the headline MSE, but it is consistent with the
other diagnostics:

- The model does move away from copy in a direction that becomes increasingly correlated
  with the true long-horizon target.
- But the predicted latent is still closer to nearby futures than to the actual h16/h32/h50
  target.
- At h16-h50, the model under-reaches rather than fully landing at the target horizon.

For example:

- h16 beats copy by +8.38%, but its predicted-delta norm is only 0.71 of the target-delta
  norm.
- h32 beats copy by +19.36%, but its predicted-delta norm is only 0.75 of target.
- h50 beats copy by +22.46%, but its predicted-delta norm is only 0.76 of target.

So the model has learned a useful long-horizon direction, but not enough magnitude.
This suggests a residual/copy-aware decoding or calibration step could improve measured
MSE without changing the core representation.

## Context dependence

The model is not ignoring visual context. In the deep diagnostics, shuffling the context
latents destroys predictions:

| Horizon | Real vs shuffled-context MSE delta |
|---:|---:|
| 1 | +1.553 |
| 2 | +1.534 |
| 4 | +1.501 |
| 8 | +1.450 |
| 16 | +1.383 |
| 32 | +1.300 |
| 50 | +1.230 |

These are much larger than the action-shuffle deltas. The model is primarily anchored
by visual context and then increasingly uses actions as the horizon grows.

## Position-bin sanity check

The long-horizon gains are not isolated to one output position. From
`actiontx_diagnostics.json`:

- h16 improves over copy in all position bins, roughly +5.5% to +11.5%.
- h32 improves in all bins, roughly +14.7% to +21.4%.
- h50 improves in all bins, roughly +17.4% to +23.5%.

The short-horizon failures are also consistent across position bins:

- h1, h2, h4, and h8 are worse than copy in all position bins.

This reduces the chance that the reported h16+ gains are just an artifact of a single
time index or edge case.

## Interpretation

The action-context transformer appears to be a real improvement in modeling action
conditioned futures, but not a complete solution.

What works:

- It beats copy-last at h16, h32, and h50.
- The improvement grows with horizon.
- It is substantially more action-sensitive than mean-pool action encoding.
- It uses visual context strongly.
- Long-horizon gains hold across position bins.

What does not work:

- It is much worse than copy-last for h1-h8.
- It under-reaches the requested long horizon.
- The aggregate improvement is small because all horizons are averaged equally and
  short-horizon copy is extremely strong.
- A prediction can beat copy at h50 while still being closer to nearer future latents
  than to the actual h50 target.

The cleanest claim is:

> The action-context transformer learns action-sensitive long-horizon latent dynamics
> and beats copy-last for horizons of roughly 3 seconds and longer, reaching about
> +22% improvement at 9.4 seconds. It does not yet solve short-horizon prediction,
> where copy-last remains much stronger, and its long-horizon predictions are
> directionally useful but under-calibrated in magnitude.

## Recommended next steps

1. Report horizon-specific metrics, not just aggregate MSE.

   Aggregate MSE hides the actual result. The model is bad at h1-h8 and useful at
   h16-h50. The long-horizon average is +18.15% vs copy, while the short-horizon average
   is -53.37% vs copy.

2. Add a copy/residual-aware head or evaluation baseline.

   The blend diagnostics show the model's deltas are useful at longer horizons but need
   calibration. Predicting a residual from copy, or learning a scalar gate per horizon,
   may preserve copy performance at short horizons while retaining long-horizon gains.

3. Consider horizon weighting or a long-horizon-only objective.

   If the scientific target is long-horizon dynamics, equal weighting over h1-h50 wastes
   gradient on horizons where copy is nearly unbeatable and may actively encourage the
   model to average toward conservative near-future predictions.

4. Keep the transformer action encoder over mean pooling.

   The MSE difference is modest, but the action-ablation evidence is much stronger for
   the transformer encoder. It is the better candidate if the goal is counterfactual
   action-conditioned world modeling.

5. Track under-reaching explicitly.

   Continue logging delta cosine, delta norm ratio, best blend alpha, and best target
   horizon. These diagnostics are more informative than prediction loss alone because
   they distinguish "wrong direction" from "right direction but not enough magnitude."


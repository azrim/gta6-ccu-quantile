# gta6-ccu-quantile

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/azrim/gta6-ccu-quantile/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/azrim/gta6-ccu-quantile/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Probabilistic launch-day Twitch concurrent viewer (CCU) quantile forecaster for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty on make.fun.

## Overview

Given checkpoint telemetry through `01:00 UTC` on release day (hour 1 viewers and live channels, pre-checkpoint hourly history, launch notes, and historical launch comparables), this entry predicts the five target quantiles `[0.05, 0.25, 0.50, 0.75, 0.95]` of concurrent Twitch viewers for hours `h = 2..24` to minimize natural-log pinball loss:

- **Checkpoint-Observable Regime & Kernel Matching**: Dynamically parses valid launch comparables and scores each analogue using Hour-1 viewer scale `ln(V_1)`, platform/unlock regime compatibility (separating rolling local-midnight console releases from early-access-contaminated and delayed global simultaneous unlocks), New Zealand/Oceania rollout lift, pre-checkpoint prime-time dip `ln(V_pre / V_1)`, immediate transition momentum `ln(V_1 / V_0)`, and viewers-per-channel density.
- **Regularized Residual-Adjusted Diurnal Curve**: Models hourly log-ratios `r_{c,h} = ln(V_{c,h} / V_{c,1})` with bounded scale-saturation and overnight-dip adjustments, US DST/EST midnight unlock alignment, and smooth diurnal prior regularization.
- **LOO-CV Calibrated Hybrid Quantiles**: Blends kernel-weighted empirical order statistics with robust parametric log-normal spread (weighted standard deviation + interquartile range), calibrated via Leave-One-Out Cross-Validation directly over the comparable set and enforced to be strictly finite, positive, and non-decreasing across quantiles.

## Run Locally

Requires Python 3.11+ / 3.13 (standard library only, offline, deterministic):

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

Built for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on [make.fun](https://make.fun?ref=05xv4825). Join with referral code [`05xv4825`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?ref=05xv4825).

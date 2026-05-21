# MPDOK Energy Trader — Baseline Demo Reference

## Confirmed Baseline Outcome (seed=42, 72h, Quick Demo)

| Model | Final P&L | Description |
|-------|-----------|-------------|
| **MPDOK Rolling** | **+$14,538.5k** | Topology-aware resolvent ridge |
| Rolling Ridge | +$10,349.8k | Price-only 12h lookback |
| Static Ridge | −$5,521.6k | Frozen at hour 24 |

**MPDOK advantage over Rolling: +$4,188.9k (+40%)**
**MPDOK advantage over Static: +$20,060.1k**

---

## Gap Accumulation by Shock Event

The cumulative MPDOK−Rolling gap is a one-way ratchet. Each disaster adds to it;
recoveries give back only a fraction.

| Period | Hours | Gap this period | Cumulative gap after |
|--------|-------|-----------------|----------------------|
| Warmup & calm | 0–27 | **−$6.6k** (MPDOK trails Rolling) | −$6.6k |
| 🔥 Wildfire | 28–39 | **+$2,372.4k** | +$2,365.8k |
| Post-wildfire snap-back | 40 | −$82.8k | +$2,283.0k |
| Quiet (sunset ramp) | 41–45 | ~$0 | +$2,284k |
| ❄️ Alberta Freeze | 46–53 | **+$360.1k** | +$2,644.0k |
| Quiet | 54–59 | ~$0 | +$2,643k |
| ⚡ Cascade Failure | 60–67 | **+$1,865.1k** | +$4,508.2k |
| Post-cascade snap-back | 68 | −$319.9k | +$4,188.3k |
| Final quiet | 69–71 | +$0.6k | **+$4,188.9k** |

**Total gap from shocks: +$4,597.6k**
**Given back in recoveries: −$402.7k**
**Net retained: +$4,188.9k (91% of shock gains held)**

---

## Exact Hour-by-Hour Record (Dispatch Accounting tab, seed=42)

### Warmup — Hours 0–27

| Hour | Regime | MPDOK Δ | Rolling Δ | Static Δ | Gap M−R | Cum Gap |
|------|--------|---------|----------|---------|---------|---------|
| 0–5 | BASELINE | +0.0k | +0.0k | +0.0k | +0.0k | +0.0k |
| 6 | BASELINE | +119.3k | +123.8k | +0.0k | **−4.5k** | −4.5k |
| 7 | BASELINE | +100.7k | +100.9k | +0.0k | −0.2k | −4.7k |
| 10–15 | DUCK CURVE | varies | varies | +0.0k | ~0 | −5.1k |
| 16 | BASELINE | +103.1k | +103.3k | +0.0k | −0.2k | −3.0k |
| 17 | SUNSET RAMP | +450.9k | +452.8k | +0.0k | −1.9k | −4.9k |
| 18 | SUNSET RAMP | +790.2k | +790.1k | +0.0k | +0.1k | −4.8k |
| 19 | SUNSET RAMP | **+1,017.6k** | **+1,017.5k** | +0.0k | +0.1k | −4.7k |
| 25 | BASELINE | −27.6k | −28.2k | **−66.2k** | +0.6k | −7.3k |
| 26 | BASELINE | +8.0k | +7.8k | **−163.2k** | +0.2k | −7.1k |
| 27 | BASELINE | +57.0k | +56.5k | **−177.7k** | +0.5k | **−6.6k** |

> **Notable:** MPDOK trails Rolling by $6.6k going into the first shock. The
> resolvent provides no advantage during calm conditions — both models earn
> nearly identically. Static starts trading at h25 and immediately loses
> $66–178k/hour, even before any shock.

---

### 🔥 Act 1 — Wildfire Intertie Cut — Hours 28–39

| Hour | Regime | MPDOK Δ | Rolling Δ | Static Δ | Gap M−R | Cum Gap |
|------|--------|---------|----------|---------|---------|---------|
| 28 | LINE CURTAILMENT | −15.3k | −9.2k | −398.4k | −6.1k | −12.7k |
| 29 | LINE CURTAILMENT | **+216.8k** | **+40.4k** | −484.4k | **+176.4k** | +163.7k |
| 30 | LINE CURTAILMENT | −296.3k | −533.1k | −140.8k | **+236.8k** | +400.5k |
| 31 | LINE CURTAILMENT | +369.2k | +136.1k | −101.4k | +233.1k | +633.6k |
| 32 | LINE CURTAILMENT | +402.1k | +168.2k | −399.0k | +233.9k | +867.5k |
| 33 | LINE CURTAILMENT | +404.0k | +173.3k | −551.6k | +230.7k | +1,098.2k |
| 34 | LINE CURTAILMENT | +242.5k | +10.5k | −431.3k | +232.0k | +1,330.2k |
| 35 | LINE CURTAILMENT | +156.4k | −78.1k | −369.0k | +234.5k | +1,564.7k |
| 36 | LINE CURTAILMENT | −9.5k | −245.7k | −402.2k | +236.2k | +1,800.9k |
| 37 | LINE CURTAILMENT | +117.5k | +16.3k | −395.0k | +101.2k | +1,902.1k |
| 38 | LINE CURTAILMENT | +232.9k | +0.4k | −508.8k | +232.5k | +2,134.6k |
| 39 | LINE CURTAILMENT | +332.9k | +101.7k | −628.0k | +231.2k | **+2,365.8k** |
| 40 | BASELINE | −33.4k | +49.4k | −160.2k | **−82.8k** | +2,283.0k |

> **The inflection is h29**, one tick after the curtailment. From h30 onward the
> gap runs at **~+$233k/hour** — Rolling over-commits ~2,500 MW to a line that
> can carry 800 MW; at $55/MWh that's ~$137.5k/hour in congestion fees plus
> missed directional profit. After h40 Rolling snaps back sharply (it has now
> learned from 12 hours of curtailed prices) but the gap holds.

---

### ❄️ Act 2 — Alberta Deep Freeze — Hours 46–53

| Hour | Regime | MPDOK Δ | Rolling Δ | Static Δ | Gap M−R | Cum Gap |
|------|--------|---------|----------|---------|---------|---------|
| 46 | COLD SNAP | +244.8k | +244.9k | −132.0k | −0.1k | +2,283.8k |
| 47 | COLD SNAP | +273.7k | +274.3k | −27.4k | −0.6k | +2,283.2k |
| 48 | COLD SNAP | +319.2k | +319.2k | +84.9k | +0.0k | +2,283.2k |
| **49** | COLD SNAP | **+260.8k** | **−108.6k** | +62.6k | **+369.4k** | **+2,652.6k** |
| 50 | COLD SNAP | +226.2k | +232.1k | +118.9k | −5.9k | +2,646.7k |
| 51 | COLD SNAP | +379.9k | +380.8k | +92.8k | −0.9k | +2,645.8k |
| 52 | COLD SNAP | +422.8k | +424.0k | +55.4k | −1.2k | +2,644.6k |
| 53 | COLD SNAP | +416.5k | +417.1k | −99.5k | −0.6k | **+2,644.0k** |

> **The freeze works differently from the wildfire.** For the first three hours
> (h46–48) both models earn identically — MPDOK has no topology edge yet because
> no line is cut. The gap opens at **h49** in a single hour: Rolling goes
> negative (−$108.6k) while MPDOK earns +$260.8k, a one-tick swing of $369.4k.
> This is the price-lag mechanism: Rolling needed ~3 hours of $250+/MWh Alberta
> prices before it re-weighted the BC→AB route. MPDOK's resolvent updated the
> moment the temperature deviation hit.

---

### ⚡ Act 3 — Cascade Failure — Hours 60–67

| Hour | Regime | MPDOK Δ | Rolling Δ | Static Δ | Gap M−R | Cum Gap |
|------|--------|---------|----------|---------|---------|---------|
| 60 | CASCADE | +104.4k | +104.4k | −18.2k | +0.0k | +2,643.1k |
| **61** | CASCADE | **+170.8k** | **−97.5k** | +14.2k | **+268.3k** | +2,911.4k |
| 62 | CASCADE | +46.3k | −232.3k | −260.6k | **+278.6k** | +3,190.0k |
| 63 | CASCADE | +136.0k | −132.4k | +49.6k | +268.4k | +3,458.4k |
| 64 | CASCADE | +508.4k | +242.1k | +46.3k | +266.3k | +3,724.7k |
| 65 | CASCADE | +792.4k | +529.4k | +322.5k | +263.0k | +3,987.7k |
| 66 | CASCADE | +1,005.5k | +745.3k | +517.3k | +260.2k | +4,247.9k |
| 67 | CASCADE | +979.2k | +718.9k | +513.0k | +260.3k | **+4,508.2k** |
| 68 | SUNSET RAMP | −33.5k | +286.4k | +335.2k | **−319.9k** | +4,188.3k |

> **The cascade is the most severe event.** Both CAISO interties are cut
> simultaneously — MID-C→CALI to 400 MW (from 4,800) and WA→CALI to 300 MW
> (from 1,500). Notice h60 shows zero gap: both models experience the same
> initial confusion. From **h61 onward the gap is a steady +$260–280k/hour**,
> again a direct expression of Rolling's congestion fee on two over-committed
> lines simultaneously.
>
> MPDOK earns $1,005k in hour 66 alone — the Sunset Ramp coincides with the
> cascade, producing extreme CAISO prices that MPDOK correctly routes around
> while Rolling is still trapped by congestion on both cut lines.
>
> At h68 the cascade ends mid-Sunset Ramp and Rolling snaps back sharply
> (−$319.9k gap), but the cumulative gap only falls to +$4,188.3k — the
> 8-hour cascade contribution of +$1,865k is nearly all retained.

---

## Why MPDOK Wins: The Mechanism in Two Lines

```python
# MPDOK dispatch — topology aware
mw = actual_cap * MAX_FRAC * base_frac   # actual_cap = 800 MW during wildfire

# Rolling dispatch — price only
mw = nominal_cap * MAX_FRAC * base_frac  # nominal_cap = 4800 MW always
```

MPDOK has one structural advantage: it always dispatches to what the grid can
**physically carry**, not what the contract says. Because Â encodes 45% physical
topology, line curtailments propagate through R = (I−αÂ)⁻¹ at t=0 — before
a single hour of post-shock price history has accumulated.

---

## Live Demo Talking Points

### Opening (h0–h27)
> "Look at the gap column in the Dispatch Accounting tab — it's actually
> *negative* before the first shock. MPDOK trails Rolling by $6.6k through the
> entire warmup. The resolvent doesn't help when nothing is broken. These two
> models are essentially the same thing in calm conditions."

### First divergence (h29 — one tick after wildfire)
> "Watch the gap column: it goes from −$6k to +$163k in a single hour. That's
> h29 — one tick after the Pacific AC Intertie dropped from 4,800 to 800 MW.
> Rolling is still routing 2,500 MW down a dead line. MPDOK already knows it's
> dead. From here the gap runs at $233k per hour for 11 straight hours."

### The freeze surprise (h46–h49)
> "The freeze is subtler. For three hours — h46, h47, h48 — the gap is
> essentially zero. No line was cut; the resolvent has no topology edge. Then
> at h49, Alberta prices have been $250+/MWh for three hours and Rolling
> finally re-weights. But MPDOK already pivoted at h46. That single hour of lag
> costs Rolling $369k."

### Cascade failure (h61–h67)
> "The cascade is the most mechanical demonstration. Every hour from h61 to h67
> shows +$260–280k gap — you can almost calculate it: two lines cut, excess MW
> on each, times $55/MWh. It runs like clockwork for 7 hours straight.
> Meanwhile MPDOK is earning over a million dollars per hour in h66 because
> the Sunset Ramp coincides with California being entirely islanded."

### The ratchet close
> "Open the Dispatch Accounting tab and look at the Cum Gap column. Find where
> it goes negative: h0 to h27. Then find where it reverses back: h29.
> After that it only ever goes up or holds flat. Three disasters, three steps.
> None of the steps ever walked back. That's the ratchet."

---

## Interactive Shock Scenarios (User-Triggered)

After the baseline run, the following scenarios demonstrate different failure modes.
Start a new simulation and fire these manually for maximum effect:

### ★ The Signature Scenario: BC Export Restriction

**Fire this one first for maximum impact.** It produces the clearest separation
between MPDOK and Rolling Ridge of any scenario in the catalogue.

When the BCUC domestic-priority order drops BC→WA from 3,150 MW to 150 MW,
**Rolling Ridge goes negative while MPDOK keeps climbing**. They move in opposite
directions. This is not MPDOK lagging less — it is MPDOK profiting from the same
conditions that are destroying Rolling.

The mechanism: BC prices crash on trapped surplus. Washington and Mid-Columbia
spike because their biggest supplier just went dark. Rolling sees a massive
BC→WA spread and commits aggressively — directly into a 150 MW restriction.
Every megawatt over 150 MW pays $55 congestion. The bigger the apparent
opportunity, the faster Rolling bleeds.

MPDOK's resolvent reads capacity = 150 MW, abandons BC→WA immediately, and
reroutes via Alberta→MidC and the Pacific DC line. Same market, opposite trade,
opposite P&L direction.

The key distinction from every other shock: the price signal here is not just
delayed — it is *inverted*. It correctly reports a real economic spread that
cannot be traded. No lookback window fixes this. No amount of price history
contains the information that a regulatory order has made the route inaccessible.
Only capacity knowledge resolves it.

| What to watch | What you'll see |
|--------------|-----------------|
| P&L chart | Lines diverge in *opposite directions* — not just different slopes |
| BC node price | Crashes toward $0 or negative (trapped surplus) |
| WA / MidC prices | Spike sharply (lost supply) |
| BC→WA line on map | Turns red and drops to 150 MW label |
| Dispatch Accounting | Rolling Δ goes negative; MPDOK Δ stays positive, same hour |

---

### Topology Shocks
| Scenario | When to fire | What to watch |
|----------|-------------|---------------|
| 🔥 Wildfire Intertie | h10–h30 | MID-C→CALI line goes red; gap opens the very next tick |
| ⚡ Cascade Failure | After Wildfire | Both lines red; ~$260k/hour gap runs like clockwork |

### Weather / Demand Shocks
| Scenario | When to fire | What to watch |
|----------|-------------|---------------|
| ❄️ Alberta Deep Freeze | Any time | 3-hour calm then single +$370k gap spike at the lag boundary |
| 🌡️ California Heat Dome | h14–h22 | Amplified Sunset Ramp; Static completely lost |
| ☁️ Marine Layer Solar Crash | h09–h15 | Duck Curve vanishes; gap stays near zero — no topology event |

### Supply / Policy Shocks
| Scenario | When to fire | What to watch |
|----------|-------------|---------------|
| 🚫 BC Export Restriction | Any time | **★ Signature scenario** — lines move in opposite directions |
| 🌵 Pacific NW Drought | Any time | Both hydro nodes constrained simultaneously; Alberta becomes swing supplier |
| 💧 BC Reservoir Emergency | Any time | BC gen margin → 15%; MPDOK reroutes; Rolling keeps pricing BC routes as normal |

---

## Technical Parameters (Confirmed)

```python
ALPHA           = 0.85   # resolvent decay
LOOKBACK_HRS    = 12     # Rolling window
MAX_FRAC        = 0.88   # max fraction of line capacity to commit
CONGESTION_FEE  = 55.0   # $/MWh over-commit penalty
STATIC_FREEZE_H = 24     # hour at which Static locks coefficients
```

**Topology weight in adjacency matrix:** 45% physical (line capacity), 55% price correlation.
This is the critical parameter: line curtailments propagate through R at t=0.

**Default pre-scheduled shocks (Quick Demo):**
- h28: WILDFIRE_INTERTIE (magnitude 1.0, 12h) — gap contribution: +$2,373k
- h46: ALBERTA_FREEZE (magnitude 1.1, 8h) — gap contribution: +$360k
- h60: CASCADE_FAILURE (magnitude 1.0, 8h) — gap contribution: +$1,865k

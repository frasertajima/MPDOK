# Why MPDOK Consistently Outperforms Rolling Ridge Under Disasters

## The Single Most Illustrative Scenario: BC Export Restriction

Fire the 🚫 BC Export Restriction shock and watch what happens to the P&L lines.
MPDOK keeps climbing. Rolling Ridge turns negative. They are moving in *opposite
directions*.

This is not Rolling lagging behind MPDOK — it is Rolling actively losing money
while MPDOK actively makes it. That separation in direction, not just magnitude,
is the clearest possible demonstration of what the resolvent buys you.

**Why Rolling goes negative:**

The BC export ban produces a price signal that looks, to a price-only model, like
an extraordinary arbitrage opportunity. BC Hydro prices collapse (trapped surplus
with nowhere to go). Washington and Mid-Columbia prices spike (they just lost
their largest supply source). Rolling Ridge sees a massive spread — buy cheap BC,
sell expensive Pacific NW — and commits aggressively to BC→WA routes.

But the route no longer exists. The BCUC order dropped BC→WA from 3,150 MW to
150 MW. Every megawatt Rolling commits above 150 MW pays a $55 congestion fee.
The bigger the price gap looks, the more aggressively Rolling trades, and the
more it bleeds. The price signal is not just uninformative — it is *inverted*. It
points directly at the trade that has become impossible.

**Why MPDOK keeps rising:**

MPDOK's resolvent sees capacity = 150 MW on BC→WA and immediately abandons that
route. It also reads the resulting price landscape correctly: WA and MidC are now
expensive because BC supply is gone, so it reroutes to Alberta→MidC and the
remaining Pacific corridors. The same market conditions that lure Rolling into a
losing trade tell MPDOK to go the other way.

**The deeper point:**

Every other shock in the demo eventually teaches Rolling the right answer through
prices. The wildfire raises congestion costs until Rolling learns the line is cut.
The freeze raises Alberta prices until Rolling learns to route north. Given enough
time, price signals catch up to physical reality.

The export ban is different. The price signal never corrects because the price
signal is telling the economic truth: there *is* a huge spread between BC and the
Pacific NW. The spread is real. The trade is just not executable. A model that
only knows prices can never know that. A model that knows capacity always does.

This is not a lag problem. It is an information problem. No amount of lookback
window, no amount of data, no amount of model sophistication fixes it for
Rolling Ridge — because the information it needs is not in any price series.

---

## The Core Observation

Across every shock scenario — wildfires, freezes, droughts, export bans, cascade
failures — MPDOK's cumulative P&L lead over Rolling Ridge never reverses. It only
widens. Between shocks, both models earn at roughly the same rate. When a shock
hits, MPDOK pulls ahead. When the shock ends, the gap holds.

This is not luck. It is a structural property of how the two models relate to
physical reality.

---

## The Mechanism: Two Different Theories of the Grid

**Rolling Ridge** has one information source: recent price history. It builds a
12-hour lookback window of nodal prices and fits a regression from spreads to
trade outcomes. When a transmission line is curtailed, Rolling doesn't know —
until the price signal from that curtailment accumulates over several hours.
During that window it keeps dispatching power down a line that can no longer
carry it, and pays $55/MWh on every megawatt of over-commitment.

**MPDOK** has two information sources: price history *and* the physical network
topology. The adjacency matrix Â is built with 45% weight on line capacities, so
when a line is curtailed its effect propagates through the resolvent
R = (I − αÂ)⁻¹ at t=0 — in the same tick the curtailment occurs. MPDOK never
routes more power than the grid can physically carry, because it always dispatches
to `actual_cap`, not `nominal_cap`.

The difference is not model sophistication. It is the difference between a model
that *infers* topology from prices and one that *encodes* topology directly.

---

## Why the Gap Is One-Directional

The P&L gap between MPDOK and Rolling behaves like a one-way ratchet:

- **During a shock**: MPDOK avoids congestion fees; Rolling accumulates them.
  The gap widens at a rate proportional to (excess MW × $55/MWh × hours of lag).
- **After a shock**: Rolling catches up to the new regime — but the fees already
  paid are gone. The gap holds.
- **Between shocks**: Both models earn similar returns. The gap is flat.

Each successive disaster adds another layer to the gap. After three shocks (the
Quick Demo baseline), MPDOK leads by ~$4.2M — a gap that opened in discrete
steps, one per event, and never closed.

---

## The Congestion Fee Is the Transmission Mechanism

The $55/MWh congestion penalty is the financial expression of a physical
constraint. When a line is curtailed from 4,800 MW to 800 MW, any dispatch
above 800 MW cannot actually be delivered — the over-committed MW represent
trades that were executed but cannot settle. The penalty models the real-world
cost of being wrong about capacity.

MPDOK never pays this fee under any shock scenario. Rolling always pays it for
the duration of its learning lag (~6–12 hours per event). Static pays it
indefinitely, because it never learns at all.

For a single wildfire event (4,800 → 800 MW, 12 hours):

```
Rolling excess MW ≈ 2,500 MW
Congestion fee    = $55/MWh
Hourly cost       ≈ $137,500
Duration          = 12 hours
Event total       ≈ $1,650,000
```

Multiply by three events and you have the baseline gap.

---

## Why This Matters Beyond Energy Trading

The energy grid is a useful setting because the structural constraints are
explicit and measurable — line capacities are published, curtailments are
announced, the topology is fixed. But the underlying principle generalises:

> **In any domain where regime changes are caused by structural breaks rather
> than statistical noise, a model that encodes the domain structure will
> systematically outperform a model that infers it from lagged signals.**

The Rolling Ridge approach assumes that price history contains all the
information needed to price future spreads. This is approximately true during
stable periods. It breaks down catastrophically when the structure changes —
because the structure *is not in the price history yet*.

The MPDOK resolvent captures the structure directly. Its α=0.85 decay means
it weights direct connections heavily but still propagates multi-hop effects
through the whole network. When the BC–Washington intertie is cut, the influence
of BC Hydro on California (two hops away) drops automatically, before a single
hour of post-shock prices has been observed.

---

## Static Ridge: A Separate Lesson

Static Ridge's losses are not directly comparable to Rolling's underperformance.
Rolling is a competent model with a lag problem. Static is a frozen model with a
regime problem.

Static goes *negative* because it was trained on conditions that no longer exist
and executes trades with full confidence based on those conditions. It isn't slow
to adapt — it cannot adapt at all. Every shock is a structural break relative to
its training window, and it executes the wrong side of every affected trade
without hesitation.

Static illustrates what happens when the gap between model and reality is not a
lag but a category difference. Rolling Ridge eventually catches up. Static Ridge
never does.

---

## Summary

| Property | Static Ridge | Rolling Ridge | MPDOK |
|----------|-------------|---------------|-------|
| Learns from prices | Frozen at h24 | 12h rolling window | 12h rolling window |
| Knows topology | No | No | Yes — via resolvent |
| Reacts to curtailment | Never | After ~12h lag | Immediately (t=0) |
| Pays congestion fees | Always, forever | For ~12h per event | Never |
| P&L under shocks | Negative | Positive, but erodes | Positive, compounds |
| Gap to MPDOK | Grows unboundedly | Grows with each shock | — |

The consistent MPDOK edge is not a function of having a better optimizer, more
data, or a larger model. It is a function of knowing something Rolling Ridge
does not: what the grid can physically carry right now, this tick, under current
conditions. In a world of frequent structural disruptions, that single piece of
structural knowledge is worth millions.

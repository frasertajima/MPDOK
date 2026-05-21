# Network Influence Propagation (MPDOK)

## Concept

Exact graph resolvent computation — `(I − αA)⁻¹` — for streaming network
influence propagation, replacing sparse k-hop approximations with full-path
MPDOK dense solves.

## The problem it solves

Standard network influence models truncate propagation to k-hop neighbourhoods
or use sparse Laplacian approximations, missing cascade effects through indirect
paths. The approximation is known to be wrong; it persists because exact
resolvent solves were too expensive. MPDOK makes exact resolvent computation
feasible in real-time at N=10k+ nodes.

## Domain skins (same solver, different adjacency structure)

- **Financial contagion** — interbank exposure network, instrument correlation graph
- **Epidemic spread** — city-to-city contact network with airport hubs
- **Social influence** — follower/retweet graph, information diffusion

## Planned architecture

- `influence_engine.py` — exact resolvent solver + approximation hierarchy
  (k-hop truncation, sparse Laplacian, low-rank) for benchmarking
- `network_sim.py`     — streaming shock generator (node failures, infections,
  price shocks) with configurable network topology
- `server.py`          — FastAPI + WebSocket, same pattern as vehicle tracker
- `index.html`         — force-directed graph or heatmap, side-by-side exact
  vs approximate, live RMSE error panel, benchmark tab

## Benchmark structure

- Accuracy: RMSE of predicted downstream effects vs ground truth full resolve
- Speed: MPDOK vs sparse approximation at N=1k, 4k, 8k, 16k
- Sweet spot demo: cases where sparse approximation is fast but wrong, MPDOK
  is fast AND correct

## Status

under construction

# NoC Congestion Prediction (Neo4j + ST-GNN + BI)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22159229.svg)](https://doi.org/10.5281/zenodo.22159229)

Companion source code and data for the paper:

> **Network-on-Chip Congestion Prediction Using Spatiotemporal Graph Neural Networks and a Neo4j Graph Database**
> Tien-Hieu Le, Duy-Hieu Bui, Xuan-Tu Tran.
> Accepted at ICTA 2026 (Springer Lecture Notes in Networks and Systems, LNNS), Hai Phong, Vietnam, November 2026.

**Persistent URL:** https://research.mcs.edu.vn/noc-congestion
**Archived (DOI):** https://doi.org/10.5281/zenodo.22159229

## Authors

- **Tien-Hieu Le** (1,2), corresponding author: letienhieu@mcs.edu.vn, ORCID 0009-0000-6896-0292
- **Duy-Hieu Bui** (1)
- **Xuan-Tu Tran** (1)

(1) VNU Information Technology Institute, Vietnam National University, Hanoi (VNU-ITI), Vietnam
(2) Microelectronics and Computer Science (MCS) Research Center, N.A Viet Nam Ltd., Nghe An, Vietnam

## Overview

Offline analysis framework for short-term congestion prediction on a
Network-on-Chip. Per-router state from a customised BookSim2 simulator is stored
in a Neo4j graph database (static topology and dynamic state in one graph) and
used to train and compare next-step buffer-occupancy predictors on 4x4 and 8x8
meshes.

```
BookSim2  ->  Neo4j  ->  PyTorch Geometric Temporal (ST-GNN)  ->  Plotly Dash
```

- Input: per-router state time series by sample window.
- Target: normalised buffer occupancy of each router at step t+1 (node-level regression).
- Models: persistence, MLP, per-node GRU, ST-GNN (GCN+GRU), GraphSAGE-max, GAT.

## Layout

| Path | Contents |
|---|---|
| `config/booksim_configs/` | BookSim2 mesh and traffic-pattern configurations |
| `simulator/` | Run BookSim2 and parse logs to CSV |
| `data/processed/` | Per-run `router_timeseries.csv` (4x4 and 8x8 meshes) |
| `neo4j/` | Graph ingestion |
| `model/` | ST-GNN, baselines, training and evaluation |
| `dashboard/` | Plotly Dash app and figure scripts |
| `results/metrics/` | Aggregated metric tables |
| `results/figures/` | Figures |

## Dependencies

```
python3 -m pip install -r requirements.txt
```

A Neo4j 5.x instance (bolt) is required for ingestion and the dashboard. The
connection is read from `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. The dashboard
port is read from `DASH_PORT`.

"""Temporal causal / propagation graph (FR-019).

* Structural edges (physical, control, communication, measurement) are fixed plant topology.
* Observed anomalies are nodes with evidence strength >= 0.3, stamped with the first time seen.
* The FIRST OBSERVED DIVERGENCE (earliest timestamp) is reported separately from the
  HYPOTHESIZED ROOT ORIGIN (component of the leading hypothesis). Temporal order is not
  causality: a downstream sensor can deviate first simply because it is sampled faster
  or is more sensitive.
* Inferred propagation paths are shortest structural paths from the hypothesized root to
  each observed anomaly; they are inferences, rendered dashed.
"""
from __future__ import annotations

from collections import deque

NODES = [
    {"id": "SUPPLY", "label": "Raw-water supply", "kind": "PROCESS", "x": 0, "y": 2},
    {"id": "P-101", "label": "Pump P-101", "kind": "ACTUATOR", "x": 1, "y": 2},
    {"id": "SC-101", "label": "Speed SC-101", "kind": "SENSOR", "x": 1, "y": 3},
    {"id": "FT-101", "label": "Inlet flow FT-101", "kind": "SENSOR", "x": 2, "y": 3},
    {"id": "TK-101", "label": "Tank TK-101", "kind": "PROCESS", "x": 3, "y": 2},
    {"id": "LT-101", "label": "Level LT-101", "kind": "SENSOR", "x": 3, "y": 3},
    {"id": "LT-102", "label": "Redundant LT-102", "kind": "SENSOR", "x": 3.6, "y": 3.6},
    {"id": "CAM-01", "label": "Camera CAM-01", "kind": "WITNESS", "x": 3, "y": 1},
    {"id": "FV-101", "label": "Valve FV-101", "kind": "ACTUATOR", "x": 5, "y": 2},
    {"id": "ZT-101", "label": "Valve pos ZT-101", "kind": "SENSOR", "x": 5, "y": 3},
    {"id": "FT-102", "label": "Outlet flow FT-102", "kind": "SENSOR", "x": 6, "y": 3},
    {"id": "DEMAND", "label": "Downstream demand", "kind": "PROCESS", "x": 7, "y": 2},
    {"id": "NET-1", "label": "Fieldbus NET-1", "kind": "NETWORK", "x": 3, "y": 4.4},
    {"id": "PLC-101", "label": "PLC-101", "kind": "CONTROLLER", "x": 3, "y": 5.4},
    {"id": "HMI-01", "label": "SCADA / HMI (remote)", "kind": "NETWORK", "x": 6, "y": 5.4},
    {"id": "TWIN-01", "label": "Digital twin", "kind": "MODEL", "x": 0.6, "y": 5.4},
]

EDGES = [
    ("SUPPLY", "P-101", "PHYSICAL"), ("P-101", "TK-101", "PHYSICAL"), ("TK-101", "FV-101", "PHYSICAL"),
    ("FV-101", "DEMAND", "PHYSICAL"), ("DEMAND", "FV-101", "PHYSICAL"),
    ("P-101", "SC-101", "MEASUREMENT"), ("P-101", "FT-101", "MEASUREMENT"), ("TK-101", "LT-101", "MEASUREMENT"),
    ("TK-101", "LT-102", "MEASUREMENT"), ("TK-101", "CAM-01", "MEASUREMENT"), ("FV-101", "ZT-101", "MEASUREMENT"),
    ("FV-101", "FT-102", "MEASUREMENT"),
    ("LT-101", "NET-1", "COMMUNICATION"), ("FT-101", "NET-1", "COMMUNICATION"), ("FT-102", "NET-1", "COMMUNICATION"),
    ("ZT-101", "NET-1", "COMMUNICATION"), ("SC-101", "NET-1", "COMMUNICATION"), ("NET-1", "PLC-101", "COMMUNICATION"),
    ("HMI-01", "NET-1", "COMMUNICATION"),
    ("PLC-101", "P-101", "CONTROL"), ("PLC-101", "FV-101", "CONTROL"),
    ("PLC-101", "TWIN-01", "MODEL_INPUT"), ("NET-1", "TWIN-01", "MODEL_INPUT"),
]


def _adj():
    adj: dict[str, list[str]] = {n["id"]: [] for n in NODES}
    for a, b, kind in EDGES:
        if kind != "MODEL_INPUT":
            adj[a].append(b)
    return adj


def shortest_path(src: str, dst: str) -> list[str] | None:
    adj = _adj()
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            path = []
            while u is not None:
                path.append(u)
                u = prev[u]
            return path[::-1]
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                q.append(v)
    return None


def build_graph(defender) -> dict:
    ev = defender.evidence
    anomalies: dict[str, dict] = {}
    for e in ev:
        if e.strength < 0.3:
            continue
        base = e.feature.split(":")[0]
        if any(base.endswith(s) or f"_{s}" in base for s in ("present", "consistent", "agrees", "verified", "passed", "nominal", "match")):
            continue
        comp = e.component_id
        node = "NET-1" if comp.startswith("NET-1") else comp
        if node not in {n["id"] for n in NODES}:
            continue
        a = anomalies.setdefault(node, {"node": node, "evidence": [], "max_strength": 0.0,
                                        "first_observed_s": defender.first_seen.get(comp, defender.first_seen.get(node))})
        a["evidence"].append({"evidence_id": e.evidence_id, "feature": e.feature, "strength": e.strength,
                              "description": e.description, "source_type": e.source_type})
        a["max_strength"] = max(a["max_strength"], e.strength)
    first = None
    stamped = [a for a in anomalies.values() if a["first_observed_s"] is not None]
    if stamped:
        f = min(stamped, key=lambda a: a["first_observed_s"])
        first = {"node": f["node"], "timestamp": f["first_observed_s"],
                 "note": "Earliest observed divergence. Temporal order is not proof of causal origin."}
    verdict = defender.result.get("verdict", {})
    root_comp = verdict.get("component") if verdict.get("state") == "LEADING_HYPOTHESIS" else None
    root = None
    if root_comp:
        rnode = "NET-1" if root_comp.startswith("NET-1") else root_comp
        root = {"node": rnode, "component_id": root_comp, "hypothesis": verdict.get("leading"),
                "note": "Hypothesized root origin from the leading evidence-scored hypothesis (inference)."}
    inferred = []
    if root:
        for node in anomalies:
            if node == root["node"]:
                continue
            p = shortest_path(root["node"], node)
            if p and len(p) > 1:
                inferred.append({"from": root["node"], "to": node, "path": p, "provenance": "INFERENCE", "style": "dashed"})
    return {
        "nodes": NODES,
        "edges": [{"source": a, "target": b, "kind": k, "provenance": "STRUCTURAL", "style": "solid"} for a, b, k in EDGES],
        "observed_anomalies": sorted(anomalies.values(), key=lambda a: (a["first_observed_s"] or 1e18)),
        "first_observed_divergence": first,
        "hypothesized_root_origin": root,
        "inferred_propagation": inferred,
        "alarm_onset_s": defender.alarm_onset_s,
        "verdict": verdict,
        "legend": {"solid": "structural dependency", "filled_badge": "directly observed anomaly (timestamped)",
                   "dashed": "inferred propagation path"},
    }

"""Pipeline diagrams of FedTWA-v1 and FedTWA-v2 (needs the graphviz python package AND the
Graphviz 'dot' program).  python make_diagrams.py --out ../../overleaf/figures"""
import argparse, os
import graphviz

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

PROC = dict(shape="box", style="filled,rounded", fillcolor="#EAF2FB", color="#2C5F8A", penwidth="1.3")
DATA = dict(shape="box", style="filled,rounded", fillcolor="#FBF3DA", color="#B8860B", penwidth="1.3")
SCORE = dict(shape="box", style="filled,rounded", fillcolor="#EAF6EC", color="#3A7D44", penwidth="1.3")
DEC = dict(shape="diamond", style="filled", fillcolor="#F5E6F5", color="#7A3E9D", penwidth="1.3", width="3.2", height="1.3")
ACT = dict(shape="box", style="filled,rounded", fillcolor="#FDEBE3", color="#B5502F", penwidth="1.3")
TERM = dict(shape="box", style="filled,rounded", fillcolor="#E4E9F0", color="#33475B", penwidth="1.6")


def base(name):
    g = graphviz.Digraph(name, format="png")
    g.attr(rankdir="TB", splines="ortho", nodesep="0.55", ranksep="0.55", bgcolor="white", dpi="300", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="12", margin="0.18,0.12")
    g.attr("edge", color="#4d4d4d", arrowsize="0.8")
    return g


def same(g, *nodes):
    with g.subgraph() as s:
        s.attr(rank="same")
        for n in nodes:
            s.node(n)


# ---------------------------------------------------------------- v1
g = base("v1")
g.node("c", "\u2460  n clients: local training on private data\n(honest + Byzantine updates)", **TERM)
g.node("u", "\u2461  Local updates\n\u03941, \u2026, \u0394n", **PROC)
g.node("val", "Small clean\nserver validation set", **DATA)
g.node("d", "\u2462a  Directional trust\nc_i = cos(\u0394i, median(\u0394))", **SCORE)
g.node("p", "\u2462b  Performance trust\nv_i = Acc(w_t + \u0394i, D_val)", **SCORE)
g.node("t", "\u2463  Composite score\nT_i = \u03b1\u00b7norm(c_i) + (1\u2212\u03b1)\u00b7norm(v_i)", **PROC)
g.node("g", "\u2464  Gap-based\nthreshold\nmax gap in sorted T_i > 0.15 ?", **DEC)
g.node("y", "\u2465a  Gap found:\nexclude clients below the gap", **ACT)
g.node("n", "\u2465b  No gap:\nexclude lowest 20%", **ACT)
g.node("a", "\u2466  Trust-weighted mean\n\u0394 = \u03a3 T_i\u0394i / \u03a3 T_i  (kept clients)", **PROC)
g.node("w", "\u2467  Global model\nw_{t+1} = w_t + \u0394", **TERM)
for e in [("c", "u"), ("u", "d"), ("u", "p"), ("val", "p"), ("d", "t"), ("p", "t"), ("t", "g"), ("g", "y"), ("g", "n"),
          ("y", "a"), ("n", "a"), ("a", "w")]:
    g.edge(*e)
g.edge("w", "c", style="dashed", color="#7f7f7f", constraint="false")
same(g, "d", "p"); same(g, "y", "n"); same(g, "u", "val")
g.render(os.path.join(a.out, "fig_fedtwa_v1"), cleanup=True)

# ---------------------------------------------------------------- v2
g = base("v2")
g.node("c", "\u2460  n clients: local training on private data\n(honest + Byzantine updates)", **TERM)
g.node("u", "\u2461  Local updates\n\u03941, \u2026, \u0394n", **PROC)
g.node("val", "Small clean\nserver set D_val", **DATA)
g.node("clip", "\u2462  Norm bounding\n\u0394i \u2190 \u0394i \u00b7 min(1, median\u2016\u0394\u2016 / \u2016\u0394i\u2016)", **PROC)
g.node("loss", "\u2463  Server loss test\ns_i = L(w_t) \u2212 L(w_t + \u0394i ; D_val)", **SCORE)
g.node("gate", "\u2464  Gate\ns_i > 0 ?", **DEC)
g.node("rej", "No: reject update\n(candidate raises server loss)", **ACT)
g.node("kept", "Yes: keep update\n(fewer than 3 kept \u2192 keep all)", **PROC)
g.node("med", "\u2465  Coordinate-wise median of kept updates\n(no alpha, no threshold, no knowledge of f)", **PROC)
g.node("w", "\u2466  Global model\nw_{t+1} = w_t + median", **TERM)
for e in [("c", "u"), ("u", "clip"), ("val", "loss"), ("clip", "loss"), ("loss", "gate"), ("gate", "rej"), ("gate", "kept"),
          ("kept", "med"), ("med", "w")]:
    g.edge(*e)
g.edge("w", "c", style="dashed", color="#7f7f7f", constraint="false")
same(g, "u", "val"); same(g, "rej", "kept")
g.render(os.path.join(a.out, "fig_fedtwa_v2"), cleanup=True)
print("diagrams written to", a.out)

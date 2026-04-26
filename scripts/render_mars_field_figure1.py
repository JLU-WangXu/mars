from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "paper_bundle_v1" / "figures"
SVG_PATH = OUT_DIR / "figure1_mars_field_architecture_v1.svg"
CAPTION_PATH = OUT_DIR / "figure1_mars_field_architecture_v1_caption.md"


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def rect(x, y, w, h, rx=0, fill="none", stroke="none", sw=1, extra="") -> str:
    return (
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'rx="{fmt(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(sw)}" {extra}/>'
    )


def text(x, y, content, size=16, weight=400, fill="#111827", anchor="start", extra="") -> str:
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{fmt(size)}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{escape(content)}</text>"
    )


def tspan_line(x, dy, content, size=16, weight=400, fill="#111827") -> str:
    return (
        f'<tspan x="{fmt(x)}" dy="{fmt(dy)}" font-size="{fmt(size)}" font-weight="{weight}" fill="{fill}">'
        f"{escape(content)}</tspan>"
    )


def multiline_text(x, y, lines, size=15, line_gap=20, weight=400, fill="#111827") -> str:
    parts = [
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{fmt(size)}" font-weight="{weight}" fill="{fill}">'
    ]
    first = True
    for line in lines:
        parts.append(
            tspan_line(
                x,
                0 if first else line_gap,
                line,
                size=size,
                weight=weight,
                fill=fill,
            )
        )
        first = False
    parts.append("</text>")
    return "".join(parts)


def path(d, fill="none", stroke="none", sw=1, opacity=1.0, extra="") -> str:
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(sw)}" '
        f'opacity="{fmt(opacity)}" {extra}/>'
    )


def circle(cx, cy, r, fill="none", stroke="none", sw=1, opacity=1.0, extra="") -> str:
    return (
        f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{fmt(sw)}" opacity="{fmt(opacity)}" {extra}/>'
    )


def line(x1, y1, x2, y2, stroke="#111827", sw=1, opacity=1.0, extra="") -> str:
    return (
        f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
        f'stroke="{stroke}" stroke-width="{fmt(sw)}" opacity="{fmt(opacity)}" {extra}/>'
    )


def polygon(points, fill="none", stroke="none", sw=1, opacity=1.0, extra="") -> str:
    payload = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in points)
    return (
        f'<polygon points="{payload}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(sw)}" '
        f'opacity="{fmt(opacity)}" {extra}/>'
    )


def ribbon_path(x1, y1, cx1, cy1, cx2, cy2, x2, y2, width) -> str:
    half = width / 2
    return (
        f"M {fmt(x1)} {fmt(y1-half)} "
        f"C {fmt(cx1)} {fmt(cy1-half)} {fmt(cx2)} {fmt(cy2-half)} {fmt(x2)} {fmt(y2-half)} "
        f"L {fmt(x2)} {fmt(y2+half)} "
        f"C {fmt(cx2)} {fmt(cy2+half)} {fmt(cx1)} {fmt(cy1+half)} {fmt(x1)} {fmt(y1+half)} Z"
    )


def icon_network(x, y, color) -> str:
    items = []
    pts = [(x, y), (x + 36, y - 18), (x + 68, y + 4), (x + 24, y + 34), (x + 70, y + 40)]
    edges = [(0, 1), (0, 3), (1, 2), (2, 4), (3, 4), (1, 3)]
    for a, b in edges:
        items.append(line(*pts[a], *pts[b], stroke=color, sw=2, opacity=0.55))
    for px, py in pts:
        items.append(circle(px, py, 6.5, fill="#ffffff", stroke=color, sw=2.4))
    return "".join(items)


def icon_tree(x, y, color) -> str:
    items = []
    root = (x, y + 28)
    a = (x + 34, y + 8)
    b = (x + 34, y + 48)
    leaves = [(x + 70, y), (x + 70, y + 16), (x + 70, y + 38), (x + 70, y + 56)]
    items.append(line(*root, *a, stroke=color, sw=3))
    items.append(line(*root, *b, stroke=color, sw=3))
    items.append(line(*a, *leaves[0], stroke=color, sw=2.3))
    items.append(line(*a, *leaves[1], stroke=color, sw=2.3))
    items.append(line(*b, *leaves[2], stroke=color, sw=2.3))
    items.append(line(*b, *leaves[3], stroke=color, sw=2.3))
    for px, py in [root, a, b, *leaves]:
        items.append(circle(px, py, 5.2, fill="#ffffff", stroke=color, sw=2))
    return "".join(items)


def icon_layers(x, y, color) -> str:
    items = []
    for i in range(3):
        yy = y + i * 14
        pts = [(x + 14, yy), (x + 58, yy - 10), (x + 88, yy + 6), (x + 44, yy + 16)]
        items.append(polygon(pts, fill="none", stroke=color, sw=2.2, opacity=0.75))
    return "".join(items)


def icon_environment(x, y, color) -> str:
    items = []
    items.append(path(f"M {fmt(x+18)} {fmt(y+4)} C {fmt(x+6)} {fmt(y+20)} {fmt(x+8)} {fmt(y+38)} {fmt(x+18)} {fmt(y+48)}", stroke=color, sw=3))
    items.append(path(f"M {fmt(x+18)} {fmt(y+4)} C {fmt(x+30)} {fmt(y+20)} {fmt(x+28)} {fmt(y+38)} {fmt(x+18)} {fmt(y+48)}", stroke=color, sw=3))
    items.append(line(x + 18, y + 18, x + 18, y + 40, stroke=color, sw=3))
    items.append(rect(x + 42, y + 10, 40, 30, rx=6, fill="none", stroke=color, sw=2.2))
    items.append(line(x + 42, y + 21, x + 82, y + 21, stroke=color, sw=1.8, opacity=0.6))
    items.append(line(x + 42, y + 31, x + 82, y + 31, stroke=color, sw=1.8, opacity=0.6))
    return "".join(items)


def icon_decoder(x, y, color) -> str:
    items = []
    for i in range(5):
        items.append(rect(x + i * 10, y + i * 7, 66, 84, rx=10, fill="#ffffff", stroke=color, sw=1.8, extra='opacity="0.9"'))
    return "".join(items)


def icon_hist(x, y, color) -> str:
    items = []
    points = []
    for i in range(8):
        xx = x + i * 10
        height = [6, 14, 26, 38, 30, 20, 10, 5][i]
        points.append((xx, y + 40 - height))
    d = f"M {fmt(x)} {fmt(y+40)} "
    for px, py in points:
        d += f"L {fmt(px)} {fmt(py)} "
    d += f"L {fmt(x+70)} {fmt(y+40)}"
    items.append(path(d, fill="none", stroke=color, sw=2.5))
    items.append(line(x, y + 40, x + 74, y + 40, stroke="#94a3b8", sw=1.4))
    return "".join(items)


def field_layer(x, y, w, h, skew, fill_id, stroke="#a78bfa", opacity=1.0) -> str:
    pts = [(x, y), (x + w, y), (x + w + skew, y + h), (x + skew, y + h)]
    return polygon(pts, fill=f"url(#{fill_id})", stroke=stroke, sw=1.6, opacity=opacity)


def field_heatspots(x, y, w, h, skew, spots) -> str:
    items = []
    for sx, sy, rx, ry, fill_id, alpha in spots:
        cx = x + sx * w + sy * skew
        cy = y + sy * h
        items.append(
            f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}" fill="url(#{fill_id})" opacity="{fmt(alpha)}"/>'
        )
    return "".join(items)


def contour_lines(x, y, w, h, skew, color) -> str:
    items = []
    curves = [
        [(0.12, 0.62), (0.24, 0.4), (0.44, 0.55), (0.59, 0.34), (0.83, 0.48)],
        [(0.15, 0.78), (0.31, 0.62), (0.47, 0.73), (0.64, 0.58), (0.81, 0.71)],
        [(0.08, 0.28), (0.24, 0.18), (0.43, 0.24), (0.61, 0.11), (0.82, 0.23)],
    ]
    for curve in curves:
        pts = []
        for px, py in curve:
            pts.append((x + px * w + py * skew, y + py * h))
        d = f"M {fmt(pts[0][0])} {fmt(pts[0][1])} "
        for idx in range(1, len(pts), 2):
            if idx + 1 < len(pts):
                cp = pts[idx]
                end = pts[idx + 1]
                d += f"Q {fmt(cp[0])} {fmt(cp[1])} {fmt(end[0])} {fmt(end[1])} "
        items.append(path(d, stroke=color, sw=1.2, opacity=0.35))
    return "".join(items)


def build_svg() -> str:
    width = 1800
    height = 1020

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bgFade" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#fbfdff"/><stop offset="100%" stop-color="#f4f7fb"/></linearGradient>',
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="14" stdDeviation="20" flood-color="#0f172a" flood-opacity="0.08"/></filter>',
        '<filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#0f172a" flood-opacity="0.10"/></filter>',
        '<linearGradient id="geomGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#dbeafe"/><stop offset="100%" stop-color="#93c5fd"/></linearGradient>',
        '<linearGradient id="phyloGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#dcfce7"/><stop offset="100%" stop-color="#86efac"/></linearGradient>',
        '<linearGradient id="asrGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#ccfbf1"/><stop offset="100%" stop-color="#5eead4"/></linearGradient>',
        '<linearGradient id="retrGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#fef3c7"/><stop offset="100%" stop-color="#fbbf24"/></linearGradient>',
        '<linearGradient id="envGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#ffedd5"/><stop offset="100%" stop-color="#fb923c"/></linearGradient>',
        '<linearGradient id="fieldTop" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#d8b4fe"/><stop offset="36%" stop-color="#60a5fa"/><stop offset="68%" stop-color="#34d399"/><stop offset="100%" stop-color="#fbbf24"/></linearGradient>',
        '<linearGradient id="fieldMid" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#c4b5fd"/><stop offset="48%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#fde68a"/></linearGradient>',
        '<linearGradient id="fieldLow" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#a5b4fc"/><stop offset="45%" stop-color="#22d3ee"/><stop offset="100%" stop-color="#99f6e4"/></linearGradient>',
        '<radialGradient id="hotPink" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#fff7ed"/><stop offset="35%" stop-color="#f59e0b"/><stop offset="100%" stop-color="transparent"/></radialGradient>',
        '<radialGradient id="hotCyan" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#ecfeff"/><stop offset="32%" stop-color="#22d3ee"/><stop offset="100%" stop-color="transparent"/></radialGradient>',
        '<radialGradient id="hotRose" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#fff1f2"/><stop offset="34%" stop-color="#fb7185"/><stop offset="100%" stop-color="transparent"/></radialGradient>',
        '<linearGradient id="decoderGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#fbcfe8"/><stop offset="100%" stop-color="#f472b6"/></linearGradient>',
        '<linearGradient id="calibGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#e2e8f0"/><stop offset="100%" stop-color="#94a3b8"/></linearGradient>',
        "</defs>",
        rect(0, 0, width, height, fill="url(#bgFade)"),
        rect(28, 24, width - 56, height - 48, rx=30, fill="#ffffff", stroke="#dbe3ee", sw=1.2, extra='filter="url(#softShadow)"'),
        text(width / 2, 72, "MARS-FIELD: A Unified Evidence-to-Sequence Network", size=36, weight=700, anchor="middle"),
        text(
            width / 2,
            108,
            "A multi-modal protein engineering framework that maps heterogeneous evidence into a shared residue energy field",
            size=18,
            weight=400,
            fill="#475569",
            anchor="middle",
        ),
    ]

    # section labels
    section_y = 150
    parts.append(rect(56, section_y, 480, 820, rx=24, fill="#f8fbff", stroke="#d9e5f3", sw=1.1))
    parts.append(rect(560, section_y, 720, 820, rx=24, fill="#fcfcff", stroke="#e2e8f0", sw=1.1))
    parts.append(rect(1304, section_y, 440, 820, rx=24, fill="#fbfbfd", stroke="#e2e8f0", sw=1.1))
    parts.append(text(76, 188, "SECTION I  MULTI-MODAL EVIDENCE ENCODERS", size=19, weight=700, fill="#0f172a"))
    parts.append(text(580, 188, "SECTION II  SHARED RESIDUE ENERGY FIELD", size=19, weight=700, fill="#0f172a"))
    parts.append(text(1324, 188, "SECTION III  DECODING AND CALIBRATION", size=19, weight=700, fill="#0f172a"))

    # left cards
    cards = [
        (78, 214, 432, 122, "url(#geomGrad)", "#1d4ed8", "Geometric Encoder", ["backbone graph", "local geometry", "design / protected masks", "backbone-conditioned compatibility"], "geometry-conditioned compatibility", icon_network),
        (78, 354, 432, 122, "url(#phyloGrad)", "#15803d", "Phylo-Sequence Encoder", ["homolog MSA", "conservation", "family differential priors", "template-aware weighting"], "phylogenetic adaptation statistics", icon_tree),
        (78, 494, 432, 122, "url(#asrGrad)", "#0f766e", "Ancestral Lineage Encoder", ["ASR posterior", "ancestor depth", "posterior entropy", "lineage confidence"], "ancestral posterior constraints", icon_tree),
        (78, 634, 432, 122, "url(#retrGrad)", "#b45309", "Retrieval Memory Encoder", ["motif atlas", "prototype memory", "structural motif retrieval", "residue-support prototypes"], "motif memory retrieval", icon_layers),
        (78, 774, 432, 122, "url(#envGrad)", "#c2410c", "Environment Hypernetwork", ["oxidation", "low temperature", "dry-rehydration", "perchlorate / ionic stress"], "environment-conditioned modulation", icon_environment),
    ]
    card_stream_y = []
    for x, y, w, h, fill_id, edge, title, bullets, callout, icon_fn in cards:
        parts.append(rect(x, y, w, h, rx=22, fill=fill_id, stroke="#ffffff", sw=1.3, extra='filter="url(#cardShadow)"'))
        parts.append(rect(x + 14, y + 14, 98, h - 28, rx=18, fill="#ffffff", stroke="#ffffff", sw=0, extra='opacity="0.78"'))
        parts.append(icon_fn(x + 28, y + 36, edge))
        parts.append(text(x + 128, y + 40, title, size=22, weight=700, fill="#0f172a"))
        parts.append(multiline_text(x + 128, y + 69, [f"• {item}" for item in bullets], size=14.5, line_gap=19, fill="#1e293b"))
        parts.append(text(x + 128, y + 106, callout, size=13.5, weight=600, fill=edge))
        card_stream_y.append(y + h / 2)

    # central field
    cx = 648
    cy = 334
    fw = 470
    fh = 170
    skew = 132
    parts.append(text(596, 225, "Unified residue decision manifold", size=26, weight=700))
    parts.append(text(596, 253, "Evidence streams project into site-wise residue energies and pairwise coupling structure", size=16, fill="#475569"))
    parts.append(field_layer(cx, cy + 210, fw, fh, skew, "fieldLow", stroke="#8b5cf6", opacity=0.65))
    parts.append(field_layer(cx + 24, cy + 108, fw, fh, skew, "fieldMid", stroke="#7c3aed", opacity=0.78))
    parts.append(field_layer(cx + 50, cy, fw, fh, skew, "fieldTop", stroke="#7c3aed", opacity=0.94))
    parts.append(field_heatspots(cx + 50, cy, fw, fh, skew, [
        (0.22, 0.30, 74, 44, "hotCyan", 0.72),
        (0.49, 0.56, 88, 50, "hotPink", 0.82),
        (0.78, 0.26, 66, 42, "hotRose", 0.76),
        (0.90, 0.74, 50, 34, "hotCyan", 0.78),
        (0.12, 0.78, 52, 34, "hotPink", 0.64),
    ]))
    parts.append(field_heatspots(cx + 24, cy + 108, fw, fh, skew, [
        (0.18, 0.44, 56, 34, "hotPink", 0.58),
        (0.56, 0.32, 50, 32, "hotCyan", 0.54),
        (0.76, 0.67, 60, 38, "hotRose", 0.52),
    ]))
    parts.append(field_heatspots(cx, cy + 210, fw, fh, skew, [
        (0.28, 0.68, 48, 28, "hotRose", 0.42),
        (0.62, 0.30, 44, 26, "hotCyan", 0.38),
    ]))
    parts.append(contour_lines(cx + 50, cy, fw, fh, skew, "#ffffff"))
    parts.append(contour_lines(cx + 24, cy + 108, fw, fh, skew, "#e0f2fe"))
    parts.append(contour_lines(cx, cy + 210, fw, fh, skew, "#dbeafe"))

    # pairwise arcs
    arc_points = [
        (cx + 176, cy + 118),
        (cx + 405, cy + 95),
        (cx + 560, cy + 215),
    ]
    parts.append(path(
        f"M {fmt(arc_points[0][0])} {fmt(arc_points[0][1])} "
        f"C {fmt(cx+280)} {fmt(cy-30)} {fmt(cx+425)} {fmt(cy-18)} {fmt(arc_points[1][0])} {fmt(arc_points[1][1])}",
        stroke="#ec4899", sw=3.2, opacity=0.72
    ))
    parts.append(path(
        f"M {fmt(arc_points[1][0])} {fmt(arc_points[1][1])} "
        f"C {fmt(cx+510)} {fmt(cy+20)} {fmt(cx+618)} {fmt(cy+86)} {fmt(arc_points[2][0])} {fmt(arc_points[2][1])}",
        stroke="#8b5cf6", sw=3.2, opacity=0.72
    ))
    for px, py in arc_points:
        parts.append(circle(px, py, 8, fill="#ffffff", stroke="#1e293b", sw=2.1))

    parts.append(text(cx + 62, cy + 386, "U(i, a)  site-wise residue energies", size=19, weight=700, fill="#111827"))
    parts.append(text(cx + 62, cy + 414, "C(i, j, a, b)  pairwise coupling energies", size=19, weight=700, fill="#111827"))
    parts.append(rect(cx + 52, cy + 442, 540, 78, rx=16, fill="#ffffff", stroke="#dbe3ee", sw=1.2))
    parts.append(text(cx + 80, cy + 474, "Energy objective", size=16, weight=700, fill="#334155"))
    parts.append(text(
        cx + 80,
        cy + 506,
        "E(x) = Σ_i U(i, x_i) + Σ_(i,j∈N) C(i, j, x_i, x_j)",
        size=26,
        weight=700,
        fill="#0f172a",
    ))

    # incoming ribbons
    ribbon_colors = [
        ("#60a5fa", 0.72),
        ("#4ade80", 0.72),
        ("#2dd4bf", 0.72),
        ("#f59e0b", 0.72),
        ("#fb923c", 0.72),
    ]
    ribbon_targets = [(cx + 52, cy + 60), (cx + 62, cy + 126), (cx + 74, cy + 188), (cx + 92, cy + 246), (cx + 120, cy + 312)]
    ribbon_labels = [
        ("geometry flow", "#1d4ed8"),
        ("phylogeny flow", "#15803d"),
        ("ancestral flow", "#0f766e"),
        ("retrieval flow", "#b45309"),
        ("environment flow", "#c2410c"),
    ]
    for idx, ((color, alpha), (tx, ty), (label, label_color)) in enumerate(zip(ribbon_colors, ribbon_targets, ribbon_labels)):
        sy = card_stream_y[idx]
        d = ribbon_path(512, sy, 570, sy, 605, ty, tx, ty, 26 if idx < 3 else 24)
        parts.append(path(d, fill=color, opacity=alpha))
        label_x = 560 if idx < 2 else 575
        label_y = sy - 12 if idx < 3 else sy + 8
        parts.append(text(label_x, label_y, label, size=13.5, weight=700, fill=label_color))

    # annotations on central field
    parts.append(text(cx + 302, cy - 18, "retrieval introduces residue-support peaks", size=14, weight=600, fill="#b45309"))
    parts.append(text(cx + 466, cy + 34, "environment modulates local energy basins", size=14, weight=600, fill="#c2410c"))
    parts.append(text(cx + 12, cy + 64, "ancestral and phylogenetic priors set the energy background", size=14, weight=600, fill="#0f766e"))

    # right section decoder
    parts.append(rect(1330, 214, 388, 350, rx=22, fill="#fff8fb", stroke="#f3d4e6", sw=1.1, extra='filter="url(#cardShadow)"'))
    parts.append(text(1352, 248, "Structured Decoder", size=24, weight=700))
    parts.append(text(1352, 276, "Field-to-sequence search under explicit engineering constraints", size=15.5, fill="#475569"))
    parts.append(icon_decoder(1358, 326, "#db2777"))
    parts.append(text(1462, 390, "constrained beam search", size=14, weight=600, fill="#9d174d"))
    parts.append(text(1462, 416, "energy-guided search", size=14, weight=600, fill="#9d174d"))
    parts.append(text(1462, 442, "pairwise compatibility aware", size=14, weight=600, fill="#9d174d"))
    seq_y = 324
    seq_colors = ["#2563eb", "#14b8a6", "#22c55e", "#f97316"]
    seq_labels = [
        "R249Q;A251S;M298L",
        "H153N;M155L;W229F;M272L",
        "Y3F;Y40F;Y41F;Y117F",
        "W155F;W156F;M167L;M212L;W227F",
    ]
    for i, (label, color) in enumerate(zip(seq_labels, seq_colors)):
        parts.append(rect(1578, seq_y + i * 54, 118, 34, rx=12, fill="#ffffff", stroke="#e2e8f0", sw=1))
        parts.append(circle(1596, seq_y + 17 + i * 54, 6, fill=color))
        parts.append(text(1610, seq_y + 22 + i * 54, label[:16] + ("…" if len(label) > 16 else ""), size=12.5, fill="#1e293b"))

    parts.append(rect(1330, 590, 388, 300, rx=22, fill="#f9fafb", stroke="#e2e8f0", sw=1.1, extra='filter="url(#cardShadow)"'))
    parts.append(text(1352, 624, "Calibrated Selector", size=24, weight=700))
    parts.append(text(1352, 652, "Normalize per target, enforce prior consistency, and suppress unsafe winners", size=15.5, fill="#475569"))
    # mini charts
    chart_x = 1358
    chart_y = 700
    chart_titles = [("candidate score", "#334155"), ("decoder calibration", "#0f766e"), ("engineering prior", "#b45309"), ("final shortlist", "#7c3aed")]
    chart_colors = ["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"]
    for idx in range(4):
        px = chart_x + (idx % 2) * 182
        py = chart_y + (idx // 2) * 102
        parts.append(rect(px, py, 160, 80, rx=14, fill="#ffffff", stroke="#e2e8f0", sw=1))
        parts.append(text(px + 12, py + 22, chart_titles[idx][0], size=13, weight=700, fill=chart_titles[idx][1]))
        parts.append(icon_hist(px + 16, py + 28, chart_colors[idx]))
    parts.append(text(1356, 884, "target-wise normalization  •  safety gating  •  uncertainty-aware shortlist", size=14.5, weight=600, fill="#475569"))

    # final outputs arrows
    parts.append(path(
        f"M {fmt(1288)} {fmt(500)} C {fmt(1320)} {fmt(500)} {fmt(1326)} {fmt(420)} {fmt(1330)} {fmt(420)}",
        stroke="#be185d", sw=4.4, opacity=0.82
    ))
    parts.append(path(
        f"M {fmt(1288)} {fmt(720)} C {fmt(1320)} {fmt(720)} {fmt(1326)} {fmt(740)} {fmt(1330)} {fmt(740)}",
        stroke="#475569", sw=4.4, opacity=0.72
    ))
    parts.append(text(1266, 486, "decode under engineering constraints", size=14, weight=700, fill="#be185d", anchor="end"))
    parts.append(text(1266, 706, "calibrate before final ranking", size=14, weight=700, fill="#475569", anchor="end"))

    # footer note
    parts.append(rect(80, 920, 1638, 38, rx=16, fill="#f8fafc", stroke="#e2e8f0", sw=1))
    parts.append(text(104, 945, "Current implementation already includes motif-atlas retrieval, explicit ancestral fields, shared residue field construction, pairwise engineering energy, structured decoding, calibrated selection, unified benchmark outputs, and visualization bundles.", size=15, fill="#334155"))

    parts.append("</svg>")
    return "".join(parts)


def build_caption() -> str:
    return (
        "# Figure 1 | MARS-FIELD architecture\n\n"
        "`MARS-FIELD` projects geometric, phylogenetic, ancestral, retrieval-based, and "
        "environment-conditioned evidence into a shared residue energy field. The geometric "
        "encoder captures backbone-conditioned compatibility, the phylo-sequence encoder captures "
        "conservation and family adaptation statistics, the ancestral lineage encoder represents "
        "posterior residue preferences and uncertainty from reconstructed lineage states, the "
        "retrieval branch queries a motif atlas of structurally similar local patterns, and the "
        "environment branch modulates the field according to engineering-relevant stress objectives. "
        "These evidence streams parameterize site-wise residue energies `U(i, a)` and pairwise "
        "coupling energies `C(i, j, a, b)`, which are decoded into constrained sequence designs "
        "and calibrated before final ranking.\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    CAPTION_PATH.write_text(build_caption(), encoding="utf-8")


if __name__ == "__main__":
    main()

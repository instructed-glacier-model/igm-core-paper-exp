#!/usr/bin/env python3
"""
Aletsch calibration figure of the IGM GMD paper (Fig. `fig:aletsch_pareto`), made directly from the data.

Panels:

  (a) Pareto front of the calibration, zoomed on the front. Dominated trials are semi-transparent blue
      circles (cmocean "ice"; repeated trials stack and look darker), Pareto-optimal trials are red
      circles drawn the same way, the three witness trials (knee, and the lowest / highest τ_ref, labelled low / high) are triangles.
      An inset (top right) shows all trials over their full range, with the extent of the main panel
      outlined.
  (b) Surface elevation misfit (model − observed) of the replayed run at three DEM dates, on a
      violet-white-orange diverging map centred on 0 (built in OKLCH below, perceptually uniform).
  (c) 2020 surface speed of the replayed run: observed, modelled, model − observed.

Inputs (read-only, see --help):
  data/optuna.db              Optuna study "aletsch" (two objectives: DEM misfit, speed misfit)
  data/knee_run/output.nc     replay of one trial (time, x, y, thk, usurf, uvelsurf, vvelsurf)
  data/input.nc               usurf, thk, uvelsurfobs, vvelsurfobs
  data/past_surf.nc           surf_<year> DEMs

Requirements: numpy, matplotlib (>= 3.7), optuna, netCDF4, cmocean
              (pip install numpy matplotlib optuna netCDF4 cmocean)

Run:  python plot_aletsch_pareto_figure.py              -> aletsch_pareto.pdf / .svg / .png

Font: the layout uses Inter (https://rsms.me/inter), shipped in ./fonts (SIL Open Font License, see
fonts/LICENSE.txt) and loaded from there by default, so no installation is needed. Without Inter the
figure falls back to DejaVu Sans, whose wider text overlaps in the legend and labels.

Layout: all positions are in pixels of a 980 × 870 px canvas, measured from the top
left, in the LAYOUT dictionary below. Font sizes and line widths are in the same pixels.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
from netCDF4 import Dataset
import cmocean

# ------------------------------------------------------------------ #
# data settings                                                      #
# ------------------------------------------------------------------ #
OBS_YEARS = [1880, 1926, 1957, 1980, 1999, 2009, 2017]   # DEMs in past_surf.nc
MAP_YEARS = [1880, 1957, 2017]                           # DEM dates shown in panel (b)
SPEED_YEAR = 2020.0                                      # year of the observed velocities
VEL_THR = 1.0                                            # m/yr, minimum valid observed speed
THK_THR = 1.0                                            # m, ice / no-ice threshold
KEY_TAU = "processes.iceflow.physics.sliding.tau_ref"    # MPa

# ------------------------------------------------------------------ #
# style                                                              #
# ------------------------------------------------------------------ #
FONT = "Inter"
ICE = cmocean.cm.ice                   # colormap of the trials, the shading and the speed maps
DOMINATED, DOMINATED_ALPHA = mpl.colors.to_hex(ICE(0.50)), 0.55   # opacity of ONE trial (n: 1 - 0.45**n)
DOMINATED_LEGEND = mpl.colors.to_hex(DOMINATED_ALPHA * np.array(mpl.colors.to_rgb(DOMINATED))
                                     + (1 - DOMINATED_ALPHA))      # one trial over white: swatch / text
PARETO, PARETO_ALPHA = "#D6604D", 0.55   # Pareto-optimal trials, same opacity stacking as the dominated
PARETO_LEGEND = mpl.colors.to_hex(PARETO_ALPHA * np.array(mpl.colors.to_rgb(PARETO)) + (1 - PARETO_ALPHA))
LEGEND_TEXT_ALPHA = 0.85               # legend text: darker than the one-trial swatch, for contrast on white
PARETO_TEXT = mpl.colors.to_hex(LEGEND_TEXT_ALPHA * np.array(mpl.colors.to_rgb(PARETO)) + (1 - LEGEND_TEXT_ALPHA))
DOMINATED_TEXT = mpl.colors.to_hex(LEGEND_TEXT_ALPHA * np.array(mpl.colors.to_rgb(DOMINATED)) + (1 - LEGEND_TEXT_ALPHA))
SHADE, SHADE_ALPHA = DOMINATED_LEGEND, 0.085  # region dominated by the front
FRONT_GREY = "#737373"
GRID_GREY = "#DEDEDE"
TICK_GREY = "#4C4C4C"
NOTE_GREY = "#737373"
SPINE_GREY = "#8D8D8D"
LEGEND_TRIANGLE = "#A8A8A8"
INSET_FRAME = "#A8A8A8"


def oklch_diverging(h_neg, h_pos, L_light=0.965, L_dark=0.30, dark_chroma=0.35, chroma_cap=0.115, n=257,
                    name="diverging"):
    """Perceptually uniform diverging colormap built in OKLCH (Ottosson 2020).

    Lightness falls linearly and symmetrically from L_light at 0 to L_dark at both ends, the hue of each
    side is fixed (no drift), and both sides get the same chroma profile, capped at the strongest chroma
    that BOTH hues reach along the whole map without leaving the sRGB gamut, so neither side looks
    stronger. dark_chroma = share of the peak chroma kept at the dark ends; chroma_cap limits the peak
    (0.115 is about the strength of cmocean ice and keeps the perceptual steps even)."""
    M = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                  [-1.2684380046, 2.6097574011, -0.3413193965],
                  [-0.0041960863, -0.7034186147, 1.7076147010]])

    def lin_rgb(L, C, h):
        a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
        lms = np.array([L + 0.3963377774 * a + 0.2158037573 * b,
                        L - 0.1055613458 * a - 0.0638541728 * b,
                        L - 0.0894841775 * a - 1.2914855480 * b]) ** 3
        return M @ lms

    def in_gamut(L, C, h):
        rgb = lin_rgb(L, C, h)
        return np.all(rgb >= -1e-7) and np.all(rgb <= 1 + 1e-7)

    def max_chroma(L, h):
        lo, hi = 0.0, 0.4
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if in_gamut(L, mid, h) else (lo, mid)
        return lo

    def shape(u):   # chroma along one side, u = 0 at the centre (white) to 1 at the end
        rise = np.sin(0.5 * np.pi * np.clip(u / 0.55, 0, 1)) ** 1.3
        return rise * (1 - (1 - dark_chroma) * np.clip((u - 0.55) / 0.45, 0, 1) ** 1.5)

    u = np.linspace(0, 1, 129)
    L = L_light + (L_dark - L_light) * u
    c_max = min([chroma_cap] + [max_chroma(Li, h) / shape(ui) for h in (h_neg, h_pos)
                                for Li, ui in zip(L, u) if shape(ui) > 0.05])
    cols = []
    for x in np.linspace(-1, 1, n):
        ui = abs(x)
        rgb = np.clip(lin_rgb(L_light + (L_dark - L_light) * ui, 0.999 * c_max * shape(ui), h_neg if x < 0 else h_pos), 0, 1)
        cols.append(np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055))
    return mpl.colors.LinearSegmentedColormap.from_list(name, cols, N=n)


# misfit maps: violet (negative) - white - orange (positive), built above; colour-blind safe, orange is the
# complement of the ice-blue speed maps and violet stays outside the hue range of cmocean ice. Positive misfits
# (model too high / too fast) are on the warm side.
DIFF_CMAP, SPEED_CMAP = oklch_diverging(300.0, 55.0, name="violet_orange"), ICE
DIFF_LIM = 100.0                       # misfit colour limits ±DIFF_LIM; white is pinned to 0
LEGEND_DOMINATED_LABEL = "Dominated"

# witness trials: label shown (knee; low / high = lowest / highest τ_ref), colour, marker width (px), marker offset from the point (px, +y = down),
# and label offset of the first line's baseline centre from the point (px)
WITNESS = {
    "high": dict(label="high", color="#1E6B3C", marker_w=15.5, marker_off=(0.0, 0.0), label_off=(14.0, 26.0)),
    "knee": dict(label="knee", color="#1B446F", marker_w=15.5, marker_off=(0.0, 0.0), label_off=(-3.2, 25.5)),
    "low": dict(label="low", color="#98128F", marker_w=17.1, marker_off=(3.2, 4.7), label_off=(-55.0, 25.0)),
}
LABEL_LINE_GAP = 16.5                  # px between the two lines of a witness-trial label

# font sizes in px
FS_TITLE, FS_LABEL, FS_TEXT, FS_SMALL, FS_LEGEND, FS_INSET = 16.0, 15.5, 14.5, 14.0, 14.0, 14.0

# ------------------------------------------------------------------ #
# layout (px on the 980 × 870 canvas, y measured downwards)          #
# ------------------------------------------------------------------ #
CANVAS = (980.0, 870.0)
LAYOUT = {
    "title_a": (31.3, 48.31),                       # left, baseline
    "pareto": (29.973, 72.462, 935.04, 326.4),      # left, top, width, height
    "xlim": (31.6, 37.6), "ylim": (18.40, 19.56), "ygrid_step": 0.1,
    "ytick_x": 33.5, "ytick_dy": -2.49,             # labels sit just above their gridline
    "xtick_baseline": 417.95, "xtick_len": 5.3,
    "xlabel": (420.0, 437.0),                       # centre, baseline
    "ylabel": (23.35, 215.25),                      # baseline x, centre y (rotated)
    "lower_box": (100.31, 311.0, 118.0, 42.0),      # white patch behind the note
    "lower_text": (107.4, 325.72), "lower_arrow": (110.9, 350.0),
    "inset": (684.0, 74.8495, 281.0, 200.0),
    "legend": {                                     # symbol centre, text left edge and baseline
        "witness": dict(sym=(582.0, 432.0), x=594.0, y=437.0),
        "front": dict(sym=(770.0, 790.0, 432.0), x=796.0, y=437.0),
        "pareto": dict(sym=(582.0, 453.0), x=594.0, y=458.0),
        "dominated": dict(sym=(783.0, 453.0), x=796.0, y=458.0),
    },
    "title_b": (30.7, 497.51), "title_c": (517.5, 497.51),
    "run_marker_dx": 10.6, "run_marker_y": 508.6, "run_text_dx": 23.0, "run_text_baseline": 514.4,
    "maps_x": (29.37, 180.09, 330.81, 516.09, 666.81, 817.53), "maps_top": 541.52,
    "map_w": 146.88, "map_h": 251.2, "map_title_baseline": 538.95,
    # top-left corners of the dashed RMSE / bias boxes, by map index (0-2 DEM, 5 speed difference)
    "stats": {0: (59.35, 687.06), 1: (201.74, 685.84), 2: (358.44, 685.04), 5: (856.25, 686.69)},
    "stats_pad": (4.5, 14.0, 16.0, 36.0),           # text inset x, 1st baseline, line gap, box height
    "cbar_top": 800.38, "cbar_h": 13.44,
    "cbar_dem": (29.44, 477.76), "cbar_speed": (516.16, 802.24), "cbar_diff": (829.12, 964.48),
    "cbar_tick_len": 3.3, "cbar_tick_baseline": 829.5, "cbar_label_baseline": 849.0,
}
INSET = {   # inside the inset box, px
    "margins": (52.0, 24.0, 12.0, 42.0),            # left, top, right, bottom
    "zoom_box_min": (16.0, 11.0),                   # the main-panel outline is drawn at least this big
}


# ------------------------------------------------------------------ #
# canvas helper: pixel layout -> matplotlib figure                    #
# ------------------------------------------------------------------ #
class Canvas:
    """Place matplotlib elements in pixel coordinates of a W × H canvas (y down)."""

    def __init__(self, fig, width_px, height_px, pt_per_px):
        self.fig, self.W, self.H, self.k = fig, width_px, height_px, pt_per_px

    def fx(self, x):
        return x / self.W

    def fy(self, y):
        return 1.0 - y / self.H

    def rect(self, x, y, w, h):
        return [x / self.W, 1.0 - (y + h) / self.H, w / self.W, h / self.H]

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes(self.rect(x, y, w, h), **kw)

    def pt(self, px):
        return px * self.k

    def text(self, x, y, s, size, ha="left", va="baseline", **kw):
        return self.fig.text(self.fx(x), self.fy(y), s, fontsize=self.pt(size), ha=ha, va=va, **kw)

    def patch(self, x, y, w, h, **kw):
        r = Rectangle((self.fx(x), self.fy(y + h)), w / self.W, h / self.H, transform=self.fig.transFigure, **kw)
        self.fig.add_artist(r)
        return r

    def line(self, xs, ys, **kw):
        ln = mpl.lines.Line2D([self.fx(x) for x in xs], [self.fy(y) for y in ys],
                              transform=self.fig.transFigure, **kw)
        self.fig.add_artist(ln)
        return ln

    def offset(self, dx_px, dy_px):
        """Translation by (dx, dy) px, for use with + on a transform."""
        return ScaledTranslation(self.pt(dx_px) / 72, -self.pt(dy_px) / 72, self.fig.dpi_scale_trans)


def triangle_ms(width_px, c):
    """Marker size (pt) of a down-pointing equilateral triangle of the given width (px)."""
    return c.pt(width_px) / 0.866


def setup_fonts(font_dir=None):
    dirs = [font_dir] if font_dir else []
    for d in dirs:
        for f in font_manager.findSystemFonts(fontpaths=[d]):
            font_manager.fontManager.addfont(f)
    for f in font_manager.findSystemFonts():
        if os.path.basename(f).startswith("Inter"):
            font_manager.fontManager.addfont(f)
    have_inter = any(f.name == FONT for f in font_manager.fontManager.ttflist)
    mpl.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"] if have_inter else ["DejaVu Sans"],
        "mathtext.fontset": "custom" if have_inter else "dejavusans",
        "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": "DejaVu Sans Mono",
        "axes.unicode_minus": True,
        "svg.fonttype": "none", "pdf.fonttype": 42,     # editable text in vector output
    })
    if not have_inter:
        print("WARNING: font Inter not found (expected in ./fonts), using DejaVu Sans: text will overlap")


# ------------------------------------------------------------------ #
# data                                                               #
# ------------------------------------------------------------------ #
def load_trials(db, study):
    s = optuna.load_study(study_name=study, storage=f"sqlite:///{db}")
    trials = [t for t in s.trials if t.state.name == "COMPLETE" and t.values is not None]
    pts = np.array([t.values for t in trials])
    # only completed trials are used; report what else the study holds
    states = {}
    for t in s.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    distinct = len({tuple(sorted(t.params.items())) for t in trials})
    print(f"study '{study}': {len(s.trials)} trial records {states}; using {len(trials)} completed trials "
          f"({distinct} distinct parameter sets)")
    is_par = np.array([not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)) for p in pts])
    return trials, pts, is_par


def pick_witnesses(trials, is_par):
    """knee: closest to the utopia point in normalised objectives; low / high: lowest / highest τ_ref."""
    pareto = [t for t, p in zip(trials, is_par) if p]
    v = np.array([t.values for t in pareto])
    norm = (v - v.min(0)) / np.maximum(1e-9, v.max(0) - v.min(0))
    return {"knee": pareto[int(np.argmin(norm.sum(1)))],
            "low": min(pareto, key=lambda t: t.params[KEY_TAU]),
            "high": max(pareto, key=lambda t: t.params[KEY_TAU])}


def load_run(run_dir: Path, input_nc: Path):
    with Dataset(run_dir / "output.nc") as nc:
        run = {k: np.asarray(nc.variables[v][:]) for k, v in
               dict(t="time", x="x", y="y", thk="thk", usurf="usurf", uvel="uvelsurf", vvel="vvelsurf").items()}
    with Dataset(input_nc) as nc:   # bed = initial surface − thickness
        run["topg"] = np.squeeze(np.asarray(nc["usurf"][:])) - np.squeeze(np.asarray(nc["thk"][:]))
    return run


def load_obs(past_surf: Path, input_nc: Path):
    with Dataset(past_surf) as nc:
        dems = {y: np.squeeze(np.asarray(nc.variables[f"surf_{y}"][:])) for y in OBS_YEARS}
    with Dataset(input_nc) as nc:
        uo = np.squeeze(np.asarray(nc.variables["uvelsurfobs"][:]))
        vo = np.squeeze(np.asarray(nc.variables["vvelsurfobs"][:]))
    return dems, uo, vo


def nearest(t, target):
    return int(np.argmin(np.abs(t - target)))


def run_scores(run, dems, uo, vo):
    """The two calibration scores of a run: mean RMSE over the DEM dates (1880 = initial state excluded)
    on observed ∪ modelled ice, and the RMSE of the surface speed over valid observed pixels."""
    rmse = []
    for yr in OBS_YEARS[1:]:
        k = nearest(run["t"], yr)
        mask = (run["thk"][k] > THK_THR) | ((dems[yr] - run["topg"]) > THK_THR)
        rmse.append(np.sqrt(np.nanmean((run["usurf"][k] - dems[yr])[mask] ** 2)))
    k = nearest(run["t"], SPEED_YEAR)
    so = np.hypot(uo, vo)
    sm = np.where(run["thk"][k] > THK_THR, np.hypot(run["uvel"][k], run["vvel"][k]), 0.0)
    valid = np.isfinite(so) & (so >= VEL_THR)
    return float(np.mean(rmse)), float(np.sqrt(np.mean((sm - so)[valid] ** 2)))


def identify_run(scores, trials, witnesses):
    """Name, trial number and colour of the trial that the replayed run reproduces."""
    best = min(trials, key=lambda t: np.hypot(t.values[0] - scores[0], t.values[1] - scores[1]))
    if np.hypot(best.values[0] - scores[0], best.values[1] - scores[1]) > 0.05:
        print(f"WARNING: replayed run (DEM {scores[0]:.2f}, speed {scores[1]:.2f}) matches no trial closely")
    for name, t in witnesses.items():
        if t.values == best.values:
            return f"{WITNESS[name]['label']} run (#{best.number})", WITNESS[name]["color"]
    return f"trial #{best.number}", "black"


def staircase(par_pts, xmax, ymax):
    p = np.unique(par_pts, axis=0)
    p = p[np.argsort(p[:, 0])]
    xs, ys = [p[0, 0]], [ymax]
    for (x0, y0), (x1, _) in zip(p[:-1], p[1:]):
        xs += [x0, x1]; ys += [y0, y0]
    xs += [p[-1, 0], xmax]; ys += [p[-1, 1], p[-1, 1]]
    return np.array(xs), np.array(ys)


def blend(color, alpha, bg="white"):
    """Opaque colour of `color` at `alpha` over `bg` (used for the inset background)."""
    c, b = np.array(mpl.colors.to_rgb(color)), np.array(mpl.colors.to_rgb(bg))
    return mpl.colors.to_hex(alpha * c + (1 - alpha) * b)


# ------------------------------------------------------------------ #
# panel (a)                                                          #
# ------------------------------------------------------------------ #
def panel_pareto(c, pts, is_par, witnesses):
    L = LAYOUT
    xlim, ylim = L["xlim"], L["ylim"]
    ax = c.axes(*L["pareto"])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.patch.set_visible(False)

    # gridlines with their values just above the line, inside the chart
    levels = np.arange(np.ceil(ylim[0] / L["ygrid_step"] - 1e-9),
                       np.floor(ylim[1] / L["ygrid_step"] + 1e-9) + 1) * L["ygrid_step"]
    px_per_y = L["pareto"][3] / (ylim[1] - ylim[0])
    for yv in levels:
        ax.axhline(yv, color=GRID_GREY, lw=c.pt(0.8), zorder=0)
        y_px = L["pareto"][1] + (ylim[1] - yv) * px_per_y
        c.text(L["ytick_x"], y_px + L["ytick_dy"], f"{yv:.1f}", FS_TEXT, color=TICK_GREY)

    # dominated region, trials, front
    sx, sy = staircase(pts[is_par], xlim[1], ylim[1])
    ax.fill_between(sx, sy, ylim[1], color=SHADE, alpha=SHADE_ALPHA, lw=0, zorder=1)
    dom = pts[~is_par]
    ax.scatter(dom[:, 0], dom[:, 1], s=c.pt(6.53) ** 2, c=DOMINATED, alpha=DOMINATED_ALPHA, edgecolors="none", zorder=2)
    ax.plot(sx, sy, color=FRONT_GREY, lw=c.pt(1.6), zorder=3, solid_joinstyle="miter")
    par = pts[is_par]     # every trial drawn: repeated trials stack and look darker
    ax.scatter(par[:, 0], par[:, 1], s=c.pt(7.3) ** 2, c=PARETO, alpha=PARETO_ALPHA, edgecolors="none", zorder=4)

    # witness trials: triangle + two-line label (name, τ_ref)
    for name, t in witnesses.items():
        st = WITNESS[name]
        x, y = t.values
        ax.plot(x, y, ls="none", marker=(3, 0, 180), ms=triangle_ms(st["marker_w"], c), mfc=st["color"],
                mec="white", mew=c.pt(1.0), zorder=6,
                transform=ax.transData + c.offset(*st["marker_off"]))
        dx, dy = st["label_off"]
        ax.text(x, y, f"{st['label']} (#{t.number})", fontsize=c.pt(FS_TEXT), color=st["color"], ha="center",
                va="baseline", zorder=7, transform=ax.transData + c.offset(dx, dy))
        # italic, except the subscript "ref" and the unit (upright)
        ax.text(x, y, rf"$\tau_\mathrm{{ref}}$ = {t.params[KEY_TAU]:.2f} $\mathrm{{MPa}}$", fontsize=c.pt(FS_TEXT),
                color=st["color"], style="italic", ha="center", va="baseline", zorder=7,
                transform=ax.transData + c.offset(dx, dy + LABEL_LINE_GAP))

    # x axis: grey base line, black ticks, labels; no y axis line
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_GREY); ax.spines["bottom"].set_linewidth(c.pt(1.4667))
    ax.spines["bottom"].set_zorder(5)
    ticks = np.arange(np.ceil(xlim[0]), np.floor(xlim[1]) + 1)
    ax.set_xticks(ticks); ax.set_yticks([])
    ax.tick_params(axis="x", length=c.pt(L["xtick_len"]), width=c.pt(1.2), color="black", labelbottom=False)
    px_per_x = L["pareto"][2] / (xlim[1] - xlim[0])
    for xv in ticks:
        c.text(L["pareto"][0] + (xv - xlim[0]) * px_per_x, L["xtick_baseline"], f"{xv:g}", FS_TEXT,
               ha="center", color="black")
    c.text(*L["xlabel"], "Surface elevation misfit (m)", FS_LABEL, ha="center")
    c.text(*L["ylabel"], r"Surface speed misfit (m yr$^{-1}$)", FS_LABEL, ha="center", rotation=90,
           rotation_mode="anchor")

    # "Lower is better" note on a white patch
    c.patch(*L["lower_box"], fc="white", ec="none", zorder=1)
    c.text(*L["lower_text"], "Lower is better", FS_TEXT, color=NOTE_GREY, zorder=2)
    c.text(*L["lower_arrow"], "\u2199", 28.0, color=NOTE_GREY, zorder=2)

    c.text(*L["title_a"], f"(a) Pareto front of the calibration ({len(pts)} trials)", FS_TITLE,
           fontweight="bold")
    return ax


def legend_pareto(c, n_par, n_dom):
    lg = LAYOUT["legend"]
    sx, sy = lg["witness"]["sym"]
    mpl_marker(c, sx, sy, marker=(3, 0, 180), ms=triangle_ms(14.0, c), mfc=LEGEND_TRIANGLE, mec="black",
               mew=c.pt(0.75))
    c.text(lg["witness"]["x"], lg["witness"]["y"], "Witness trials", FS_LEGEND, color=NOTE_GREY)
    x0, x1, y = lg["front"]["sym"]
    c.line([x0, x1], [y, y], color=FRONT_GREY, lw=c.pt(1.87))
    c.text(lg["front"]["x"], lg["front"]["y"], "Pareto front", FS_LEGEND, color=NOTE_GREY)
    for key, color, tcolor, txt in (("pareto", PARETO_LEGEND, PARETO_TEXT, f"Pareto-optimal (n = {n_par})"),
                                    ("dominated", DOMINATED_LEGEND, DOMINATED_TEXT, f"{LEGEND_DOMINATED_LABEL} (n = {n_dom})")):
        sx, sy = lg[key]["sym"]
        mpl_marker(c, sx, sy, marker="o", ms=c.pt(11.24), mfc=color, mec="white", mew=c.pt(0.5))
        c.text(lg[key]["x"], lg[key]["y"], txt, FS_LEGEND, color=tcolor)


def mpl_marker(c, x, y, **kw):
    ln = mpl.lines.Line2D([c.fx(x)], [c.fy(y)], ls="none", transform=c.fig.transFigure, **kw)
    c.fig.add_artist(ln)


# ------------------------------------------------------------------ #
# inset: all trials, full range                                      #
# ------------------------------------------------------------------ #
def draw_inset(c, box, pts, is_par, zoom=None, background=None, frame=True, zorder=2):
    """Full-range scatter of all trials inside `box` = (x, y, w, h) px; `zoom` = (xlim, ylim) outlined."""
    bx, by, bw, bh = box
    ml, mt, mr, mb = INSET["margins"]
    if background:
        c.patch(bx, by, bw, bh, fc=background, ec="none", zorder=zorder - 0.5)
    if frame:   # dotted grey outline of the inset
        c.patch(bx, by, bw, bh, fill=False, ec=INSET_FRAME, lw=c.pt(1.0), ls=(0, (1, 2)), zorder=zorder + 1)
    ax = c.axes(bx + ml, by + mt, bw - ml - mr, bh - mt - mb, zorder=zorder)
    ax.set_facecolor("none")
    xmax = np.ceil(pts[:, 0].max() / 50) * 50
    ymax = np.ceil(pts[:, 1].max() / 20) * 20
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.scatter(pts[~is_par, 0], pts[~is_par, 1], s=c.pt(2.65) ** 2, c=DOMINATED, alpha=DOMINATED_ALPHA,
               edgecolors="none", zorder=2)
    ax.scatter(pts[is_par, 0], pts[is_par, 1], s=c.pt(3.0) ** 2, c=PARETO, alpha=PARETO_ALPHA,
               edgecolors="none", zorder=3)

    if zoom is not None:   # outline of the main panel, drawn at least zoom_box_min px so it stays visible
        (zx0, zx1), (zy0, zy1) = zoom
        aw, ah = bw - ml - mr, bh - mt - mb
        min_w, min_h = INSET["zoom_box_min"]
        w = max(zx1 - zx0, min_w * xmax / aw); h = max(zy1 - zy0, min_h * ymax / ah)
        cx, cy = (zx0 + zx1) / 2, (zy0 + zy1) / 2
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="black", lw=c.pt(1.0), zorder=5))
        # key in the title line: swatch + "main panel"
        kx, ky = bx + bw - mr, by + mt - 8.0
        t = c.text(kx, ky, "main panel", FS_INSET, ha="right", color=TICK_GREY, zorder=zorder + 1)
        tw = t.get_window_extent(c.fig.canvas.get_renderer()).width / c.fig.bbox.width * c.W
        c.patch(kx - tw - 15.0, ky - 9.0, 11.0, 9.0, fill=False, ec="black", lw=c.pt(1.0), zorder=zorder + 1)

    ax.set_xticks(np.arange(0, xmax + 1, 100)); ax.set_yticks(np.arange(0, ymax + 1, 40))
    ax.tick_params(labelsize=c.pt(FS_INSET), colors=TICK_GREY, length=c.pt(3.0), width=c.pt(0.8), pad=c.pt(2.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TICK_GREY); ax.spines[side].set_linewidth(c.pt(0.8))
    ax.set_xlabel("Elevation misfit (m)", fontsize=c.pt(FS_INSET), color=NOTE_GREY, labelpad=c.pt(2.0))
    ax.set_ylabel(r"Speed misfit (m yr$^{-1}$)", fontsize=c.pt(FS_INSET), color=NOTE_GREY, labelpad=c.pt(3.0))
    c.text(bx + ml, by + mt - 8.0, f"All {len(pts)} trials", FS_INSET, color=TICK_GREY,
           zorder=zorder + 1)
    return ax


# ------------------------------------------------------------------ #
# panels (b) and (c)                                                 #
# ------------------------------------------------------------------ #
def diff_norm():
    """Misfit colour scale: ±DIFF_LIM with the colormap centre (white) pinned to 0."""
    return mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-DIFF_LIM, vmax=DIFF_LIM)


def stats_box(c, x, y, values):
    pad_x, base1, gap, box_h = LAYOUT["stats_pad"]
    t1 = c.text(x + pad_x, y + base1, f"RMSE {np.sqrt(np.mean(values ** 2)):.1f}", FS_SMALL, zorder=3)
    t2 = c.text(x + pad_x, y + base1 + gap, f"bias {np.mean(values):+.1f}", FS_SMALL, zorder=3)
    r = c.fig.canvas.get_renderer()
    w_px = max(t.get_window_extent(r).width for t in (t1, t2)) / c.fig.bbox.width * c.W
    c.patch(x, y, w_px + 2 * pad_x, box_h, fill=False, ec="black", lw=c.pt(1.0), ls=(0, (2, 2)), zorder=3)


def cbar(c, mappable, x0, x1, ticks, label):
    L = LAYOUT
    cax = c.axes(x0, L["cbar_top"], x1 - x0, L["cbar_h"])
    cb = c.fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cax.tick_params(length=c.pt(L["cbar_tick_len"]), width=c.pt(1.07), color="black", labelbottom=False)
    vmin, vmax = mappable.get_clim()
    for tv in ticks:
        xp = x0 + (tv - vmin) / (vmax - vmin) * (x1 - x0)
        c.text(xp, L["cbar_tick_baseline"], f"{tv:g}".replace("-", "\u2212"), FS_SMALL, ha="center")
    c.text((x0 + x1) / 2, L["cbar_label_baseline"], label, FS_TEXT, ha="center")


def run_label(c, x_panel, label, color):
    L = LAYOUT
    mpl_marker(c, x_panel + L["run_marker_dx"], L["run_marker_y"], marker=(3, 0, 180),
               ms=triangle_ms(15.5, c), mfc=color, mec="white", mew=c.pt(1.0))
    c.text(x_panel + L["run_text_dx"], L["run_text_baseline"], label, FS_LABEL, color=color)


def panel_maps(c, run, dems, uo, vo, label, label_color):
    L = LAYOUT
    extent = (run["x"].min(), run["x"].max(), run["y"].min(), run["y"].max())
    axes = []
    for i, x in enumerate(L["maps_x"]):
        ax = c.axes(x, L["maps_top"], L["map_w"], L["map_h"])
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        axes.append(ax)

    # (b) DEM misfit on observed ∪ modelled ice
    for i, yr in enumerate(MAP_YEARS):
        k = nearest(run["t"], yr)
        ice = run["thk"][k] > THK_THR
        mask = ice | ((dems[yr] - run["topg"]) > THK_THR)
        diff = np.where(mask, run["usurf"][k] - dems[yr], np.nan)
        im_dem = axes[i].imshow(diff, origin="lower", extent=extent, cmap=DIFF_CMAP, norm=diff_norm(),
                                interpolation="nearest")
        axes[i].contour(ice, levels=[0.5], extent=extent, colors="k", linewidths=c.pt(0.667), alpha=0.7)
        c.text(L["maps_x"][i] + L["map_w"] / 2, L["map_title_baseline"], f"{yr}", FS_TEXT, ha="center")
        stats_box(c, *L["stats"][i], diff[np.isfinite(diff)])

    # (c) speed on all valid observed pixels; modelled speed 0 where the modelled ice is gone
    k = nearest(run["t"], SPEED_YEAR)
    ice = run["thk"][k] > THK_THR
    so = np.hypot(uo, vo)
    sm = np.where(ice, np.hypot(run["uvel"][k], run["vvel"][k]), 0.0)
    valid = np.isfinite(uo) & np.isfinite(vo) & (so >= VEL_THR)
    dv = np.where(valid, sm - so, np.nan)
    vmax = max(50.0, np.ceil(float(np.nanpercentile(so[valid], 99)) / 10.0) * 10.0)
    speed_norm = mpl.colors.Normalize(0, vmax)
    fields = [(np.where(valid, so, np.nan), SPEED_CMAP, speed_norm, "w", "observed"),
              (np.where(valid, sm, np.nan), SPEED_CMAP, speed_norm, "w", "modelled"),
              (dv, DIFF_CMAP, diff_norm(), "k", "model \u2212 observed")]
    ims = []
    for j, (f, cmap, norm, ccol, title) in enumerate(fields):
        ax = axes[3 + j]
        ims.append(ax.imshow(f, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest"))
        ax.contour(ice, levels=[0.5], extent=extent, colors=ccol, linewidths=c.pt(0.667), alpha=0.7)
        c.text(L["maps_x"][3 + j] + L["map_w"] / 2, L["map_title_baseline"], title, FS_TEXT, ha="center")
    stats_box(c, *L["stats"][5], dv[np.isfinite(dv)])

    cbar(c, im_dem, *L["cbar_dem"], np.arange(-DIFF_LIM, DIFF_LIM + 1, DIFF_LIM / 4), "Surface elevation misfit (m)")
    cbar(c, ims[0], *L["cbar_speed"], np.arange(0, vmax, 25), r"Surface speed (m yr$^{-1}$)")
    cbar(c, ims[2], *L["cbar_diff"], [-DIFF_LIM, 0, DIFF_LIM], r"Speed misfit (m yr$^{-1}$)")

    c.text(*L["title_b"], "(b) Surface elevation misfit", FS_TITLE, fontweight="bold")
    c.text(*L["title_c"], f"(c) Surface speed misfit {SPEED_YEAR:.0f}", FS_TITLE, fontweight="bold")
    run_label(c, L["maps_x"][0], label, label_color)
    run_label(c, L["maps_x"][3], label, label_color)


# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optuna-db", default="data/optuna.db")
    p.add_argument("--study", default="aletsch")
    p.add_argument("--replay", default="data/knee_run", help="run directory with output.nc")
    p.add_argument("--past-surf", default="data/past_surf.nc")
    p.add_argument("--input-nc", default="data/input.nc")
    p.add_argument("--out", default="aletsch_pareto", help="output file name without extension")
    p.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    p.add_argument("--font-dir", default="fonts", help="folder with the Inter .ttf files (default: ./fonts)")
    return p.parse_args()


def main():
    os.chdir(Path(__file__).resolve().parent)   # data paths are relative to this folder
    args = parse_args()
    setup_fonts(args.font_dir)

    trials, pts, is_par = load_trials(args.optuna_db, args.study)
    witnesses = pick_witnesses(trials, is_par)
    run = load_run(Path(args.replay), Path(args.input_nc))
    dems, uo, vo = load_obs(Path(args.past_surf), Path(args.input_nc))
    label, label_color = identify_run(run_scores(run, dems, uo, vo), trials, witnesses)

    W, H = CANVAS
    fig = plt.figure(figsize=(W / 96, H / 96), facecolor="white")   # 1 px = 1/96 in = 0.75 pt
    c = Canvas(fig, W, H, pt_per_px=0.75)
    panel_pareto(c, pts, is_par, witnesses)
    legend_pareto(c, int(is_par.sum()), int((~is_par).sum()))
    draw_inset(c, LAYOUT["inset"], pts, is_par, zoom=(LAYOUT["xlim"], LAYOUT["ylim"]),
               background=blend(SHADE, SHADE_ALPHA))
    panel_maps(c, run, dems, uo, vo, label, label_color)

    for fmt in args.formats:
        out = Path(args.out).with_suffix(f".{fmt}")
        fig.savefig(out, dpi=192 if fmt == "png" else 300, facecolor="white")
        print("saved", out.resolve())
    for name, t in witnesses.items():
        print(f"{name:5s}: #{t.number}  DEM {t.values[0]:.2f} m, speed {t.values[1]:.2f} m/yr, "
              f"tau_ref {t.params[KEY_TAU]:.3f} MPa")


if __name__ == "__main__":
    sys.exit(main())

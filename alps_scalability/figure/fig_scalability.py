#!/usr/bin/env python3
"""
Scalability test of the IGM GMD paper: ONE double-column figure -> fig_scalability.{pdf,png,svg}

  left : offline surface speed at t = 500 yr, dx = 200 m, over a soft hillshade, lat/lon graticule,
         locator inset, colour bar, scale bar, circular close-up
  right: GPU memory (top), run time (middle) and L2 error of the depth-averaged speed (bottom) vs grid
         spacing, sharing the grid-spacing axis; online and offline (pretrained network), one RTX 4090

  python fig_scalability.py                (from anywhere)
  needs: numpy, matplotlib, netCDF4, cartopy, cmocean, shapely. Font "Inter" is loaded from fonts/
         (falls back to DejaVu Sans). cartopy fetches Natural Earth borders on first use.

Input data (read only):
  data/rtx4090_local/scaling*.csv        online sweep        (status, ratio, cells, dx_m, n_iter, iceflow_mean_s,
  data/rtx4090_offline/scaling*.csv      offline sweep        tf_peak_mib)
  data/rtx4090_local_nz10/scaling*.csv   online sweep, Nz = 10 Lagrange (cost and memory only)
  data/isoL_b1/accuracy.csv              accuracy runs, online  (dx_m, time_yr, l2_velbar)
  data/warm_local_offline/accuracy.csv   accuracy runs, offline
  data/map_offline_200m_t500.nc          200 m offline run at t = 500 yr (x, y, time, thk, usurf, velsurf_mag)
Plotted: run time = n_iter x iceflow_mean_s (ice-flow solver time of the whole run, s);
         memory = tf_peak_mib / 1024 (GiB); L2 error = l2_velbar at t = 500 yr (m/yr), online and offline.

Everything you are likely to tweak is in the SETTINGS block. The SVG opens as a few named layers (see SVG_LAYERS).
"""
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.image
import matplotlib.patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.ticker as mticker
import netCDF4
import numpy as np
import cmocean  # noqa: F401  (registers the "cmo.*" colormaps)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import LightSource, LogNorm, to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D

# =============================================================================================== SETTINGS
B = Path(__file__).resolve().parent                 # this folder (figure/); data made by ../runs/tools/make_figure_data.py
ONLINE = B / "data/rtx4090_local"
OFFLINE = B / "data/rtx4090_offline"
ONLINE_NZ10 = B / "data/rtx4090_local_nz10"           # online sweep with Nz = 10 (Lagrange): cost and memory only
ACC_ONLINE = B / "data/isoL_b1"                       # accuracy runs on the sub-domain (L2 error row)
ACC_OFFLINE = B / "data/warm_local_offline"
MAP_NC = B / "data/map_offline_200m_t500.nc"
OUT = B
NAME = "fig_scalability"
FORMATS = ("pdf", "png", "svg")
GRID_CRS = ccrs.UTM(32)                              # model grid: UTM 32N (EPSG:32632)
LL = ccrs.PlateCarree()

# --- page layout (inches). Width is fixed; the height follows from the two plot panels, and the map
#     extent (CROP) is fitted to the space left of them.
FIG_WIDTH_CM = 17.5                     # double column
TITLE_BAND = 0.34                       # band at the top for the panel titles a) and b)
PANEL_TITLES = ("a) Resolution scalability using an Alps-wide simulation",
                "b) Performance scalability\n(GPU memory, run time and accuracy)")
FS_TITLE = 8.0                          # points, bold
MAP_MARGIN = dict(left=0.48, top=0.24, bottom=0.24)   # room for the lat / lon labels around the map
GAP_MAP_PLOTS = 0.66                    # map right edge -> plot axes (y tick labels + y label live here)
PLOT_WIDTH = 1.95                       # width of the plot axes
RIGHT_MARGIN = 0.14
PLOT_TOP, PLOT_GAP, PLOT_BOTTOM = 0.03, 0.13, 0.42   # above memory | between panels | under the last panel
PANEL_HEIGHT = 1.20                     # height of the memory and run-time axes
ERROR_HEIGHT = 0.95                     # height of the accuracy axes (bottom row)
YLABEL_X = -0.16                        # y-label position of the three panels, in axes widths (left of the axis)

# --- map extent: fixed height and centre (km, UTM 32N); the width is set by the layout above
CROP_CENTER_KM = (602.5, 5179.0)
CROP_HEIGHT_KM = 188.0
T_MAP = 500.0                           # yr, time slice shown

# --- fonts and line weights (points)
FONT = "Inter"                          # DejaVu Sans if not installed
FS, FS_LABEL = 7.0, 7.0                 # ticks/annotations, axis labels (>= 7 pt at print size, GMD)
LW_DATA, MS_DATA = 1.0, 2.9             # data lines, marker size
LW_AXIS = 0.5

# --- colours
ONLINE_COLOR = "#3178cb"                # cobalt blue   (CIELAB L* 50, hue 278)
OFFLINE_COLOR = "#e2662c"               # burnt orange  (CIELAB L* 58, hue 50)
OUT_OPACITY = 0.54                      # outside the training distribution: offline colour at this
                                        # opacity over white (flattened -> no transparency in the SVG)
INK, INK2 = "#15191d", "#525b63"
GRID_MAJOR, GRID_MINOR = "#cbcac5", "#cbcac5"       # plot grid
GRID_MINOR_ALPHA = 0.27
BOX = "#e4572e"                                      # map extent box in the inset
SEA, BORDER = "#dde8ef", "#4a4f55"

# --- map content
SPEED_RANGE = (1, 1000)                 # m/yr, log colour scale
SPEED_CMAP = "cmo.ice"
THK_MIN = 1.0                           # m, speed shown where thk > THK_MIN
GRID_STEP = 0.5                         # degrees between graticule lines
GRATICULE = dict(color="#3c434a", linewidth=0.3, alpha=0.36)
GRATICULE_NODES = dict(marker="D", markersize=2.0, alpha=0.55, markeredgewidth=0)
HS_AZIMUTHS, HS_ALTITUDE, HS_VERT_EXAG = (270, 315, 360), 50, 1.0
HS_RANGE = (0.55, 1.0)                  # grey levels of the hillshade (raise the first = lighter)
SCALEBAR_KM = 50.0
INSET_SUBSAMPLE = 4
COUNTRIES = (("FR", 5.2, 47.6), ("CH", 7.6, 47.35), ("DE", 10.5, 48.5), ("AT", 14.6, 48.1),
             ("IT", 11.0, 44.6), ("SI", 14.7, 46.1), ("HR", 15.3, 45.3))
ZOOM_CENTER_KM = (643.0, 5229.0)        # close-up: centre and radius of the area magnified
ZOOM_RADIUS_KM = 7.0
ZOOM_EDGE = "#2b3136"
ZOOM_LABEL_XY = (0.5, 0.5)              # "×4" in the close-up, axes fraction (centre = on the glacier)

# --- map overlays, in layout units ("u", MAP_U u = map width), measured from the map's top-left corner,
#     so they scale with the map
MAP_U = 518.47
INSET_BOX = (4.98, 4.98, 160.00)        # x, y, width (height from the domain shape)
KEY_TOP = 4.98                          # top of the inset, colour bar and scale bar (aligned)
CBAR_BOX = (184.00, KEY_TOP, 150.00, 6.64)  # colour bar x, y, w, h
SCALEBAR_X = 358.00                     # left x of the scale bar; its top is KEY_TOP and its end ticks are as
                                        # long as the colour bar is thick. "50 km" sits under it, on the row
                                        # of the colour-bar tick labels; "Δx = 200 m" on the row of the colour-bar title
DX_LABEL_RIGHT = 4.98                   # "Δx = 200 m": right edge this far from the map's right edge
MAP_KEY_HALO = 1.8                      # points, white stroke around the scale bar and both labels
ZOOM_BOX = (447.09, 343.40, 66.40)      # close-up centre x, y and radius

# --- plots
DX_MAX_OFF = 600.0                      # offline shown from 600 m only
DX_TRAIN = 250.0                        # offline network trained on 50-250 m
CARD_GIB = 24.0 * 1e9 / 2**30           # 24 GB card = 22.4 GiB
XLIM = (6600, 190)
XTICKS = (6000, 3000, 1200, 600, 300, 200)
YLIM = dict(memory=(0.1, 60), time=(1, 8000), error=(1, 100))
NZ10_COLOR = "#7d4fb0"                  # online, Nz = 10 (Lagrange): cost and memory only
LEGEND_PANEL, LEGEND_LOC, LEGEND_POS = "memory", "upper left", (0.0, 0.86)   # one key (online / offline / Nz = 10)
OUT_SIDE = dict(memory=+1, time=-1)     # bracket of the faded segment: +1 below-right, -1 above-left
OUT_LABEL_XY = (1050, 2300)             # "outside training distribution": centre, run-time data coordinates
OUT_LEADER = dict(memory=0.35, time=0.12)   # curvature of the dashed leaders to each bracket (arc3 rad)
OUT_LEADER_STYLE = dict(lw=0.6, dashes=(2.4, 1.6), halo=1.8)
OUT_BRACKET_OFFSET, OUT_BRACKET_TICK = 6.5, 3.2               # points
GUIDE_SHIFT = 0.40                      # dx^-3 guide = finest online point x this
GUIDE_START = 1000.0                    # m, where the dx^-3 regime starts (bracket)
GUIDE_TAIL_TO, GUIDE_TAIL_ALPHA = 200.0, 0.4   # faded dotted continuation
GUIDE_LABEL_OFFSET = 5.5                # points, label beside the middle of the guide
BRACKET_SIZE = 5.5                      # points, "[" at the start of the guide

# =============================================================================================== style
for _f in sorted((OUT / "fonts").glob("*.ttf")):     # Inter shipped with the figure (fonts/)
    fm.fontManager.addfont(str(_f))
_font = FONT if any(f.name == FONT for f in fm.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": _font, "font.size": FS, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS_LABEL,
    "mathtext.fontset": "custom", "mathtext.rm": _font, "mathtext.it": f"{_font}:italic",
    "mathtext.bf": f"{_font}:bold", "mathtext.cal": _font, "mathtext.fallback": "stixsans",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": LW_AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
    "xtick.minor.width": LW_AXIS * 0.66, "ytick.minor.width": LW_AXIS * 0.66,
    "xtick.major.size": 2.7, "ytick.major.size": 2.7, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
    "lines.linewidth": LW_DATA, "lines.markersize": MS_DATA, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "svg.fonttype": "none", "svg.hashsalt": "igm",     # editable text, stable ids
})


def fade(color, opacity=OUT_OPACITY):
    """color at `opacity` over white, as a solid colour."""
    return to_hex(1 - opacity * (1 - np.array(to_rgb(color))))


OUT_COLOR = fade(OFFLINE_COLOR)

# =============================================================================================== svg
# SVG layers, bottom -> top. Artists get gid "<layer>--<name>"; tidy_svg() groups them by layer.
SVG_LAYERS = ("background", "grid", "hillshade", "speed", "features", "graticule", "frame", "labels",
              "reference", "guide", "outside", "offline", "online", "box", "legend",
              "map", "inset", "colorbar", "scalebar", "zoom", "memory", "time", "leaders", "titles")
AXES_GIDS = ("map", "inset", "colorbar", "zoom", "memory", "time")
SVGNS, XLINK = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"


def tidy_svg(path):
    """Regroup a matplotlib SVG into named layers (SVG_LAYERS), data near the front; no stray clip boxes."""
    ET.register_namespace("", SVGNS); ET.register_namespace("xlink", XLINK)
    tree = ET.parse(path)
    is_g = lambda e: e.tag == f"{{{SVGNS}}}g"
    fig = next(e for e in tree.getroot().iter() if is_g(e) and e.get("id") == "figure_1")

    def layer(i):
        if "--" in i:
            return i.split("--")[0]
        for pre, key in (("matplotlib.axis", "frame"), ("patch_", "frame"), ("legend", "legend"),
                         ("text_", "labels"), ("LineCollection", "graticule"), ("line2d", "lines"),
                         ("image", "images"), ("PathCollection", "lines")):
            if i.startswith(pre):
                return key
        return i

    def regroup(cont):
        buckets = {}
        for ch in list(cont):
            cont.remove(ch)
            i = ch.get("id", "")
            if not i:                      # unnamed wrapper (e.g. a clip group): use what it contains
                i = next((e.get("id") for e in ch.iter() if e.get("id")), "")
            buckets.setdefault(layer(i), []).append(ch)
            if is_g(ch) and ch.get("id") and (i.startswith("axes_") or i in AXES_GIDS
                                              or i.split("--")[0] in AXES_GIDS):
                regroup(ch)
        for k in [k for k in SVG_LAYERS if k in buckets] + [k for k in buckets if k not in SVG_LAYERS]:
            kids = buckets[k]
            if len(kids) == 1 and kids[0].get("id") == k:
                cont.append(kids[0])
            else:
                g = ET.SubElement(cont, f"{{{SVGNS}}}g", id=k)
                g.extend(kids)

    regroup(fig)
    for e in tree.getroot().iter(f"{{{SVGNS}}}image"):     # no whitespace inside base64
        for att in (f"{{{XLINK}}}href", "href"):
            if e.get(att, "").startswith("data:"):
                e.set(att, "".join(e.get(att).split()))
    tree.write(path, xml_declaration=True, encoding="utf-8")


def save(fig, name):
    fig.patch.set_facecolor("white"); fig.patch.set_gid("background--white")
    for a in fig.axes:
        a.patch.set_gid("background--axes")
    for ext in FORMATS:
        fig.savefig(OUT / f"{name}.{ext}")
        if ext == "svg":
            tidy_svg(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / name, "(" + ", ".join(FORMATS) + ")")


def gid(artists, name):
    artists = artists if isinstance(artists, (list, tuple)) else [artists]
    for n, a in enumerate(artists):
        a.set_gid(name if len(artists) == 1 else f"{name}-{n + 1}")
    return artists


# =============================================================================================== plots
def load_sweep(d):
    """Successful grids of a sweep, sorted by number of cells: dx (m), run time (s), memory (GiB)."""
    rows = {}
    for path in sorted(d.glob("scaling*.csv")):
        for r in csv.DictReader(open(path)):
            if r["status"] == "ok" and math.isfinite(float(r["iceflow_mean_s"] or "nan")):
                rows[r["ratio"]] = r
    rows = sorted(rows.values(), key=lambda r: float(r["cells"]))
    get = lambda k: np.array([float(r[k]) for r in rows])
    return get("dx_m"), get("n_iter") * get("iceflow_mean_s"), get("tf_peak_mib") / 1024


def load_accuracy(d, t=500.0):
    """Accuracy runs on the sub-domain: dx (m) and L2 error of the depth-averaged speed (m/yr) at time t."""
    rows = [r for r in csv.DictReader(open(d / "accuracy.csv")) if float(r["time_yr"]) == t]
    rows = sorted(rows, key=lambda r: float(r["dx_m"]), reverse=True)
    return np.array([float(r["dx_m"]) for r in rows]), np.array([float(r["l2_velbar"]) for r in rows])


def load_both():
    out = {}
    for name, d in (("online", ONLINE), ("offline", OFFLINE), ("nz10", ONLINE_NZ10)):
        if name == "nz10" and not list(d.glob("scaling*.csv")):
            continue
        dx, t, m = load_sweep(d)
        keep = dx <= DX_MAX_OFF if name == "offline" else np.ones_like(dx, bool)
        out[name] = dict(dx=dx[keep], time=t[keep], memory=m[keep])
    for name, d in (("online", ACC_ONLINE), ("offline", ACC_OFFLINE)):
        out[name]["dx_error"], out[name]["error"] = load_accuracy(d)
    return out


def to_px(ax, x, y):
    return ax.transData.transform(np.c_[np.atleast_1d(x), np.atleast_1d(y)])


def to_data(ax, P):
    return ax.transData.inverted().transform(P)


def plot_offline(ax, dx, y, kind):
    """Offline curve: full colour inside the training distribution, faded (flattened) outside it; the two
    parts meet at DX_TRAIN. A bracket parallel to the faded part marks it, with its label."""
    out = dx > DX_TRAIN
    j = np.where(out)[0].max()
    f = np.log(DX_TRAIN / dx[j]) / np.log(dx[j + 1] / dx[j])
    y_t = np.exp(np.log(y[j]) + f * (np.log(y[j + 1]) - np.log(y[j])))
    xo, yo = np.r_[dx[out], DX_TRAIN], np.r_[y[out], y_t]
    gid(ax.plot(xo, yo, "-", color=OUT_COLOR, zorder=4), "outside--line")
    gid(ax.plot(dx[out], y[out], "o", color=OUT_COLOR, mew=0, zorder=4.5), "outside--points")
    gid(ax.plot(np.r_[DX_TRAIN, dx[j + 1:]], np.r_[y_t, y[j + 1:]], "-o", color=OFFLINE_COLOR, mew=0,
                markevery=list(range(1, len(dx) - j)), zorder=5), "offline--curve")

    # bracket "[____]" parallel to the faded segment, offset to one side, end ticks pointing at the line
    P0, P1 = to_px(ax, xo[[0, -1]], yo[[0, -1]])
    d = (P1 - P0) / np.hypot(*(P1 - P0))
    n = np.array([d[1], -d[0]]) * OUT_SIDE[kind]                     # +1: below-right of a rising line
    k = ax.figure.dpi / 72
    A, Bp = P0 + n * OUT_BRACKET_OFFSET * k, P1 + n * OUT_BRACKET_OFFSET * k
    t = -n * OUT_BRACKET_TICK * k
    xb, yb = to_data(ax, np.array([A + t, A, Bp, Bp + t])).T
    gid(ax.plot(xb, yb, "-", color=OUT_COLOR, lw=LW_DATA, solid_joinstyle="miter", zorder=4,
                clip_on=False), "outside--bracket")
    mid = (A + Bp) / 2                                               # bracket midpoint (display px)
    return ax.figure.transFigure.inverted().transform(mid)          # -> figure fraction


def bracket_marker(ax, x, y, ang, color, name):
    """'[' end cap across a line of page direction ang (deg), serifs pointing along the line."""
    m = MarkerStyle(MPath([(0.45, 1), (0, 1), (0, -1), (0.45, -1)], [MPath.MOVETO] + [MPath.LINETO] * 3),
                    fillstyle="none", transform=Affine2D().rotate_deg(ang))
    gid(ax.plot([x], [y], ls="none", marker=m, ms=BRACKET_SIZE, mec=color, mew=0.9, zorder=6,
                clip_on=False), name)


def dx3_guide(ax, dx, y):
    """Grey dx^-3 line parallel to the online curve: '[' where the regime starts, faded dotted
    continuation beyond the finest online point, label beside its middle."""
    k = np.argmin(dx)
    law = lambda xx: GUIDE_SHIFT * y[k] * (np.asarray(xx, float) / dx[k]) ** -3
    gx = np.array([GUIDE_START, dx[k]]); gy = law(gx)
    gid(ax.plot(gx, gy, "-", color=INK2, lw=0.8, zorder=2, clip_on=False), "guide--line")
    p0, p1 = to_px(ax, gx, gy)
    ang = np.degrees(np.arctan2(*(p1 - p0)[::-1]))
    bracket_marker(ax, gx[0], gy[0], ang, INK2, "guide--start")
    if GUIDE_TAIL_TO:
        tx = np.array([dx[k], GUIDE_TAIL_TO])
        gid(ax.plot(tx, law(tx), ":", color=INK2, lw=0.8, alpha=GUIDE_TAIL_ALPHA, zorder=2, clip_on=False),
            "guide--continuation")
    up = (ang + 90) % 180 - 90
    a = np.radians(up); nrm = np.array([np.sin(a), -np.cos(a)])
    xm = np.sqrt(gx[0] * gx[1])
    gid(ax.annotate("$\\propto \\Delta x^{-3}$", xy=(xm, law(xm)), xytext=GUIDE_LABEL_OFFSET * nrm,
                    textcoords="offset points", rotation=up, rotation_mode="anchor", ha="center",
                    va="center", color=INK2, fontsize=FS, zorder=3), "guide--label")


def plot_panel(ax, kind, data, show_x):
    on, off = data["online"], data["offline"]
    ylim = YLIM[kind]
    ax.set_gid(kind)
    ax.patch.set_visible(False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(*ylim)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if show_x:
        ax.set_xlabel("grid spacing $\\Delta x$ (m)")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(dict(memory="GPU memory (GiB)", time="Run time (s)", error="L2 error (m yr$^{-1}$)")[kind])
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)                          # same x for the three rows

    e = np.arange(np.floor(np.log10(ylim[0])), np.ceil(np.log10(ylim[1])) + 1)
    major = [10 ** v for v in e if ylim[0] < 10 ** v < ylim[1]]
    minor = [m * 10 ** v for v in e for m in range(2, 10) if ylim[0] < m * 10 ** v < ylim[1]]
    gid(ax.hlines(minor, *XLIM, colors=GRID_MINOR, lw=0.4, alpha=GRID_MINOR_ALPHA, zorder=0.5), "grid--minor")
    gid(ax.hlines(major, *XLIM, colors=GRID_MAJOR, lw=0.5, zorder=0.6), "grid--major")

    if kind == "error":                                               # accuracy runs: online / offline at 3 grids
        gid(ax.plot(on["dx_error"], on["error"], "-o", color=ONLINE_COLOR, mew=0, zorder=6), "online--error")
        gid(ax.plot(off["dx_error"], off["error"], "-o", color=OUT_COLOR, mew=0, zorder=5), "outside--error")
        for a in ax.get_children():
            a.set_clip_on(False)
        return None
    gid(ax.plot(on["dx"], on[kind], "-o", color=ONLINE_COLOR, mew=0, zorder=6), "online--curve")
    if "nz10" in data:
        gid(ax.plot(data["nz10"]["dx"], data["nz10"][kind], "-o", color=NZ10_COLOR, mew=0, zorder=5.5), "online--nz10")
    bracket_mid = plot_offline(ax, off["dx"], off[kind], kind)

    if kind == "memory":
        gid(ax.axhline(CARD_GIB, color=INK2, lw=0.8, ls=":", zorder=1), "reference--gpu-line")
        gid(ax.annotate("24 GB GPU", xy=(XLIM[0], CARD_GIB), xytext=(2, 1.5), textcoords="offset points",
                        ha="left", va="bottom", color=INK2, fontsize=FS * 0.95), "reference--gpu-label")
        order = ("online", "offline")
    else:
        dx3_guide(ax, on["dx"], on["time"])
        order = ("offline", "online")

    if kind == LEGEND_PANEL:
        col = dict(online=ONLINE_COLOR, offline=OFFLINE_COLOR, nz10=NZ10_COLOR)
        names = dict(online="online", offline="offline", nz10="online, $N_z=10$")
        order = order + (("nz10",) if "nz10" in data else ())
        handles = [Line2D([], [], color=col[o], marker="o", mew=0) for o in order]
        leg = ax.legend(handles, [names[o] for o in order], loc=LEGEND_LOC, bbox_to_anchor=LEGEND_POS, handlelength=2.0,
                        handletextpad=0.6, borderaxespad=0.2, labelspacing=0.3,
                        labelcolor=[col[o] for o in order])
        leg.set_gid("legend")
    for a in ax.get_children():                                       # nothing here needs clipping
        a.set_clip_on(False)
    return bracket_mid


# =============================================================================================== map
def soft_hillshade(z, dx):
    """Multi-directional hillshade on HS_RANGE (rows run south -> north, LightSource wants north first)."""
    zn = np.flipud(np.nan_to_num(z, nan=float(np.nanmin(z))))
    hs = np.mean([LightSource(azdeg=a, altdeg=HS_ALTITUDE).hillshade(zn, vert_exag=HS_VERT_EXAG, dx=dx, dy=dx)
                  for a in HS_AZIMUTHS], axis=0)
    return np.flipud(HS_RANGE[0] + (HS_RANGE[1] - HS_RANGE[0]) * hs)


def _segments(geom):
    if geom.is_empty:
        return []
    t = geom.geom_type
    if t in ("LineString", "LinearRing"):
        return [np.asarray(geom.coords)]
    if t == "Polygon":
        return [np.asarray(geom.exterior.coords)] + [np.asarray(r.coords) for r in geom.interiors]
    return [seg for g in getattr(geom, "geoms", []) for seg in _segments(g)]


def _lonlat_extent(box):
    c = LL.transform_points(GRID_CRS, np.array(box[:2] * 2), np.array([box[2]] * 2 + [box[3]] * 2))
    return c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()


def unclipped_feature(ax, feature, box, name, zorder, polygons=False, **style):
    """Natural Earth feature projected and cut to box here: ONE collection, no clip path in the SVG."""
    from shapely.geometry import box as sbox
    from matplotlib.collections import LineCollection, PathCollection
    Bx = sbox(box[0], box[2], box[1], box[3])
    lo0, lo1, la0, la1 = _lonlat_extent(box)
    geoms = [GRID_CRS.project_geometry(g, feature.crs).intersection(Bx)
             for g in feature.intersecting_geometries([lo0 - 1, lo1 + 1, la0 - 1, la1 + 1])]
    if polygons:
        polys = [p for g in geoms for p in (getattr(g, "geoms", None) or [g]) if p.geom_type == "Polygon"]
        paths = [MPath.make_compound_path(*[MPath(np.asarray(r.coords)) for r in [p.exterior, *p.interiors]])
                 for p in polys if not p.is_empty]
        art = PathCollection(paths, zorder=zorder, **style)
    else:
        art = LineCollection([sg for g in geoms for sg in _segments(g)], zorder=zorder, **style)
    art.set_clip_on(False); art.set_gid(name)
    ax.add_collection(art, autolim=False)


def graticule_lines(ax, box, step, name, zorder, **style):
    """Lat/lon lines cut to box here: one LineCollection without clip path."""
    from shapely.geometry import LineString, box as sbox
    from matplotlib.collections import LineCollection
    Bx = sbox(box[0], box[2], box[1], box[3])
    lo0, lo1, la0, la1 = _lonlat_extent(box)
    lo = np.arange(np.floor(lo0 / step) - 1, np.ceil(lo1 / step) + 2) * step
    la = np.arange(np.floor(la0 / step) - 1, np.ceil(la1 / step) + 2) * step
    segs = []
    for v in lo:
        rng = np.linspace(la[0], la[-1], 200)
        segs += _segments(LineString(GRID_CRS.transform_points(LL, np.full_like(rng, v), rng)[:, :2])
                          .intersection(Bx))
    for v in la:
        rng = np.linspace(lo[0], lo[-1], 200)
        segs += _segments(LineString(GRID_CRS.transform_points(LL, rng, np.full_like(rng, v))[:, :2])
                          .intersection(Bx))
    art = LineCollection(segs, zorder=zorder, colors=style["color"], linewidths=style["linewidth"],
                         alpha=style["alpha"])
    art.set_clip_on(False); art.set_gid(name)
    ax.add_collection(art, autolim=False)
    # diamonds at the intersections
    LO, LA = np.meshgrid(lo, la)
    P = GRID_CRS.transform_points(LL, LO.ravel(), LA.ravel())
    ins = (P[:, 0] > box[0]) & (P[:, 0] < box[1]) & (P[:, 1] > box[2]) & (P[:, 1] < box[3])
    gid(ax.plot(P[ins, 0], P[ins, 1], ls="none", color=style["color"], transform=GRID_CRS, zorder=zorder + 0.1,
                clip_on=False, **GRATICULE_NODES), "graticule--nodes")


def edges(c, d):
    return [c[0] - d / 2, c[-1] + d / 2]


def ext_tangents(c1, r1, c2, r2):
    """End points of the two outer tangents between circles (c1, r1) and (c2, r2)."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = np.hypot(*(c2 - c1)); th = np.arctan2(*(c2 - c1)[::-1]); b = np.arccos((r1 - r2) / d)
    return [(c1 + r1 * np.array([np.cos(th + s * b), np.sin(th + s * b)]),
             c2 + r2 * np.array([np.cos(th + s * b), np.sin(th + s * b)])) for s in (1, -1)]


def draw_map(fig, W, H, box_in):
    """Map in box_in = (left, bottom, width, height) inches, with all overlays."""
    with netCDF4.Dataset(MAP_NC) as d:
        k = int(np.argmin(abs(np.asarray(d["time"][:]) - T_MAP)))
        x = np.asarray(d["x"][:], float); y = np.asarray(d["y"][:], float)
        thk = np.asarray(d["thk"][k], np.float32)
        usurf = np.asarray(d["usurf"][k], np.float32)
        spd = np.asarray(d["velsurf_mag"][k], np.float32)
    dx = abs(x[1] - x[0])
    v = np.where(thk > THK_MIN, spd, np.nan)
    norm, cmap = LogNorm(*SPEED_RANGE), plt.get_cmap(SPEED_CMAP)
    E = edges(x, dx) + edges(y, dx)

    L0, B0, mw, mh = box_in
    cw = CROP_HEIGHT_KM * mw / mh                                    # crop width fitted to the box
    cx, cy = CROP_CENTER_KM
    i0, i1 = np.searchsorted(x, (cx - cw / 2) * 1e3), np.searchsorted(x, (cx + cw / 2) * 1e3)
    j0, j1 = np.searchsorted(y, (cy - CROP_HEIGHT_KM / 2) * 1e3), np.searchsorted(y, (cy + CROP_HEIGHT_KM / 2) * 1e3)
    Ez = edges(x[i0:i1], dx) + edges(y[j0:j1], dx)
    print(f"map extent: {(Ez[1] - Ez[0]) / 1e3:.1f} x {(Ez[3] - Ez[2]) / 1e3:.1f} km  "
          f"(x {Ez[0] / 1e3:.1f}-{Ez[1] / 1e3:.1f}, y {Ez[2] / 1e3:.1f}-{Ez[3] / 1e3:.1f} km)")
    hs = soft_hillshade(usurf[j0:j1, i0:i1], dx)

    ax = fig.add_axes([L0 / W, B0 / H, mw / W, mh / H], projection=GRID_CRS)
    ax.set_gid("map")
    gid(ax.imshow(hs, cmap="gray", vmin=0, vmax=1, origin="lower", extent=Ez, transform=GRID_CRS,
                  interpolation="nearest", zorder=0), "hillshade--map")
    im = ax.imshow(v[j0:j1, i0:i1], cmap=cmap, norm=norm, origin="lower", extent=Ez, transform=GRID_CRS,
                   interpolation="nearest", zorder=1)
    im.set_gid("speed--map")
    ax.set_extent(Ez, crs=GRID_CRS)
    ax.spines["geo"].set_visible(False)
    graticule_lines(ax, Ez, GRID_STEP, "graticule--lines", 2, **GRATICULE)
    gl = ax.gridlines(crs=LL, draw_labels=True, x_inline=False, y_inline=False, zorder=2)
    gl.xlines = gl.ylines = False
    gl.right_labels = False                                          # plots sit on the right
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, GRID_STEP))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, GRID_STEP))
    gl.xformatter = LongitudeFormatter(number_format=".1f", degree_symbol="°")
    gl.yformatter = LatitudeFormatter(number_format=".1f", degree_symbol="°")
    gl.xlabel_style = gl.ylabel_style = dict(size=FS, color=INK2)
    gl.xpadding = gl.ypadding = 2.5

    s = mw / MAP_U                                                   # layout unit -> inch
    top = B0 + mh
    fx = lambda u: (L0 + u * s) / W                                  # layout x (from map left) -> fig frac
    fy = lambda u: (top - u * s) / H                                 # layout y (from map top)  -> fig frac
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]

    # locator inset (whole model domain)
    iw = INSET_BOX[2] * s
    ih = iw * (E[3] - E[2]) / (E[1] - E[0])
    ai = fig.add_axes([fx(INSET_BOX[0]), fy(INSET_BOX[1]) - ih / H, iw / W, ih / H], projection=GRID_CRS)
    ai.set_gid("inset")
    ai.set_extent(E, crs=GRID_CRS)
    ai.set_facecolor(SEA)
    n_ = INSET_SUBSAMPLE
    hs_i = soft_hillshade(usurf[::n_, ::n_], dx * n_)
    hs_i[usurf[::n_, ::n_] <= 0] = np.nan
    gid(ai.imshow(hs_i, cmap="gray", vmin=0, vmax=1, origin="lower", extent=E, transform=GRID_CRS,
                  interpolation="antialiased", zorder=1), "hillshade--inset")
    gid(ai.imshow(v[::n_, ::n_], cmap=cmap, norm=norm, origin="lower", extent=E, transform=GRID_CRS,
                  interpolation="antialiased", zorder=2), "speed--inset")
    unclipped_feature(ai, cfeature.LAKES.with_scale("50m"), E, "features--lakes", 3, polygons=True,
                      facecolor=SEA, edgecolor="none")
    unclipped_feature(ai, cfeature.COASTLINE.with_scale("50m"), E, "features--coastline", 3,
                      colors=BORDER, linewidths=0.3)
    unclipped_feature(ai, cfeature.BORDERS.with_scale("50m"), E, "features--borders", 3,
                      colors=BORDER, linewidths=0.4, linestyles=[(0, (4, 1.5))])
    bx = [Ez[0], Ez[1], Ez[1], Ez[0], Ez[0]]; by = [Ez[2], Ez[2], Ez[3], Ez[3], Ez[2]]
    gid(ai.plot(bx, by, color="white", lw=2.0, transform=GRID_CRS, zorder=5), "box--map-halo")
    gid(ai.plot(bx, by, color=BOX, lw=1.0, transform=GRID_CRS, zorder=6), "box--map")
    for lab, lon, lat in COUNTRIES:
        ai.text(lon, lat, lab, gid=f"labels--{lab}", transform=LL, fontsize=FS * 0.9, color=INK2,
                ha="center", va="center", fontweight="bold", path_effects=halo, zorder=7)
    ai.spines["geo"].set_edgecolor(INK2); ai.spines["geo"].set_linewidth(0.6)

    # colour bar, no background: white stroke on bar outline, tick labels (set in main) and title
    key_halo = [pe.withStroke(linewidth=MAP_KEY_HALO, foreground="white")]
    bx0, by0, bw, bh = CBAR_BOX
    cax = fig.add_axes([fx(bx0), fy(by0 + bh), bw * s / W, bh * s / H], zorder=3)
    cax.set_gid("colorbar--bar")
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 10, 100, 1000])                                 # all decades (auto thins them at large font)
    cb.set_label("surface speed (m yr$^{-1}$)", labelpad=1.5, fontsize=FS, path_effects=key_halo)
    cax.tick_params(labelsize=FS * 0.95, length=2, pad=1.2)
    cb.outline.set_linewidth(0.5)
    hw = MAP_KEY_HALO / 2 / 72                                       # white stroke OUTSIDE the bar (a stroke on
    fig.add_artist(matplotlib.patches.Rectangle(                     # the outline would eat into the thin bar)
        (fx(bx0) - hw / W, fy(by0 + bh) - hw / H), bw * s / W + 2 * hw / W, bh * s / H + 2 * hw / H,
        transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=2.9, gid="colorbar--halo"))

    # scale bar (upside-down bracket, top aligned with the colour bar) and grid spacing
    L = SCALEBAR_KM / ((Ez[1] - Ez[0]) / 1e3) * MAP_U                # bar length in layout units
    x0_, x1_ = fx(SCALEBAR_X), fx(SCALEBAR_X + L)
    y0_, y1_ = fy(KEY_TOP), fy(KEY_TOP + CBAR_BOX[3])
    fig.add_artist(Line2D([x0_, x0_, x1_, x1_], [y1_, y0_, y0_, y1_], color=INK2, lw=0.6,
                          transform=fig.transFigure, gid="scalebar--bar", solid_joinstyle="miter",
                          solid_capstyle="butt", zorder=3,
                          path_effects=[pe.Stroke(linewidth=0.6 + MAP_KEY_HALO, foreground="white"), pe.Normal()]))
    t_km = fig.text((x0_ + x1_) / 2, y1_, f"{SCALEBAR_KM:g} km", ha="center", va="center", fontsize=FS,
                    color=INK, gid="scalebar--label", zorder=3, path_effects=key_halo)
    t_dx = fig.text(fx(MAP_U - DX_LABEL_RIGHT), y1_, f"Δx = {dx:.0f} m", ha="right", va="center", fontsize=FS,
                    color=INK, gid="scalebar--dx", zorder=3, path_effects=key_halo)

    # close-up, bottom right
    zc = np.array(ZOOM_CENTER_KM) * 1e3; zr = ZOOM_RADIUS_KM * 1e3
    R = ZOOM_BOX[2] * s
    cz = np.array([L0 + ZOOM_BOX[0] * s, top - ZOOM_BOX[1] * s])     # inches
    az = fig.add_axes([(cz[0] - R) / W, (cz[1] - R) / H, 2 * R / W, 2 * R / H], zorder=4)
    az.set_gid("zoom--magnifier")
    nz = int(np.ceil(zr / dx)) + 2
    ic, jc = np.searchsorted(x, zc[0]), np.searchsorted(y, zc[1])
    sl = np.s_[jc - nz:jc + nz, ic - nz:ic + nz]
    Ezz = edges(x[ic - nz:ic + nz], dx) + edges(y[jc - nz:jc + nz], dx)
    clip = matplotlib.patches.Circle(zc, zr, transform=az.transData)
    for nm, arr, kw in (("hillshade", soft_hillshade(usurf[sl], dx), dict(cmap="gray", vmin=0, vmax=1)),
                        ("speed", v[sl], dict(cmap=cmap, norm=norm))):
        zi = az.imshow(arr, origin="lower", extent=Ezz, interpolation="nearest", **kw)
        zi.set_clip_path(clip); zi.set_gid(f"{nm}--zoom")
    az.set_xlim(zc[0] - zr, zc[0] + zr); az.set_ylim(zc[1] - zr, zc[1] + zr)
    az.set_aspect("equal"); az.axis("off")
    for c, lw in (("white", 2.2), (ZOOM_EDGE, 0.9)):
        az.add_patch(matplotlib.patches.Circle(zc, zr, fill=False, ec=c, lw=lw, clip_on=False,
                                               gid=f"box--{'halo' if c == 'white' else 'edge'}"))
        ax.add_patch(matplotlib.patches.Circle(zc, zr, fill=False, ec=c, lw=lw * 0.8, transform=GRID_CRS,
                                               gid=f"zoom--source-{'halo' if c == 'white' else 'edge'}",
                                               zorder=6))
    src = ax.transData.transform(zc) / fig.dpi
    src_r = zr / (Ez[1] - Ez[0]) * mw
    for p1, p2 in ext_tangents(src, src_r, cz, R):
        for c, lw in (("white", 1.6), (ZOOM_EDGE, 0.55)):
            fig.add_artist(Line2D([p1[0] / W, p2[0] / W], [p1[1] / H, p2[1] / H], color=c, lw=lw,
                                  gid=f"zoom--connector-{'halo' if c == 'white' else 'line'}",
                                  transform=fig.transFigure, zorder=3.5))
    az.text(*ZOOM_LABEL_XY, f"×{R / src_r:.0f}", gid="labels--magnification", transform=az.transAxes,
            ha="center", va="center", fontsize=FS + 0.5, fontweight="semibold", color=INK, path_effects=halo)

    for a in ax.get_children() + ai.get_children():
        if isinstance(a, (matplotlib.image.AxesImage, Line2D, matplotlib.text.Text, matplotlib.patches.Circle)):
            a.set_clip_on(False)
    return dict(ax=ax, cax=cax, row=(t_km, t_dx), key_halo=key_halo, bar_mid=(y0_ + y1_) / 2)


# =============================================================================================== figure
def main():
    W = FIG_WIDTH_CM / 2.54
    H = TITLE_BAND + PLOT_TOP + 2 * PANEL_HEIGHT + ERROR_HEIGHT + 2 * PLOT_GAP + PLOT_BOTTOM
    m = MAP_MARGIN
    mh = H - TITLE_BAND - m["top"] - m["bottom"]
    mw = W - m["left"] - GAP_MAP_PLOTS - PLOT_WIDTH - RIGHT_MARGIN
    fig = plt.figure(figsize=(W, H))
    print(f"figure: {W * 2.54:.1f} x {H * 2.54:.1f} cm")

    keys = draw_map(fig, W, H, (m["left"], m["bottom"], mw, mh))

    data = load_both()
    px = m["left"] + mw + GAP_MAP_PLOTS
    ae = fig.add_axes([px / W, PLOT_BOTTOM / H, PLOT_WIDTH / W, ERROR_HEIGHT / H])
    y_t = PLOT_BOTTOM + ERROR_HEIGHT + PLOT_GAP
    at = fig.add_axes([px / W, y_t / H, PLOT_WIDTH / W, PANEL_HEIGHT / H], sharex=ae)
    am = fig.add_axes([px / W, (y_t + PANEL_HEIGHT + PLOT_GAP) / H, PLOT_WIDTH / W, PANEL_HEIGHT / H], sharex=ae)
    mids = dict(memory=plot_panel(am, "memory", data, show_x=False),
                time=plot_panel(at, "time", data, show_x=False))
    plot_panel(ae, "error", data, show_x=True)
    pm, pt, pe = am.get_position(), at.get_position(), ae.get_position()   # three rows: same frame width
    assert all(np.isclose(pm.x0, q.x0) and np.isclose(pm.width, q.width) for q in (pt, pe))
    assert np.isclose(pm.height, pt.height)                             # memory and run time: same height

    # one "outside training distribution" label (run-time panel), dashed leaders to both brackets
    lab = at.text(*OUT_LABEL_XY, f"outside training\ndistribution\n(Δx > {DX_TRAIN:g} m)", color=OUT_COLOR,
                  ha="center", va="center", fontsize=FS, linespacing=1.15, zorder=3, gid="outside--label",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.0), clip_on=False)
    bb = lab.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    st = OUT_LEADER_STYLE
    for kind, (bx, by) in mids.items():
        start = ((bb.x0 + 0.70 * bb.width, bb.y1) if by > bb.y1             # leave from the side facing it
                 else (bb.x1, bb.y0 + 0.25 * bb.height))
        for c, lw, extra in (("white", st["lw"] + st["halo"], {}),
                             (OUT_COLOR, st["lw"], dict(linestyle=(0, st["dashes"])))):
            fig.add_artist(matplotlib.patches.FancyArrowPatch(
                start, (bx, by), transform=fig.transFigure, arrowstyle="-", shrinkA=1.5, shrinkB=2.2,
                connectionstyle=f"arc3,rad={OUT_LEADER[kind]}", color=c, lw=lw, zorder=5,
                capstyle="round", gid=f"leaders--{kind}-{'halo' if c == 'white' else 'dash'}", **extra))

    # panel titles a) and b), bold, in the band above the panels
    ty = 1 - 0.03 / H
    ta = fig.text(0, ty, PANEL_TITLES[0], ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", gid="titles--a")
    tb = fig.text(0, ty, PANEL_TITLES[1], ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", gid="titles--b")

    # final alignment from measured positions (needs one draw)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ext = lambda t: t.get_window_extent(r).transformed(fig.transFigure.inverted())
    #  colour-bar tick labels get the white stroke; "50 km" and "Δx" go on their row
    ticks = [t for t in keys["cax"].get_xticklabels() if t.get_text()]
    for t in ticks:
        t.set_path_effects(keys["key_halo"])
    row = np.mean([(ext(t).y0 + ext(t).y1) / 2 for t in ticks])
    keys["row"][0].set_y(row)                                         # "50 km": under its bar
    t_dx = keys["row"][1]                                             # "Δx": on the row of the colour-bar title
    lb = ext(keys["cax"].xaxis.label)
    t_dx.set_y((lb.y0 + lb.y1) / 2)
    #  a) flush with the left edge of the latitude labels, b) with the left edge of the plots' y labels
    gl = keys["ax"].artists[0]
    lat = [l.artist for l in gl._labels if l.artist.get_visible() and l.artist.get_text().endswith("N")]
    for t, target in ((ta, min(ext(a).x0 for a in lat)), (tb, min(ext(a.yaxis.label).x0 for a in (am, at, ae)))):
        t.set_x(t.get_position()[0] + target - ext(t).x0)
    save(fig, NAME)


if __name__ == "__main__":
    main()

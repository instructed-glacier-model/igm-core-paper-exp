#!/usr/bin/env python3

"""Draw the Drønbreen cold/temperate figure (Fig. 8 of the paper) from the data beside it.

    python make_figure.py                    # writes cts_figure_minimal.pdf and .png
    python make_figure.py --out figures/     # somewhere else
    python make_figure.py --data other_run/  # another set of the data files

The figure is a full-width (169.7 x 103.5 mm) figure with three panels:

    (a) map: surface contours on land, the glacier as a light-blue mask with an inner
        shadow, the ten radar tracks dashed and the featured transect (T3) in orange
        with its ends marked A (upper) and B (lower), and a key underneath;
    (b) the modelled thermal section along that transect, drawn from A to B so ice flows
        left to right, with the observed CTS laid over it;
    (c) temperate ice per transect: observed and modelled cross-sectional areas on a
        track as long as the transect's total ice cross-section, with the RMSE of the
        modelled CTS elevation as a column beside it.

Everything placed by hand in the layout is a constant near the top of this file, in one of two units:

    * figure points (pt), measured from the TOP-LEFT corner of the page, for panel
      frames, titles, keys and panel letters;
    * data units (km on the map, km/m in the section) for things tied to the data.

So moving something is a matter of editing one number. The script does no glaciology:
every quantity it draws is in section.nc, transects.csv, map.nc and tracks.csv
(written by ../runs/make_figure_data.py from a Step 1 run), plus the Svalbard coastline of the
locator inset in svalbard.csv (Natural Earth 1:10m land, public domain).

Type is Inter (bundled in fonts/, SIL Open Font License).
Requires numpy, pandas, scipy, xarray and matplotlib.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------------------
INK, INK2 = "#0b0b0b", "#52514e"      # text and axes / secondary text
GRID, AXIS = "#e1e0d9", "#c3c2b7"     # summary gridlines / obs-mod connector, hairline
FEATURED = "#ff7327"                  # the featured transect, everywhere on the page
GLACIER_KEY_TEXT = "#44819e"          # "glacier" in the map key, drawn faded
GLACIER_KEY_ALPHA = 0.5

TEMPERATE = "#b5446e"                 # modelled temperate ice: section fill and marker
TEMPERATE_TEXT = TEMPERATE
COLD = "#e3eef6"                      # modelled cold ice in the section
COLD_TEXT = "#3b6f96"
ROCK = "#a9a8a3"                      # bedrock in the section
OBSERVED = "#5e1a37"                  # observed temperate area marker
MODELLED = TEMPERATE
TRACK = "#c4e6e9"                     # total-ice bar in the summary (ice-ramp blue)
TRACK_TEXT = "#447fb9"                # its key label (darker, same colormap)
CTS_LINE = INK                        # observed CTS in the section (white-cased)

C_SHADE = "#3d3c39"                   # hillshade ink (drawn translucent over white)
C_ICE = "#cbebee"                     # glacier base colour (and the key's glacier icon)
# Ice thickness in banded steps, from the light end of cmocean's "ice" colormap
# (0.98 -> 0.64 of it; sampled here so cmocean is not needed to run the script).
ICE_RAMP = ["#e1f7f9", "#cbebee", "#b5e0e3", "#9cd4da", "#88c8d4", "#75bbcf",
            "#68aeca", "#5ba0c5"]
THK_LEVELS = np.arange(0, 181, 20)    # m, one band per 20 m of ice
C_GRATICULE = (*to_rgb(INK2), 0.28)
C_DIAMOND = "#8f8e8a"                 # graticule crossings
# White casing round the featured transect on the map, to lift it off the ice colours.
FEATURED_CASING = [patheffects.Stroke(linewidth=3.8, foreground="white"),
                   patheffects.Normal()]
C_TRACK = INK2                        # radar tracks (dashed)
C_MAP_LABEL = INK2

# ---------------------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------------------
for _f in glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
    font_manager.fontManager.addfont(_f)


def first_font(*names):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), "DejaVu Sans")


FS = 6.5            # base size: ticks, keys, map labels
FS_LABEL = 7.0      # axis labels, section title, A/B
FS_LETTER = 7.5     # panel letters
FS_SMALL = 6.3      # RMSE header and footer
FS_FEATURED = 9.0   # "T3" on the map

STYLE = {
    "font.family": first_font("Inter", "Nimbus Sans", "Helvetica", "Arial",
                              "Liberation Sans"),
    "font.size": FS, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.pad": 1.4, "ytick.major.pad": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
}

# ---------------------------------------------------------------------------------------
# Page and panel geometry, in pt from the top-left corner of the page
# ---------------------------------------------------------------------------------------
PAGE_W, PAGE_H = 481.12, 293.34     # 169.7 x 103.5 mm

# (left, top, right, bottom)
MAP_BOX = (15.8, 9.4, 231.8, 254.2)
SECTION_BOX = (265.3, 28.25, 472.7, 103.85)
SUMMARY_BOX = (268.0, 162.0, 428.0, 254.4)
RMSE_RIGHT = 475.0                  # right edge of the RMSE column

# Panel letters and titles: (left edge of the letter, vertical centre, title). The
# title follows the letter; "{section}" is replaced by the featured transect's name.
PANEL_TITLES = {
    "a": (5.0, 6.0, "Observation map"),
    "b": (246.0, 6.0, "{section} transect"),
    "c": (248.0, 135.25, "Misfit between modelled and observed CTS values"),
}
TITLE_GAP = 2.5                     # pt between letter and title

# ---------------------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------------------
MAP_XLIM = (537.0, 544.5)           # km, UTM 33N
MAP_YLIM = (8668.1, 8676.6)
# Hillshade: light from the north-west, multiplied over white so only shaded slopes
# show. LAND_SHADE is the opacity of the darkest slope.
SUN_AZIMUTH, SUN_ALTITUDE = 315.0, 40.0   # degrees
VERT_EXAG = 1.0
LAND_SHADE = 0.48
EDGE_FADE = 12.0                    # pt over which the land contours fade out at the frame
GLOW_PT = 2.0                       # pt, white glow round the outside of the margin
SHADOW_PT = 1.4                     # pt, inner shadow inside the margin
SHADOW_ALPHA = 0.45
UPSAMPLE = 8                        # the 100 m grid is resampled this much finer for the margin

# Each transect label: label centre (km), the point it points at (km; snapped to the
# nearest point of the track), and the direction the leader leaves the label and
# arrives at the track (degrees, 0 = right, 90 = up). The leader is a cubic curve.
MAP_LABELS = {
    "T1":  ((537.783, 8669.182), (538.604, 8668.992), 0, 0),
    "T2":  ((541.653, 8668.626), (540.618, 8669.044), 180, 135),
    "T4":  ((538.873, 8670.883), (538.535, 8670.138), -90, -90),
    "T5":  ((540.229, 8672.991), (541.312, 8672.013), -90, -30),
    "T6":  ((540.510, 8675.432), (541.833, 8675.347), 0, 0),
    "T7":  ((540.724, 8673.890), (541.851, 8673.524), 0, -40),
    "T8":  ((542.594, 8675.154), (541.833, 8675.277), 180, 165),
    "T9":  ((543.80, 8669.98), (542.997, 8670.659), 180, 90),
    "T10": ((542.688, 8669.633), (542.944, 8670.659), 90, 80),
}
FEATURED_LABEL_AT = (540.568, 8671.933)  # the big "T3", no leader

# Graticule (WGS 84), drawn in the map's UTM 33N with a diamond at every crossing.
GRATICULE_LON = (16.7, 16.8, 16.9)      # degrees E
GRATICULE_LAT = (78.10, 78.12, 78.14)   # degrees N
GRATICULE_LABEL_SIZE = 5.0

# Inset locating the map on Svalbard (coastline: Natural Earth 1:10m land, public
# domain, in svalbard.csv), in page pt: left, top, right, bottom.
INSET_BOX = (18.5, 13.0, 61.5, 56.0)
C_INSET_LAND = "#d9d8d3"

# Thickness colour bar, in page pt: left, top, right, bottom of the bar itself.
THK_BAR_BOX = (182.0, 24.0, 224.0, 27.2)

SCALE_BAR = ((544.1 - 1.0, 544.1), 8668.5)        # km: (x0, x1), y
NORTH_ARROW = (537.655, 8673.56, 8674.15)         # km: x, tail y, head y

# Key under the map, in page pt (x of the text's left edge, vertical centre).
KEY_Y = 273.8
KEY_OBS_TEXT_X = 46.6
KEY_GLACIER_TEXT_X = 122.5
KEY_FEATURED_TEXT_X = 181.0
KEY_OBS_ICON = (31.0, 279.6, 13.0, 8.6)          # left, bottom, width, height
KEY_GLACIER_ICON = (104.0, 282.9)                 # origin (lower-left) of the blob
KEY_FEATURED_ICON = [(161.0, 280.0), (165.1, 272.5), (179.0, 270.0)]

# The glacier icon of the key (pt, y up).
GLACIER_ICON_PATH = [
    ("M", (0.0165, 8.8609)),
    ("C", (0.1314, 11.6747), (3.9179, 10.9832), (6.0165, 12.8609)),
    ("C", (6.8397, 13.5975), (6.9231, 15.0171), (8.0165, 14.8609)),
    ("C", (9.2391, 14.6863), (8.3535, 12.9028), (9.0165, 11.8609)),
    ("C", (10.4230, 9.6508), (16.0563, 11.2655), (15.0165, 8.8609)),
    ("C", (14.3917, 7.4160), (13.0718, 6.6170), (11.5165, 6.8609)),
    ("C", (10.2814, 7.0547), (10.2493, 8.6523), (9.0165, 8.8609)),
    ("C", (6.5137, 9.2845), (10.5396, 4.3917), (9.0165, 2.3609)),
    ("C", (8.1717, 1.2345), (7.4137, 0.5356), (6.0165, 0.3609)),
    ("C", (4.2835, 0.1443), (2.6534, 0.7347), (2.0165, 2.3609)),
    ("C", (1.2835, 4.2329), (6.2739, 5.8846), (4.5165, 6.8609)),
    ("C", (3.2739, 7.5513), (2.2265, 5.1149), (1.0165, 5.8609)),
    ("C", (-0.0347, 6.5091), (-0.0338, 7.6270), (0.0165, 8.8609)),
]

# ---------------------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------------------
SECTION_YLIM = (250, 790)
SECTION_YTICKS = np.arange(300, 800, 100)
FLOW_ARROW = ((1.36, 677.5), (2.155, 642.5))    # (km, m): tail, head
FLOW_TEXT_AT = (1.76, 699.0)                    # (km, m): centre of "ice flow"
FLOW_COLOR = (*to_rgb(INK2), 0.6)
BEDROCK_TEXT_AT = (0.018, 0.090)                # axes fraction: left edge, centre
CTS_DASH = (0, (2.2, 1.2))                      # observed CTS dash pattern
CTS_CASING = [patheffects.Stroke(linewidth=2.9, foreground="white"), patheffects.Normal()]
SECTION_XTICK_PAD = 2.5                         # pt
ENDPOINT_MARKER = 7.5                           # pt, orange discs at the profile ends

# Key: one line under the (b) title, in page pt (right edge, vertical centre).
SECTION_KEY_AT = (472.7, 17.5)    # right-aligned to the end of the axis
SECTION_KEY_GAP = 6.0              # pt between key items

# ---------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------
BAR_HEIGHT = 0.42
MARKER_AREA = 24
SUMMARY_PAD_ABOVE = 0.70           # rows between the top axis and the first row
SUMMARY_PAD_BELOW = 0.60           # rows below the last row
MEAN_LINE_AT = 0.55                # rows below the last row: hairline over the mean
MEAN_TEXT_AT = 1.20                # rows below the last row: "mean 18.8 m"
# Key: one line under the chart, in page pt (left edge, vertical centre), level with
# the map's key.
SUMMARY_KEY_AT = (268.0, 273.8)
SUMMARY_KEY_GAP = 9.0              # pt between key items
KEY_SWATCH_W = 9.2                 # pt, width of key swatches and lines
KEY_SWATCH_EDGE = (*to_rgb(INK2), 0.25)
# Dashed box round the featured row: page pt left/right, half-height in rows.
FEATURED_BOX_X = (251.5, 477.5)
FEATURED_BOX_HALF = 0.47


# =======================================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=HERE, help="Folder holding the four data files")
    p.add_argument("--out", default=HERE, help="Folder for the output .png and .pdf")
    p.add_argument("--name", default="cts_figure_minimal", help="Output file stem")

    return p.parse_args()


def load(folder):
    sec = xr.open_dataset(os.path.join(folder, "section.nc"))
    df = pd.read_csv(os.path.join(folder, "transects.csv"))
    df = df.sort_values("temperate_obs_m2", ascending=False).reset_index(drop=True)
    geo = xr.open_dataset(os.path.join(folder, "map.nc"))
    tracks = {name: t for name, t in
              pd.read_csv(os.path.join(folder, "tracks.csv")).groupby("transect", sort=False)}
    coast = os.path.join(folder, "svalbard.csv")
    coast = pd.read_csv(coast) if os.path.exists(coast) else None

    return sec, df, geo, tracks, coast


def axes_pt(fig, box, **kw):
    """Axes from a (left, top, right, bottom) box in page pt measured from the top."""

    l, t, r, b = box
    return fig.add_axes([l / PAGE_W, 1 - b / PAGE_H, (r - l) / PAGE_W, (b - t) / PAGE_H],
                        **kw)


def page_axes(fig):
    """A transparent axes over the whole page in pt, y downwards, for hand-placed items."""

    ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax.set(xlim=(0, PAGE_W), ylim=(PAGE_H, 0))
    ax.axis("off")
    ax.patch.set_alpha(0)

    return ax


def endpoints(track, geo):
    """(upper, lower) end of a track, by the surface elevation of the model grid."""

    x, y = geo["x"].values, geo["y"].values
    surface = (geo["topg"] + geo["thk"]).values
    e, n = track["easting"].values, track["northing"].values
    ends = [(e[0], n[0]), (e[-1], n[-1])]
    z = [surface[np.abs(y - nn).argmin(), np.abs(x - ee).argmin()] for ee, nn in ends]
    upper, lower = (ends[0], ends[1]) if z[0] >= z[1] else (ends[1], ends[0])

    return np.array(upper) / 1e3, np.array(lower) / 1e3


# ---------------------------------------------------------------------------------------
# (a) map
# ---------------------------------------------------------------------------------------


def edge_fade(ax, width_pt):
    """White fading in towards the axes frame, so the land contours dissolve at the edge."""

    l, t, r, b = MAP_BOX
    w, h = r - l, b - t
    nx, ny = int(w * 4), int(h * 4)
    px, py = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny))
    d = np.minimum.reduce([px, w - px, py, h - py])
    alpha = np.clip(1 - d / width_pt, 0, 1) ** 1.5
    rgba = np.ones((ny, nx, 4))
    rgba[..., 3] = alpha
    ax.imshow(rgba, extent=(*MAP_XLIM, *MAP_YLIM), origin="lower", zorder=1.2,
              interpolation="bilinear", aspect="auto")


def shade_image(ax, z, dx, strength, extent, zorder):
    """Minimal hillshade: translucent ink where slopes turn away from the sun.

    z is resampled UPSAMPLE times finer (cubic) so the 100 m cells do not show; flat
    ground and sunlit slopes stay white.
    """

    zf = zoom(z, UPSAMPLE, order=3)
    d = dx / UPSAMPLE
    gy, gx = np.gradient(zf * VERT_EXAG, d, d)       # rows run south to north
    az, alt = np.radians(SUN_AZIMUTH), np.radians(SUN_ALTITUDE)
    light = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    illum = (-gx * light[0] - gy * light[1] + light[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1)
    dark = np.clip((np.sin(alt) - illum) / np.sin(alt), 0, 1)

    rgba = np.zeros(zf.shape + (4,))
    rgba[..., :3] = to_rgb(C_SHADE)
    rgba[..., 3] = strength * dark

    return ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear",
                     zorder=zorder)


def utm33(lon, lat):
    """WGS 84 longitude/latitude (degrees) to UTM zone 33N easting/northing (km).

    Transverse Mercator by Krüger's series (as in pyproj, to well under a metre), so
    the script needs no projection library.
    """

    a, f, k0, lon0 = 6378137.0, 1 / 298.257223563, 0.9996, np.radians(15.0)
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5,
             61 * n ** 3 / 240)
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)) - lon0
    c = 2 * np.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    E, N = eta, xi
    for j, aj in enumerate(alpha, start=1):
        E = E + aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        N = N + aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)

    return (500000.0 + k0 * A * E) / 1e3, k0 * A * N / 1e3


def draw_graticule(ax):
    """Lines of latitude and longitude, diamonds at the crossings, labels at the edges.

    Longitude labels sit just inside the top edge, latitude labels just inside the
    left edge, each on the white of the faded frame.
    """

    lat_span = np.linspace(77.9, 78.4, 200)
    lon_span = np.linspace(16.3, 17.3, 200)
    kw = dict(color=C_GRATICULE, lw=0.4, zorder=3, solid_capstyle="butt")
    for lon in GRATICULE_LON:
        ax.plot(*utm33(np.full_like(lat_span, lon), lat_span), **kw)
    for lat in GRATICULE_LAT:
        ax.plot(*utm33(lon_span, np.full_like(lon_span, lat)), **kw)
    lon_g, lat_g = np.meshgrid(GRATICULE_LON, GRATICULE_LAT)
    ax.plot(*utm33(lon_g.ravel(), lat_g.ravel()), "D", ms=2.2, mfc=C_DIAMOND, mec="white",
            mew=0.3, ls="none", zorder=3.5)

    label = dict(fontsize=GRATICULE_LABEL_SIZE, color=INK2, zorder=7)
    pad = 0.07  # km in from the frame
    for lon in GRATICULE_LON:
        # where this meridian meets the top edge
        e, n = utm33(np.full_like(lat_span, lon), lat_span)
        x_top = np.interp(MAP_YLIM[1], n, e)
        ax.text(x_top + 0.04, MAP_YLIM[1] - pad, f"{lon:.1f}°E", ha="left", va="top",
                **label)
    for lat in GRATICULE_LAT:
        e, n = utm33(lon_span, np.full_like(lon_span, lat))
        y_left = np.interp(MAP_XLIM[0], e, n)
        ax.text(MAP_XLIM[0] + pad, y_left + 0.04, f"{lat:.2f}°N", ha="left",
                va="bottom", **label)


def draw_inset(fig, coast):
    """Svalbard in UTM 33N on a white card, with the map's extent marked."""

    ax = axes_pt(fig, INSET_BOX, zorder=3)
    ax.set_facecolor("white")
    for _, ring in coast.groupby("ring"):
        e, n = utm33(ring["lon"].values, ring["lat"].values)
        ax.fill(e, n, color=C_INSET_LAND, lw=0)
    e, n = utm33(coast["lon"].values, coast["lat"].values)
    # Fit the archipelago in the box at equal scale, centred.
    pad = 12.0  # km
    l, t, r, b = INSET_BOX
    cx_, cy_ = (e.min() + e.max()) / 2, (n.min() + n.max()) / 2
    half = max(np.ptp(e), np.ptp(n) * (r - l) / (b - t)) / 2 + pad
    ax.set(xlim=(cx_ - half, cx_ + half),
           ylim=(cy_ - half * (b - t) / (r - l), cy_ + half * (b - t) / (r - l)),
           xticks=[], yticks=[], aspect="equal")

    # The main map's extent, too small to see at this scale, so marked by a square.
    cx, cy = np.mean(MAP_XLIM), np.mean(MAP_YLIM)
    ax.plot(cx, cy, "s", ms=4.2, mfc="none", mec=INK, mew=0.9, zorder=3)
    ax.text(0.06, 0.94, "Svalbard", transform=ax.transAxes, ha="left", va="top",
            fontsize=GRATICULE_LABEL_SIZE, color=INK2)
    for spine in ax.spines.values():
        spine.set(linewidth=0.4, edgecolor=INK2, visible=True)


def thickness_cmap():
    """Banded ice-thickness colours: one ICE_RAMP shade per THK_LEVELS band."""

    ramp = LinearSegmentedColormap.from_list("ice_light", ICE_RAMP)
    colours = ramp(np.linspace(0, 1, len(THK_LEVELS) - 1))
    cmap = ListedColormap(colours)

    return cmap, BoundaryNorm(THK_LEVELS, cmap.N)


def thickness_bar(fig):
    """Small horizontal colour bar for ice thickness, top right of the map."""

    # A white card behind the bar and its labels, so the graticule passes under it.
    l, t, r, b = THK_BAR_BOX
    card = axes_pt(fig, (l - 3.0, t - 7.6, r + 4.0, b + 8.0), zorder=1.5)
    card.set_axis_off()
    card.add_patch(Rectangle((0, 0), 1, 1, transform=card.transAxes, color="white",
                             alpha=0.9, lw=0))
    cax = axes_pt(fig, THK_BAR_BOX, zorder=2)
    cmap, norm = thickness_cmap()
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", ticks=[0, 50, 100, 150])
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor(INK2)
    cax.minorticks_off()
    cax.tick_params(length=2, width=0.4, pad=1.2, labelsize=GRATICULE_LABEL_SIZE)
    cax.set_title("Ice thickness (m)", fontsize=GRATICULE_LABEL_SIZE, color=INK2,
                  pad=2.5, loc="left")


def bezier_leader(ax, p0, p3, a0, a3, scale, **kw):
    """Cubic curve from p0 leaving at angle a0 to p3 arriving at angle a3 (degrees)."""

    p0, p3 = np.asarray(p0), np.asarray(p3)
    L = np.hypot(*(p3 - p0)) * scale
    d0 = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
    d3 = np.array([np.cos(np.radians(a3)), np.sin(np.radians(a3))])
    p1, p2 = p0 + d0 * L, p3 - d3 * L
    path = Path([p0, p1, p2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, **kw))


def draw_map(ax, geo, tracks, section):
    x, y = geo["x"].values / 1e3, geo["y"].values / 1e3
    thk = geo["thk"].values
    surface = geo["topg"].values + thk
    step = x[1] - x[0]
    ice = thk > 1

    ax.set_facecolor("white")
    extent = (x[0] - step / 2, x[-1] + step / 2, y[0] - step / 2, y[-1] + step / 2)

    # Land: hillshade of the DEM. Under the ice and in a one-cell ring round it (where
    # the DEM and the kriged bed + thickness do not quite meet) the land is carried in
    # from the nearest land cell beyond the ring, so the margin casts no false relief.
    ring = ~ice & (distance_transform_edt(~ice) <= 1.5)
    hide = ice | ring
    _, (iy, ix) = distance_transform_edt(hide, return_indices=True)
    land = gaussian_filter(surface[iy, ix], 0.45)
    shade_image(ax, land, step * 1e3, LAND_SHADE, extent, zorder=1)
    edge_fade(ax, EDGE_FADE)

    # The margin, on a grid UPSAMPLE times finer than the model's, from a lightly
    # blurred ice mask so that it is not drawn as 100 m stair steps.
    smooth = gaussian_filter(ice.astype(float), sigma=0.9)
    fine = zoom(smooth, UPSAMPLE, order=3)
    xf = np.linspace(x[0], x[-1], fine.shape[1])
    yf = np.linspace(y[0], y[-1], fine.shape[0])
    ice_f = fine > 0.5
    extent_f = (xf[0], xf[-1], yf[0], yf[-1])
    pt_per_km = (MAP_BOX[2] - MAP_BOX[0]) / (MAP_XLIM[1] - MAP_XLIM[0])
    px_per_pt = 1.0 / (pt_per_km * (xf[1] - xf[0]))       # fine cells per pt

    # White glow outside the margin (it lifts the ice off the contours).
    glow = gaussian_filter(ice_f.astype(float), GLOW_PT * px_per_pt)
    rgba = np.ones(ice_f.shape + (4,))
    rgba[..., 3] = np.clip(2.2 * glow, 0, 1) * ~ice_f
    ax.imshow(rgba, extent=extent_f, origin="lower", zorder=1.5, interpolation="bilinear")

    band = ax.contourf(xf, yf, fine, levels=[0.5, 2], colors=[C_ICE], zorder=2)
    clip = band.get_paths()[0]

    # Ice thickness in 20 m bands with a fine white line between bands, so the ice
    # reads as a body of varying depth. The field is carried a few cells past the
    # margin from the nearest ice cell, smoothed, and resampled finer, then clipped.
    _, (iy, ix) = distance_transform_edt(~ice, return_indices=True)
    thk_f = zoom(gaussian_filter(thk[iy, ix], 0.7), UPSAMPLE // 2, order=3)
    xt = np.linspace(x[0], x[-1], thk_f.shape[1])
    yt = np.linspace(y[0], y[-1], thk_f.shape[0])
    cmap, norm = thickness_cmap()
    bands = ax.contourf(xt, yt, np.clip(thk_f, 0, THK_LEVELS[-1] - 1e-3),
                        levels=THK_LEVELS, cmap=cmap, norm=norm, zorder=2.1)
    bands.set_clip_path(clip, transform=ax.transData)
    lines = ax.contour(xt, yt, thk_f, levels=THK_LEVELS[1:-1], colors="white",
                       linewidths=0.35, alpha=0.75, zorder=2.2)
    lines.set_clip_path(clip, transform=ax.transData)

    # Inner shadow: dark at the margin, fading into the ice.
    inner = gaussian_filter(ice_f.astype(float), SHADOW_PT * px_per_pt)
    shade = np.zeros(ice_f.shape + (4,))
    shade[..., :3] = to_rgb(INK2)
    shade[..., 3] = SHADOW_ALPHA * np.clip(2 * (1 - inner), 0, 1) ** 1.4 * ice_f
    im = ax.imshow(shade, extent=extent_f, origin="lower", zorder=2.4,
                   interpolation="bilinear")
    im.set_clip_path(clip, transform=ax.transData)

    # Margin: a fine white dashed line on the edge of the shadow.
    ax.contour(xf, yf, fine, levels=[0.5], colors="white", linewidths=0.3,
               linestyles=[(0, (4, 3))], zorder=2.6)

    draw_graticule(ax)

    # Tracks.
    for name, t in tracks.items():
        if name == section:
            continue
        ax.plot(t["easting"].values / 1e3, t["northing"].values / 1e3, color=C_TRACK,
                lw=0.6, ls=(0, (2.6, 1.6)), zorder=4, dash_capstyle="butt")
    t = tracks[section]
    ax.plot(t["easting"].values / 1e3, t["northing"].values / 1e3, color=FEATURED,
            lw=1.7, zorder=5, path_effects=FEATURED_CASING)

    # Its ends, A upstream and B downstream, as in the section.
    upper, lower = endpoints(t, geo)
    for (ex, ey), tag in ((upper, "A"), (lower, "B")):
        ax.plot(ex, ey, "o", ms=ENDPOINT_MARKER, color=FEATURED, mec="white", mew=0.4,
                zorder=6)
        ax.text(ex, ey, tag, ha="center", va="center_baseline", fontsize=FS_LABEL - 1.5,
                fontweight="bold", color="white", zorder=6.5)

    # Labels and their leaders.
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    km_per_pt = 1.0 / pt_per_km
    for name, (at, target, a0, a3) in MAP_LABELS.items():
        if name not in tracks:
            continue
        txt = ax.text(*at, name, ha="center", va="center", fontsize=FS,
                      fontweight="semibold", color=C_MAP_LABEL, zorder=7)
        if name == section:
            continue
        # Snap the target to the nearest point of the track.
        tr = tracks[name]
        e, n = tr["easting"].values / 1e3, tr["northing"].values / 1e3
        i = np.argmin((e - target[0]) ** 2 + (n - target[1]) ** 2)
        # Start the leader just outside the label, on the side it leaves from.
        bb = txt.get_window_extent(renderer)
        half_w = bb.width / fig.dpi * 72 / 2
        half_h = 0.36 * FS
        d = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
        off = (half_w + 1.4) * abs(d[0]) + (half_h + 1.4) * abs(d[1])
        p0 = np.array(at) + d * off * km_per_pt
        bezier_leader(ax, p0, (e[i], n[i]), a0, a3, 0.45, lw=0.5, color=INK, zorder=6.8,
                      capstyle="round")

    ax.text(*FEATURED_LABEL_AT, section, ha="center", va="center", fontsize=FS_FEATURED,
            fontweight="bold", color=FEATURED, zorder=7,
            path_effects=[patheffects.Stroke(linewidth=2.2, foreground="white"),
                          patheffects.Normal()])

    # Scale bar.
    (sx0, sx1), sy = SCALE_BAR
    ax.plot([sx0, sx1], [sy, sy], "-", color=INK, lw=1.1, marker="|", markersize=4.5,
            markeredgewidth=1.1, solid_capstyle="butt", zorder=7)
    ax.text((sx0 + sx1) / 2, sy + 0.145, "1 km", ha="center", va="center", fontsize=FS,
            color=INK, zorder=7)

    # North arrow (UTM 33N, grid north up).
    nx_, ny0, ny1 = NORTH_ARROW
    ax.annotate("", xy=(nx_, ny1), xytext=(nx_, ny0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK, mutation_scale=7,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(nx_, ny1 + 0.085, "N", ha="center", va="bottom", fontsize=FS,
            fontweight="bold", color=INK, zorder=7)

    ax.set(xlim=MAP_XLIM, ylim=MAP_YLIM, xticks=[], yticks=[], aspect="equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_map_key(page, section):
    """The key under the map: observations, glacier, featured transect."""

    # Observations: a dashed wiggle like a radar track.
    l, b, w, h = KEY_OBS_ICON
    s = np.linspace(0, 1, 200)
    xs = l + w * s
    ys = b - h * (0.5 + 0.5 * np.cos(2 * np.pi * 1.5 * s)) * 0.85 - 0.6
    ys = np.r_[b, ys, b]
    xs = np.r_[l, xs, l + w]
    page.plot(xs, ys, color=C_TRACK, lw=0.6, ls=(0, (2.2, 1.6)), dash_capstyle="butt")
    page.text(KEY_OBS_TEXT_X, KEY_Y, "observations", ha="left", va="center",
              fontsize=FS, color=INK2)

    # Glacier: the key's glacier icon.
    ox, oy = KEY_GLACIER_ICON
    verts, codes = [], []
    for op, *pts in GLACIER_ICON_PATH:
        for p in pts:
            verts.append((ox + p[0], oy - p[1]))
        codes += [Path.MOVETO] if op == "M" else [Path.CURVE4] * 3
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    page.add_patch(PathPatch(Path(verts, codes), facecolor=C_ICE, edgecolor=INK2,
                             lw=0.5))
    page.text(KEY_GLACIER_TEXT_X, KEY_Y, "glacier", ha="left", va="center", fontsize=FS,
              color=GLACIER_KEY_TEXT, alpha=GLACIER_KEY_ALPHA)

    # Featured transect.
    xs, ys = zip(*KEY_FEATURED_ICON)
    page.plot(xs, ys, color=FEATURED, lw=1.0,
              path_effects=[patheffects.Stroke(linewidth=2.4, foreground="white"),
                            patheffects.Normal()])
    page.text(KEY_FEATURED_TEXT_X, KEY_Y, f"{section} transect", ha="left",
              va="center", fontsize=FS, color=FEATURED)


# ---------------------------------------------------------------------------------------
# (b) section
# ---------------------------------------------------------------------------------------


def draw_section(ax, sec):
    dist = sec["distance_km"].values
    length = dist.max()
    xs = length - dist  # stored from the lower end; drawn from the upper end (A)
    ice = sec["ice_type"].values.astype(float)
    ice[ice < 0] = np.nan
    surface, bed = sec["surface"].values, sec["bed"].values

    ax.fill_between(xs, bed, SECTION_YLIM[0], color=ROCK, lw=0, zorder=0)
    cmap = ListedColormap([COLD, TEMPERATE])
    ax.contourf(np.tile(xs, (ice.shape[0], 1)), sec["elevation"].values, ice,
                levels=[-0.5, 0.5, 1.5], cmap=cmap,
                norm=BoundaryNorm([-0.5, 0.5, 1.5], cmap.N), zorder=1)

    ax.plot(xs, surface, color=FEATURED, lw=2.0, zorder=4)
    # The observed CTS on top of everything, black dashes in a white casing, so it reads
    # over the orange surface, both ice fills and the rock alike.
    ax.plot(xs, sec["cts_observed"].values, color=CTS_LINE, ls=CTS_DASH, lw=1.3,
            zorder=6, dash_capstyle="butt", path_effects=CTS_CASING)
    # Discs at the two ends, cut in half by the axes.
    order = np.argsort(xs)
    for xe, ze in ((xs[order[0]], surface[order[0]]), (xs[order[-1]], surface[order[-1]])):
        ax.plot(xe, ze, "o", ms=ENDPOINT_MARKER, color=FEATURED, mew=0, zorder=5)

    # Ice-flow arrow and its label, rotated to the arrow.
    (x0, y0), (x1, y1) = FLOW_ARROW
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=FLOW_COLOR,
                                mutation_scale=6, shrinkA=0, shrinkB=0), zorder=6)

    ax.set(xlabel="Distance along profile (km)", ylabel="Elevation (m)",
           ylim=SECTION_YLIM, xlim=(0, length), yticks=SECTION_YTICKS)
    ax.tick_params(axis="x", pad=SECTION_XTICK_PAD)
    ax.xaxis.labelpad = 2.6
    ax.yaxis.labelpad = 3.0

    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(*FLOW_TEXT_AT, "ice flow", fontsize=FS, color=FLOW_COLOR, ha="center",
            va="center", rotation=angle, rotation_mode="anchor", zorder=6)

    ax.text(BEDROCK_TEXT_AT[0], BEDROCK_TEXT_AT[1], "Bedrock", transform=ax.transAxes,
            ha="left", va="center", fontsize=FS, fontweight="bold", color="white",
            zorder=6)

    # (A) and (B) beside the first and last distance labels.
    lab = blended_transform_factory(ax.transData, ax.transAxes)
    tick_pad = plt.rcParams["xtick.major.size"] + SECTION_XTICK_PAD
    for xe, tag, ha, dx in ((0, "(A)", "left", 3.0), (length, "(B)", "right", -3.0)):
        ax.text(xe, 0, tag, transform=offset_copy(lab, ax.figure, x=dx, y=-tick_pad,
                                                   units="points"),
                ha=ha, va="top", fontsize=FS_LABEL, color=FEATURED)


# ---------------------------------------------------------------------------------------
# (c) summary
# ---------------------------------------------------------------------------------------


def draw_summary(ax, df, section):
    y = np.arange(len(df))
    totals = df["total_ice_m2"].to_numpy() / 1e3
    obs = df["temperate_obs_m2"].to_numpy() / 1e3
    mod = df["temperate_mod_m2"].to_numpy() / 1e3
    rmse = df["cts_rmse_m"].to_numpy()

    ax.barh(y, totals, BAR_HEIGHT, color=TRACK, zorder=0)
    ax.hlines(y, obs, mod, color=AXIS, lw=1.2, zorder=1)
    ax.scatter(mod, y, s=MARKER_AREA, color=MODELLED, zorder=3, linewidths=0.6,
               edgecolors="white")
    ax.scatter(obs, y, s=MARKER_AREA, color=OBSERVED, zorder=4, linewidths=0.6,
               edgecolors="white")

    # Distance axis along the top, under the key.
    span = totals.max()
    ax.set(yticks=y, yticklabels=df["transect"].tolist(),
           xlim=(-0.035 * span, 1.04 * span),
           ylim=(len(df) - 1 + SUMMARY_PAD_BELOW, -SUMMARY_PAD_ABOVE),
           xlabel="Cross-sectional area (10³ m²)")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.labelpad = 3.0
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for label in ax.get_yticklabels():
        if label.get_text() == section:
            label.set_fontweight("bold")
            label.set_color(FEATURED)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=3.2)

    # RMSE column at a fixed right edge on the page; its header level with the
    # distance labels.
    l, t, r, b = SUMMARY_BOX
    right = (RMSE_RIGHT - l) / (r - l)
    col = blended_transform_factory(ax.transAxes, ax.transData)
    head = offset_copy(ax.transAxes, ax.figure, y=plt.rcParams["xtick.major.size"]
                       + plt.rcParams["xtick.major.pad"], units="points")
    ax.text(right, 1.0, "CTS RMSE", transform=head, ha="right", va="bottom",
            fontsize=FS_SMALL, color=INK2)
    for yi, rv, name in zip(y, rmse, df["transect"]):
        featured = name == section
        ax.text(right, yi, f"{rv:.0f} m", transform=col, ha="right", va="center",
                fontsize=FS, color=FEATURED if featured else INK2,
                fontweight="bold" if featured else "normal")
    ax.plot([right - 37.5 / (r - l), right], [len(df) - 1 + MEAN_LINE_AT] * 2,
            transform=col, color=AXIS, lw=0.6, clip_on=False)
    ax.text(right, len(df) - 1 + MEAN_TEXT_AT, f"mean {np.nanmean(rmse):.1f} m",
            transform=col, ha="right", va="center", fontsize=FS_SMALL, color=INK2)


def key_row(page, x0, y, items, gap, align="left"):
    """A one-line key on the page (pt): each item is (kind, colour, text, text colour).

    kind is "patch" (a filled swatch), "dot" (a summary marker) or "cts" (the
    observed-CTS line). x0 is the left edge, or the right edge with align="right".
    Returns the right edge of the last label.
    """

    renderer = page.figure.canvas.get_renderer()
    if align == "right":
        width = -gap
        for kind, _, text, _ in items:
            probe = page.text(0, 0, text, fontsize=FS)
            width += (6.5 if kind == "dot" else KEY_SWATCH_W + 2.3) + gap \
                + probe.get_window_extent(renderer).width / page.figure.dpi * 72
            probe.remove()
        x0 = x0 - width
    x = x0
    for kind, colour, text, text_colour in items:
        if kind == "dot":
            page.scatter([x + 2.4], [y], s=MARKER_AREA, color=colour, linewidths=0.6,
                         edgecolors="white", zorder=7)
            tx = x + 6.5
        elif kind == "patch":
            page.add_patch(Rectangle((x, y - 2.85), KEY_SWATCH_W, 5.7, facecolor=colour,
                                     edgecolor=KEY_SWATCH_EDGE, lw=0.3, zorder=7))
            tx = x + KEY_SWATCH_W + 2.3
        else:  # cts
            page.plot([x, x + KEY_SWATCH_W], [y, y], color=CTS_LINE, ls=CTS_DASH, lw=1.3,
                      path_effects=CTS_CASING, dash_capstyle="butt", zorder=7)
            tx = x + KEY_SWATCH_W + 2.3
        label = page.text(tx, y, text, ha="left", va="center", fontsize=FS,
                          color=text_colour, zorder=7)
        x = tx + label.get_window_extent(renderer).width / page.figure.dpi * 72 + gap

    return x - gap


def box_featured_row(page, ax, df, section):
    """Dashed orange box round the featured transect's row, label to RMSE."""

    if section not in set(df["transect"]):
        return
    row = int(np.flatnonzero(df["transect"].to_numpy() == section)[0])
    fig = ax.figure
    to_pt = lambda yy: PAGE_H - ax.transData.transform((0, yy))[1] / fig.dpi * 72
    top, bottom = to_pt(row - FEATURED_BOX_HALF), to_pt(row + FEATURED_BOX_HALF)
    x0, x1 = FEATURED_BOX_X
    page.add_patch(FancyBboxPatch((x0, top), x1 - x0, bottom - top,
                                  boxstyle="round,pad=0,rounding_size=1.6",
                                  fill=False, edgecolor=FEATURED, lw=0.7,
                                  ls=(0, (3, 2)), zorder=8))


# =======================================================================================


def make_figure(sec, df, geo, tracks, coast=None):
    section = sec.attrs["transect"]
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))

    a_map = axes_pt(fig, MAP_BOX)
    a_sec = axes_pt(fig, SECTION_BOX)
    a_sum = axes_pt(fig, SUMMARY_BOX)
    page = page_axes(fig)

    draw_map(a_map, geo, tracks, section)
    thickness_bar(fig)
    if coast is not None:
        draw_inset(fig, coast)
    draw_map_key(page, section)
    draw_section(a_sec, sec)
    draw_summary(a_sum, df, section)
    key_row(page, *SECTION_KEY_AT, [
        ("patch", TEMPERATE, "Modelled temperate ice", TEMPERATE_TEXT),
        ("patch", COLD, "Modelled cold ice", COLD_TEXT),
        ("cts", CTS_LINE, "Observed CTS", INK2)], SECTION_KEY_GAP, align="right")
    key_row(page, *SUMMARY_KEY_AT, [
        ("dot", MODELLED, "Modelled (temperate)", MODELLED),
        ("dot", OBSERVED, "Observed (temperate)", OBSERVED),
        ("patch", TRACK, "Total ice", TRACK_TEXT)], SUMMARY_KEY_GAP)
    box_featured_row(page, a_sum, df, section)

    renderer = fig.canvas.get_renderer()
    for tag, (lx, ly, title) in PANEL_TITLES.items():
        letter = page.text(lx, ly, f"({tag})", ha="left", va="center",
                           fontsize=FS_LETTER, fontweight="bold", color=INK)
        width = letter.get_window_extent(renderer).width / fig.dpi * 72
        page.text(lx + width + TITLE_GAP, ly, title.format(section=section), ha="left",
                  va="center", fontsize=FS_LETTER, fontweight="bold", color=INK)

    return fig


def main():
    args = parse_args()
    sec, df, geo, tracks, coast = load(args.data)

    os.makedirs(args.out, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = make_figure(sec, df, geo, tracks, coast)
        for ext in ("pdf", "png"):
            path = os.path.join(args.out, f"{args.name}.{ext}")
            fig.savefig(path)
            print(f"Wrote {path}")
        plt.close(fig)

    obs = df["temperate_obs_m2"].sum() / 1e3
    mod = df["temperate_mod_m2"].sum() / 1e3
    print(f"\nsection panel: {sec.attrs['transect']} ({sec.attrs['radar_line']})")
    print(f"temperate ice: observed {obs:.0f}, modelled {mod:.0f} (10³ m²) — "
          f"{100 * mod / obs:.0f}% of observed")
    print(f"mean CTS RMSE over transects {df['cts_rmse_m'].mean():.2f} m")


if __name__ == "__main__":
    main()

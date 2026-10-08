#!/usr/bin/env python3
"""

===============================================================================
 Per-cell Manders' colocalization coefficients (M1, M2) in S. aureus
===============================================================================

Computes Manders' M1 and M2 (Manders, Verbeek & Aten, J. Microsc.
169:375-382, 1993) for every segmented cell in a three-channel confocal
image of Staphylococcus aureus.

CHANNELS
--------
    ch00   cell wall stain (vancomycin)  -> defines cell boundaries
    ch01   green channel (G)
    ch02   red channel   (R)

Only pixels inside ch00-segmented cells enter any calculation.

COEFFICIENTS
------------
    M1 = fraction of red (ch02) intensity on green-positive pixels
    M2 = fraction of green (ch01) intensity on red-positive pixels

Both are intensity-weighted and are not interchangeable. Pearson's r and the
overlap coefficient r are also reported per cell for context. If a cell has
no signal in a channel, the corresponding coefficient is left empty
(undefined) rather than set to 0 or 1.

All reported intensities come from the raw images; processed images are used
only to build masks.

THRESHOLDS (--threshold-method)
-------------------------------
  costes   (default) automatic; Costes et al., Biophys. J. 86:3993 (2004)
  otsu     Otsu threshold per channel
  fixed    user-supplied --t-red / --t-green; use when comparing images
  nonzero  Manders' literal "> 0"; not recommended for real data

One threshold pair is derived per image from the pooled in-cell pixels.
Coefficients near 1.0 usually mean the threshold is too permissive, not that
colocalization is perfect.

KEY OPTIONS
-----------
  --require-both    keep only cells positive in both channels; excluded cells
                    are written to *_excluded.csv and counted
  --expand-cells N  grow cell labels by N px (without merging neighbours) to
                    capture peripheral signal clipped by the wall threshold
  --register        correct the ch01/ch02 offset relative to ch00 by
                    integer-pixel shifts (the offset is always reported)
  --qc              save overlay, numbered cell map and intensity scatter

SEGMENTATION
------------
Cells are segmented from ch00 by an inverted, marker-controlled watershed:
dark lumens act as seeds and the bright wall as ridges, so septa split
dividing cells. Cells without a resolvable lumen are seeded from the
distance transform so they are not silently dropped. The fraction of cell
territory that receives a label is printed as a check (should be > ~95%).
The Otsu wall threshold was verified for this stain only; re-check it for
new stains or imaging conditions.

VALIDATION
----------
--self-test regenerates the synthetic test images of Manders et al. (1993)
and reproduces their Table 1.

USAGE
-----
    python manders_colocalization.py \
        --wall  dE-1_ch00_SV_16bit.tif \
        --green dE-1_ch01_SV_16bit.tif \
        --red   dE-1_ch02_SV_16bit.tif \
        -o results/ --px-nm 56.2 --scalebar-box 2000 2048 1900 2048 --qc

    python manders_colocalization.py --self-test

REQUIREMENTS
------------
    numpy, scipy, scikit-image, tifffile, matplotlib (only for --qc)

===============================================================================
===============================================================================
"""

import argparse
import csv
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.color import label2rgb


# =============================================================================
# Default parameters
# =============================================================================
P = dict(
    # --- preprocessing ---
    smooth_sigma=2.0,      # ~PSF width; denoises before thresholding
    bg_sigma=50.0,         # much larger than a cell: removes haze, keeps cells
    # --- ch00 cell segmentation ---
    wall_threshold="otsu",
    ring_close=1,
    min_interior=12,       # px^2; rescaled from the measured lumen size
    min_cell_area=80,      # px^2
    clear_border=False,
    # --- colocalization ---
    threshold_method="costes",
    scalebar_box=None,
)


# =============================================================================
# I/O and preprocessing
# =============================================================================
def load_plane(path, scalebar_box=None):
    """Read a single 2-D plane and blank any burned-in annotation."""
    a = tifffile.imread(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected one 2D plane, got shape {a.shape}")
    a = a.astype(np.float32)
    if scalebar_box is not None:
        y0, y1, x0, x1 = scalebar_box
        a[y0:y1, x0:x1] = 0
    return a


def flatten(raw, p):
    """
    Denoise and flatten for SEGMENTATION ONLY.

    Every intensity that enters a reported coefficient is read from the raw
    array. Processed images define masks; they never supply numbers.
    """
    return np.clip(
        filters.gaussian(raw, p["smooth_sigma"], preserve_range=True)
        - filters.gaussian(raw, p["bg_sigma"], preserve_range=True), 0, None)


def _threshold(img, spec):
    if isinstance(spec, (int, float)):
        return float(spec)
    return float(getattr(filters, f"threshold_{spec}")(img))


def _keep_larger_than(binary, min_size):
    lab, _ = ndi.label(binary)
    sizes = np.bincount(lab.ravel())
    return np.isin(lab, [i for i in range(1, len(sizes)) if sizes[i] >= min_size])


# =============================================================================
# Registration quality check
# =============================================================================
def apply_shift(img, dy, dx):
    """Integer-pixel shift by slicing; vacated edge zero-filled."""
    out = np.zeros_like(img)
    ys_src = slice(max(0, -dy), img.shape[0] - max(0, dy))
    ys_dst = slice(max(0, dy), img.shape[0] - max(0, -dy))
    xs_src = slice(max(0, -dx), img.shape[1] - max(0, dx))
    xs_dst = slice(max(0, dx), img.shape[1] - max(0, -dx))
    out[ys_dst, xs_dst] = img[ys_src, xs_src]
    return out


def measure_offset(moving, reference, max_shift=60):
    """
    Integer (dy, dx) aligning `moving` onto `reference`, by FFT
    cross-correlation of mean-centred images.

    Returns (dy, dx, sharpness) where sharpness is the correlation peak
    relative to the best competing peak outside a small exclusion radius.
    Values near 1 mean the match is ambiguous and the estimate is unreliable.
    """
    m = moving - moving.mean()
    f = reference - reference.mean()
    corr = fftconvolve(f, m[::-1, ::-1], mode="same")
    H, W = corr.shape
    cy, cx = H // 2, W // 2
    win = corr[cy - max_shift:cy + max_shift + 1,
               cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    peak = win[iy, ix]
    masked = win.copy()
    masked[max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9] = -np.inf
    runner = masked.max()
    return int(iy - max_shift), int(ix - max_shift), float(peak / max(runner, 1e-9))


# =============================================================================
# Cell segmentation from the wall channel (ch00)
# =============================================================================
def estimate_lumen(wall, p):
    """Median ring-enclosed lumen area, used to scale the pixel cut-offs."""
    ring = wall > _threshold(wall, p["wall_threshold"])
    rc = morphology.closing(ring, morphology.disk(p["ring_close"]))
    lab, _ = ndi.label(ndi.binary_fill_holes(rc) & ~rc)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes >= 20]
    return float(np.median(sizes)) if sizes.size else None


def autoscale(p, lumen):
    """
    Scale area cut-offs to the measured cell size so the script transfers
    between strains and magnifications without hand-tuning.
    """
    p = dict(p)
    p["min_interior"] = max(12, int(0.15 * lumen))
    p["min_cell_area"] = max(80, int(1.5 * lumen))
    return p


def _estimate_cell_radius(territory):
    """
    Typical cell radius in px, from compact single-cell components.

    Components that are nearly convex (solidity > 0.9) are single cells
    rather than clusters, so their median area gives a size reference that
    does not need to be supplied per image.
    """
    lab = measure.label(territory)
    areas = [r.area for r in measure.regionprops(lab)
             if r.solidity > 0.9 and r.area > 20]
    if not areas:
        areas = [r.area for r in measure.regionprops(lab)] or [100]
    return float(np.sqrt(np.median(areas) / np.pi))


def segment_cells(wall, p):
    """Inverted marker-controlled watershed; see module docstring."""
    from skimage.feature import peak_local_max

    thr = _threshold(wall, p["wall_threshold"])
    ring = wall > thr
    ring_closed = morphology.closing(ring, morphology.disk(p["ring_close"]))

    interiors = _keep_larger_than(
        ndi.binary_fill_holes(ring_closed) & ~ring_closed, p["min_interior"])

    # No fill_holes on this union -- see module docstring.
    territory = _keep_larger_than(ring_closed | interiors, p["min_cell_area"])

    # --- fallback seeding ---
    # Any territory component without a seed would be assigned label 0 by the
    # watershed and vanish. Find those components and seed them from the
    # distance transform instead.
    seeds = interiors & territory
    terr_lab = measure.label(territory)
    seeded_ids = set(np.unique(terr_lab[seeds])) - {0}
    all_ids = set(range(1, terr_lab.max() + 1))
    unseeded = all_ids - seeded_ids
    n_rescued = 0
    if unseeded:
        gap = np.isin(terr_lab, list(unseeded))
        radius = _estimate_cell_radius(territory)
        dist = filters.gaussian(ndi.distance_transform_edt(gap), 1.0)
        coords = peak_local_max(dist, min_distance=max(3, int(1.1 * radius)),
                                labels=gap, exclude_border=False)
        extra = np.zeros_like(seeds)
        if len(coords):
            extra[tuple(coords.T)] = True
            extra = ndi.binary_dilation(extra, np.ones((3, 3))) & gap
        # Any component still unseeded (too small for a distance peak) gets
        # its single brightest-distance pixel, so nothing is ever dropped.
        still = set(unseeded) - set(np.unique(terr_lab[extra])) - {0}
        for cid in still:
            m = terr_lab == cid
            idx = np.unravel_index(np.argmax(dist * m), dist.shape)
            extra[idx] = True
        seeds = seeds | extra
        n_rescued = len(unseeded)

    markers = measure.label(seeds)
    cells = measure.label(
        segmentation.watershed(wall, markers, mask=territory,
                               watershed_line=False))
    if p["clear_border"]:
        cells = measure.label(segmentation.clear_border(cells))

    coverage = 100.0 * (cells > 0).sum() / max(territory.sum(), 1)
    return cells, thr, n_rescued, coverage


def renumber_raster(labels):
    """
    Relabel 1..N in reading order (top to bottom, then left to right).

    Rows are banded by the median object diameter before sorting, so cells
    sitting side by side get consecutive numbers instead of being interleaved
    by sub-pixel differences in centroid row. The numbering in the CSV, the
    label image and the annotated figure are then the same.
    """
    props = measure.regionprops(labels)
    if not props:
        return labels.copy()
    band = max(1, int(np.median([pr.equivalent_diameter_area for pr in props])))
    order = sorted(props, key=lambda pr: (int(pr.centroid[0] // band), pr.centroid[1]))
    out = np.zeros_like(labels)
    for new_id, pr in enumerate(order, start=1):
        out[labels == pr.label] = new_id
    return out


# =============================================================================
# Thresholds for "positive" signal
# =============================================================================
def costes_thresholds(red, green, max_steps=400):
    """
    Costes automatic threshold (Costes et al., Biophys. J. 86:3993, 2004).

    Fit the orthogonal-regression (total least squares) line G = a*R + b
    through the 2-D intensity scatter, then walk the threshold down that line.
    At each step compute Pearson's r using ONLY the pixels below both
    thresholds. Stop at the highest threshold for which that sub-threshold
    population has r <= 0, i.e. the point where everything excluded is
    uncorrelated and therefore indistinguishable from noise.

    Orthogonal rather than ordinary least squares because neither channel is
    an independent variable here -- both carry measurement error, and OLS
    would bias the slope toward zero and the thresholds with it.

    Returns (t_red, t_green, slope, intercept).
    """
    R = red.astype(np.float64).ravel()
    G = green.astype(np.float64).ravel()
    if R.size < 10:
        return 0.0, 0.0, 1.0, 0.0

    mR, mG = R.mean(), G.mean()
    vR, vG = R.var(), G.var()
    cov = ((R - mR) * (G - mG)).mean()
    if abs(cov) < 1e-12:
        return float(R.max()), float(G.max()), 1.0, 0.0

    # Total-least-squares slope
    a = (vG - vR + np.sqrt((vG - vR) ** 2 + 4 * cov ** 2)) / (2 * cov)
    b = mG - a * mR

    def pearson(x, y):
        if x.size < 3:
            return 0.0
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))

    hi = float(R.max())
    lo = float(R.min())
    step = (hi - lo) / max_steps if hi > lo else 1.0
    t_red = hi
    for _ in range(max_steps):
        t_green = a * t_red + b
        below = (R < t_red) & (G < t_green)
        if below.sum() > 10 and pearson(R[below], G[below]) <= 0:
            break
        t_red -= step
        if t_red <= lo:
            t_red = lo
            break
    return float(max(t_red, 0.0)), float(max(a * t_red + b, 0.0)), float(a), float(b)


def derive_thresholds(red_vals, green_vals, method, t_red=None, t_green=None):
    """
    One threshold pair per image, computed from the pooled in-cell pixels.

    Per-cell thresholds would be derived from a few hundred pixels each and
    would differ between cells, making the per-cell coefficients
    incomparable. Restricting to in-cell pixels (rather than the whole frame)
    keeps the vast empty background from dominating the estimate.
    """
    if method == "fixed":
        if t_red is None or t_green is None:
            raise SystemExit("--threshold-method fixed requires --t-red and --t-green")
        return float(t_red), float(t_green), {}
    if method == "nonzero":
        return 0.0, 0.0, {}
    if method == "otsu":
        tr = float(filters.threshold_otsu(red_vals)) if red_vals.size else 0.0
        tg = float(filters.threshold_otsu(green_vals)) if green_vals.size else 0.0
        return tr, tg, {}
    if method == "costes":
        tr, tg, a, b = costes_thresholds(red_vals, green_vals)
        # Costes degenerates when one channel is nearly empty: the
        # total-least-squares line becomes near-vertical, so stepping down in
        # R sweeps the green threshold far above the data and every pixel is
        # scored negative. Detect that and refuse to return it silently.
        warn = None
        if green_vals.size and tg > np.percentile(green_vals, 99.9):
            warn = (f"Costes green threshold {tg:.1f} exceeds the 99.9th "
                    f"percentile of in-cell green "
                    f"({np.percentile(green_vals, 99.9):.1f}); "
                    f"regression slope {a:.2f}")
        if red_vals.size and tr > np.percentile(red_vals, 99.9):
            warn = (f"Costes red threshold {tr:.1f} exceeds the 99.9th "
                    f"percentile of in-cell red "
                    f"({np.percentile(red_vals, 99.9):.1f}); "
                    f"regression slope {a:.2f}")
        return tr, tg, {"costes_slope": a, "costes_intercept": b,
                        "warn": warn} if warn else {"costes_slope": a,
                                                    "costes_intercept": b}
    raise SystemExit(f"unknown threshold method {method!r}")


# =============================================================================
# Coefficients
# =============================================================================
def manders(red, green, t_red, t_green):
    """
    Manders' M1 and M2 over the supplied pixel vectors.

    A pixel is "positive" in a channel when its RAW intensity exceeds that
    channel's threshold; Manders' original "> 0" corresponds to t = 0.

        M1 = sum of red  intensity on green-positive pixels / total red
        M2 = sum of green intensity on red-positive pixels  / total green

    Returns None for a coefficient whose denominator is zero -- that cell has
    no signal in the relevant channel and the quantity is undefined, not 0.
    """
    sum_r = float(red.sum())
    sum_g = float(green.sum())
    g_pos = green > t_green
    r_pos = red > t_red
    m1 = float(red[g_pos].sum()) / sum_r if sum_r > 0 else None
    m2 = float(green[r_pos].sum()) / sum_g if sum_g > 0 else None
    return m1, m2, sum_r, sum_g, int(r_pos.sum()), int(g_pos.sum())


def pearson_and_overlap(red, green):
    """
    Pearson's r_p (Eq. 1) and the overlap coefficient r (Eq. 2), unthresholded.

    Reported for context only. Pearson's subtracts the means and so ranges
    -1 to 1; the overlap coefficient does not and ranges 0 to 1 but is
    confounded by the ratio of object numbers between channels, which is the
    limitation M1/M2 were introduced to solve.
    """
    R = red.astype(np.float64)
    G = green.astype(np.float64)
    if R.size < 3:
        return None, None
    sR, sG = R.std(), G.std()
    rp = (float(((R - R.mean()) * (G - G.mean())).mean() / (sR * sG))
          if sR > 1e-12 and sG > 1e-12 else None)
    denom = np.sqrt((R ** 2).sum() * (G ** 2).sum())
    ov = float((R * G).sum() / denom) if denom > 0 else None
    return rp, ov


# =============================================================================
# Per-cell measurement
# =============================================================================
def measure_cells(cells, wall_raw, green_raw, red_raw, t_red, t_green, px_nm):
    scale = (px_nm / 1000.0) if px_nm else None
    rows = []
    for pr in measure.regionprops(cells):
        sl = pr.slice
        m = cells[sl] == pr.label
        g = green_raw[sl][m]
        r = red_raw[sl][m]
        w = wall_raw[sl][m]

        m1, m2, sum_r, sum_g, n_rpos, n_gpos = manders(r, g, t_red, t_green)
        rp, ov = pearson_and_overlap(r, g)

        row = dict(
            cell_id=pr.label,
            centroid_y=round(pr.centroid[0], 2),
            centroid_x=round(pr.centroid[1], 2),
            area_px=int(pr.area),
            equiv_diameter_px=round(pr.equivalent_diameter_area, 2),
            circularity=round(
                min(4 * np.pi * pr.area / max(pr.perimeter, 1e-9) ** 2, 1.0), 3),
            # --- the coefficients ---
            M1_red_on_green=(round(m1, 4) if m1 is not None else ""),
            M2_green_on_red=(round(m2, 4) if m2 is not None else ""),
            # --- supporting numbers, so the above can be audited ---
            sum_red_ch02=round(sum_r, 1),
            sum_green_ch01=round(sum_g, 1),
            n_px_red_positive=n_rpos,
            n_px_green_positive=n_gpos,
            frac_area_red_positive=round(n_rpos / pr.area, 4),
            frac_area_green_positive=round(n_gpos / pr.area, 4),
            mean_red_ch02=round(float(r.mean()), 3),
            mean_green_ch01=round(float(g.mean()), 3),
            mean_wall_ch00=round(float(w.mean()), 3),
            # --- context coefficients ---
            pearson_r=(round(rp, 4) if rp is not None else ""),
            overlap_r=(round(ov, 4) if ov is not None else ""),
        )
        if scale:
            row["area_um2"] = round(pr.area * scale ** 2, 4)
            row["equiv_diameter_um"] = round(pr.equivalent_diameter_area * scale, 4)
        rows.append(row)
    return rows


# =============================================================================
# Outputs
# =============================================================================
def write_csv(rows, path, header_comment=None):
    if not rows:
        print(f"  (no cells; {path} not written)")
        return
    with open(path, "w", newline="") as fh:
        if header_comment:
            for line in header_comment.strip().splitlines():
                fh.write(f"# {line.strip()}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_labels(labels, path):
    if labels.max() > 65535:
        raise ValueError("more than 65535 cells; switch to uint32")
    tifffile.imwrite(path, labels.astype(np.uint16), photometric="minisblack")


def save_overlay(wall_raw, green_raw, red_raw, cells, path):
    """Composite with cell outlines: red = ch02, green = ch01, grey = ch00."""
    def norm(a, pct=99.7):
        return exposure.rescale_intensity(
            a.astype(np.float32), in_range=(0, max(np.percentile(a, pct), 1)))
    rgb = np.dstack([
        np.clip(norm(red_raw) + 0.45 * norm(wall_raw), 0, 1),
        np.clip(norm(green_raw) + 0.45 * norm(wall_raw), 0, 1),
        0.45 * norm(wall_raw)])
    rgb[segmentation.find_boundaries(cells, mode="outer")] = (1.0, 1.0, 1.0)
    tifffile.imwrite(path, (rgb * 255).astype(np.uint8), photometric="rgb")


def save_numbered(wall_raw, cells, path, dpi=200):
    """
    Index map: every cell annotated with the cell_id used in the CSV.

    Drawn with matplotlib rather than burned into the TIFF, because burnt-in
    text would corrupt the pixel values of any later measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    disp = exposure.rescale_intensity(
        wall_raw.astype(np.float32),
        in_range=(0, max(np.percentile(wall_raw, 99.5), 1)))
    rgb = (label2rgb(cells, image=disp, bg_label=0, alpha=0.3) * 255).astype(np.uint8)
    h, w = cells.shape
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="black")
    ax.imshow(rgb, interpolation="nearest")
    for pr in measure.regionprops(cells):
        y, x = pr.centroid
        ax.text(x, y, str(pr.label), color="#00ffff", fontsize=3.2,
                ha="center", va="center", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=0.9, foreground="black")])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_position([0, 0, 1, 1])
    fig.savefig(path, dpi=dpi * 2, facecolor="black")
    plt.close(fig)


def save_scatter(red_vals, green_vals, t_red, t_green, path):
    """
    2-D intensity histogram of the in-cell pixels with the thresholds drawn.

    This is the diagnostic Manders et al. use throughout their Fig. 2. The
    four quadrants are directly interpretable: upper-right is the colocalizing
    population, upper-left and lower-right are single-channel signal, and
    lower-left is what the threshold discards.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=160)
    hb = ax.hexbin(red_vals, green_vals, gridsize=60, bins="log",
                   cmap="magma", mincnt=1)
    ax.axvline(t_red, color="#00e5ff", lw=1.4, ls="--")
    ax.axhline(t_green, color="#00e5ff", lw=1.4, ls="--")
    ax.text(t_red, ax.get_ylim()[1], f" t_red={t_red:.1f}", color="#00e5ff",
            fontsize=8, va="top", ha="left")
    ax.text(ax.get_xlim()[1], t_green, f"t_green={t_green:.1f} ", color="#00e5ff",
            fontsize=8, va="bottom", ha="right")
    # Quadrant populations: upper-right is the colocalizing population that
    # drives both coefficients.
    q = dict(
        both=int(((red_vals > t_red) & (green_vals > t_green)).sum()),
        red_only=int(((red_vals > t_red) & (green_vals <= t_green)).sum()),
        green_only=int(((red_vals <= t_red) & (green_vals > t_green)).sum()),
        neither=int(((red_vals <= t_red) & (green_vals <= t_green)).sum()))
    tot = max(sum(q.values()), 1)
    ax.set_xlabel(
        f"ch02 red intensity (raw)\nquadrants: both {100*q['both']/tot:.0f}%, "
        f"red only {100*q['red_only']/tot:.0f}%, "
        f"green only {100*q['green_only']/tot:.0f}%, "
        f"neither {100*q['neither']/tot:.0f}%", fontsize=8)
    ax.set_ylabel("ch01 green intensity (raw)")
    ax.set_title("In-cell pixels, with thresholds", fontsize=10, loc="left")
    fig.colorbar(hb, ax=ax, label="pixels (log)")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# =============================================================================
# Self-test against Manders et al. (1993) Table 1
# =============================================================================
def _gaussian_field(positions, size=256, fwhm=12.0, trunc_radius=11.0):
    """
    Gaussian objects of 12 px FWHM, as in their Test images section.

    The Gaussians are truncated to a finite radius. This is not cosmetic: an
    untruncated Gaussian is non-zero everywhere, so with Manders' literal
    "> 0" criterion every pixel of every image is positive and M1 = M2 = 1.00
    for all nine pairs -- which is precisely the failure the paper documents
    for degraded images (their Fig. 2D-F). The published Table 1 presupposes
    that background is exactly zero, so the test objects must have compact
    support. A radius of 11 px is just under half the 22.6 px separation
    between an A-grid position and the nearest E-grid position, so objects
    that are not meant to coincide genuinely share no pixels.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    img = np.zeros((size, size), np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    for (cy, cx) in positions:
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        g = np.exp(-(d2 / (2 * sigma ** 2)))
        g[d2 > trunc_radius ** 2] = 0.0
        img += g
    return img


def self_test():
    """
    Rebuild their Fig. 1 test patterns and reproduce Table 1.

    Image A: 36 objects on a 6x6 grid at 32 px spacing.
    Image E: the same objects shifted 16 px right and 16 px down, so no
             object overlaps one in A.
    B, C, D: mixtures of A-positions and E-positions, giving 27, 18 and 9
             colocalizing objects respectively.
    F, G, H, I: subsets with fewer green objects.

    Expected (their Table 1), with R = image A throughout:

        pair   n_red  n_green  n_coloc   M1     M2
        A/A      36      36       36    1.00   1.00
        A/B      36      36       27    0.75   0.75
        A/C      36      36       18    0.50   0.50
        A/D      36      36        9    0.25   0.25
        A/E      36      36        0    0.00   0.00
        A/F      36      27        9    0.25   0.33
        A/G      36      18        9    0.25   0.50
        A/H      36       9        9    0.25   1.00
        A/I      36       4        3    0.08   0.75
    """
    base = [(16 + 32 * i, 16 + 32 * j) for i in range(6) for j in range(6)]
    shifted = [(y + 16, x + 16) for (y, x) in base]

    def mix(n_from_base):
        """n objects at A-positions, the rest at E-positions."""
        return base[:n_from_base] + shifted[n_from_base:]

    patterns = {
        "A": base,
        "B": mix(27),
        "C": mix(18),
        "D": mix(9),
        "E": shifted,
        "F": base[:9] + shifted[9:27],        # 27 objects, 9 colocalizing
        "G": base[:9] + shifted[9:18],        # 18 objects, 9 colocalizing
        "H": base[:9],                        # 9 objects, all colocalizing
        "I": base[:3] + shifted[3:4],         # 4 objects, 3 colocalizing
    }
    expected = {
        "A": (36, 36, 1.00, 1.00), "B": (36, 27, 0.75, 0.75),
        "C": (36, 18, 0.50, 0.50), "D": (36, 9, 0.25, 0.25),
        "E": (36, 0, 0.00, 0.00),  "F": (27, 9, 0.25, 0.33),
        "G": (18, 9, 0.25, 0.50),  "H": (9, 9, 0.25, 1.00),
        "I": (4, 3, 0.08, 0.75),
    }

    red = _gaussian_field(patterns["A"])
    print("Self-test against Manders et al. (1993) Table 1")
    print(f"{'pair':>6} {'n_green':>8} {'n_coloc':>8} "
          f"{'M1':>7} {'exp':>6} {'M2':>7} {'exp':>6}  {'':>4}")
    ok = True
    for key in "ABCDEFGHI":
        green = _gaussian_field(patterns[key])
        # Threshold at 0 reproduces Manders' literal definition on these
        # noise-free synthetic images, where background is exactly zero.
        m1, m2, *_ = manders(red.ravel(), green.ravel(), 0.0, 0.0)
        ng, nc, e1, e2 = expected[key]
        good = abs(m1 - e1) < 0.02 and abs(m2 - e2) < 0.02
        ok &= good
        print(f"   A/{key} {ng:>8} {nc:>8} {m1:>7.2f} {e1:>6.2f} "
              f"{m2:>7.2f} {e2:>6.2f}  {'OK' if good else 'FAIL':>4}")
    print("\nAll values match the published table."
          if ok else "\nMISMATCH - do not use these results.")
    return 0 if ok else 1


# =============================================================================
# Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Per-cell Manders M1/M2 for a three-channel S. aureus image.")
    ap.add_argument("--wall", help="ch00 TIFF, cell wall stain")
    ap.add_argument("--green", help="ch01 TIFF, green channel (G)")
    ap.add_argument("--red", help="ch02 TIFF, red channel (R)")
    ap.add_argument("-o", "--output", default=".")
    ap.add_argument("--threshold-method", default=P["threshold_method"],
                    choices=["costes", "otsu", "fixed", "nonzero"])
    ap.add_argument("--t-red", type=float, default=None)
    ap.add_argument("--t-green", type=float, default=None)
    ap.add_argument("--wall-threshold", default=P["wall_threshold"])
    ap.add_argument("--ring-close", type=int, default=P["ring_close"])
    ap.add_argument("--no-auto-size", action="store_true")
    ap.add_argument("--clear-border", action="store_true",
                    help="drop cells touching the image edge")
    ap.add_argument("--px-nm", type=float, default=None)
    ap.add_argument("--scalebar-box", nargs=4, type=int, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--require-both", action="store_true",
                    help="keep only cells with threshold-positive signal in "
                         "BOTH channels; see module docstring")
    ap.add_argument("--min-positive-px", type=int, default=1,
                    help="pixels above threshold required in each channel "
                         "when --require-both is set (default 1)")
    ap.add_argument("--expand-cells", type=int, default=0,
                    help="grow each cell label outward by N px to capture "
                         "peripheral arcs clipped by the wall threshold "
                         "(uses expand_labels, so cells never merge)")
    ap.add_argument("--register", action="store_true",
                    help="correct the measured ch01/ch02 offset against ch00 "
                         "by integer-pixel shifting before measuring")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())
    if not (a.wall and a.green and a.red):
        ap.error("--wall, --green and --red are all required "
                 "(or use --self-test)")

    p = dict(P)
    p["ring_close"] = a.ring_close
    p["clear_border"] = a.clear_border
    p["scalebar_box"] = tuple(a.scalebar_box) if a.scalebar_box else None
    try:
        p["wall_threshold"] = float(a.wall_threshold)
    except ValueError:
        p["wall_threshold"] = a.wall_threshold

    os.makedirs(a.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.wall))[0]
    for suffix in ("_ch00_SV_16bit", "_ch00_SV", "_ch00"):
        stem = stem.replace(suffix, "")
    base = os.path.join(a.output, stem)

    # ---- load ----
    wall_raw = load_plane(a.wall, p["scalebar_box"])
    green_raw = load_plane(a.green, p["scalebar_box"])
    red_raw = load_plane(a.red, p["scalebar_box"])

    print(f"{stem}")
    for name, arr in (("ch00 wall", wall_raw), ("ch01 green", green_raw),
                      ("ch02 red", red_raw)):
        print(f"  {name:11s}: max={arr.max():.0f} "
              f"levels={len(np.unique(arr))} frac>0={(arr > 0).mean():.4f}")

    # ---- registration QC ----
    wall_f = flatten(wall_raw, p)
    ring = wall_f > _threshold(wall_f, p["wall_threshold"])
    filled = ndi.binary_fill_holes(
        morphology.closing(ring, morphology.disk(p["ring_close"]))).astype(np.float32)
    dy_g, dx_g, s_g = measure_offset(flatten(green_raw, p), filled)
    dy_r, dx_r, s_r = measure_offset(flatten(red_raw, p), filled)
    dy_c, dx_c, _ = measure_offset(flatten(red_raw, p), flatten(green_raw, p))
    print(f"  registration : ch01 ({dy_g:+d},{dx_g:+d})  "
          f"ch02 ({dy_r:+d},{dx_r:+d})  "
          f"ch02->ch01 measured ({dy_c:+d},{dx_c:+d}) vs "
          f"predicted ({dy_r - dy_g:+d},{dx_r - dx_g:+d})")
    worst = max(abs(dy_g), abs(dx_g), abs(dy_r), abs(dx_r))
    if a.register:
        green_raw = apply_shift(green_raw, dy_g, dx_g)
        red_raw = apply_shift(red_raw, dy_r, dx_r)
        print(f"  registered   : ch01 shifted ({dy_g:+d},{dx_g:+d}), "
              f"ch02 shifted ({dy_r:+d},{dx_r:+d}) by integer-pixel slicing; "
              f"outer {worst} px now zero-filled on at least one channel")
    elif worst > 3:
        print("  ! channels are offset by more than 3 px. Colocalization is "
              "very sensitive to this; rerun with --register, or register "
              "upstream, before trusting M1/M2.")

    # ---- segment ----
    if not a.no_auto_size:
        lum = estimate_lumen(wall_f, p)
        if lum:
            p = autoscale(p, lum)
            print(f"  cell scale   : median lumen {lum:.0f} px^2 -> "
                  f"min_interior={p['min_interior']}, "
                  f"min_cell={p['min_cell_area']}")
    cells, wall_thr, n_rescued, coverage = segment_cells(wall_f, p)
    cells = renumber_raster(cells)
    n_cells = int(cells.max())
    print(f"  segmentation : wall threshold {wall_thr:.2f} -> {n_cells} cells "
          f"({n_rescued} components seeded from the distance transform)")
    print(f"  coverage     : {coverage:.1f}% of cell territory labelled")
    if coverage < 95:
        print("  ! more than 5% of the segmented cell area carries no label; "
              "cells are being dropped and the coefficients are computed on a "
              "biased subset.")

    # How much of each channel the mask actually contains, before and after
    # expansion. Printed always, because a low "inside" fraction means the
    # coefficients are being computed on a truncated signal.
    def inside_frac(mask):
        return (100 * green_raw[mask].sum() / max(green_raw.sum(), 1e-9),
                100 * red_raw[mask].sum() / max(red_raw.sum(), 1e-9))
    gi, ri = inside_frac(cells > 0)
    print(f"  signal inside: green {gi:.1f}%  red {ri:.1f}%  "
          f"of total frame intensity (unexpanded mask)")

    if a.expand_cells > 0:
        cells = segmentation.expand_labels(cells, distance=a.expand_cells)
        gi2, ri2 = inside_frac(cells > 0)
        print(f"  expanded {a.expand_cells} px : green {gi2:.1f}%  red {ri2:.1f}%  "
              f"(cell area now {100 * (cells > 0).mean():.2f}% of frame)")

    # ---- thresholds from pooled in-cell pixels ----
    incell = cells > 0
    r_vals = red_raw[incell]
    g_vals = green_raw[incell]
    t_red, t_green, extra = derive_thresholds(
        r_vals, g_vals, a.threshold_method, a.t_red, a.t_green)
    print(f"  thresholds   : method={a.threshold_method}  "
          f"ch02 red={t_red:.2f}  ch01 green={t_green:.2f}")
    if extra:
        print(f"                 Costes line: G = {extra['costes_slope']:.3f}*R "
              f"+ {extra['costes_intercept']:.3f}")
        if extra.get("warn"):
            print(f"  ! {extra['warn']}")
            print("    Costes has failed on this field -- one channel is too "
                  "sparse for the regression. Use --threshold-method fixed "
                  "or otsu; do not report these coefficients.")
    print(f"  in-cell px   : {incell.sum()} "
          f"({100 * incell.mean():.2f}% of frame); "
          f"red-positive {100 * (r_vals > t_red).mean():.1f}%, "
          f"green-positive {100 * (g_vals > t_green).mean():.1f}%")

    # ---- per-cell coefficients ----
    rows = measure_cells(cells, wall_raw, green_raw, red_raw,
                         t_red, t_green, a.px_nm)

    excluded = []
    if a.require_both:
        k = a.min_positive_px
        keep = [r for r in rows
                if r["n_px_red_positive"] >= k and r["n_px_green_positive"] >= k]
        excluded = [r for r in rows if r not in keep]
        n_tot = len(rows)
        n_red_only = sum(1 for r in excluded
                         if r["n_px_red_positive"] >= k)
        n_green_only = sum(1 for r in excluded
                           if r["n_px_green_positive"] >= k)
        n_neither = len(excluded) - n_red_only - n_green_only
        rows = keep
        print(f"  double-positive: {len(rows)}/{n_tot} cells kept "
              f"({100 * len(rows) / max(n_tot, 1):.1f}%); excluded "
              f"{n_red_only} red-only, {n_green_only} green-only, "
              f"{n_neither} neither  (>= {k} positive px required per channel)")
        if not rows:
            print("  ! no cell has signal in both channels; nothing to report")

    m1 = np.array([r["M1_red_on_green"] for r in rows
                   if r["M1_red_on_green"] != ""], float)
    m2 = np.array([r["M2_green_on_red"] for r in rows
                   if r["M2_green_on_red"] != ""], float)
    n_undef1 = sum(1 for r in rows if r["M1_red_on_green"] == "")
    n_undef2 = sum(1 for r in rows if r["M2_green_on_red"] == "")

    header = f"""
    Per-cell Manders colocalization coefficients
    source wall  = {os.path.basename(a.wall)}  (ch00, cell boundary)
    source green = {os.path.basename(a.green)} (ch01, G)
    source red   = {os.path.basename(a.red)}   (ch02, R)
    M1_red_on_green = fraction of ch02 red intensity on green-positive pixels
    M2_green_on_red = fraction of ch01 green intensity on red-positive pixels
    threshold method = {a.threshold_method}; t_red = {t_red:.4f}, t_green = {t_green:.4f}
    channel registration = {"applied: ch01 (%+d,%+d), ch02 (%+d,%+d)" % (dy_g, dx_g, dy_r, dx_r) if a.register else "measured only, not applied: ch01 (%+d,%+d), ch02 (%+d,%+d)" % (dy_g, dx_g, dy_r, dx_r)}
    all sums restricted to pixels inside a ch00-segmented cell
    cell filter = {"double-positive only, >= %d positive px per channel" % a.min_positive_px if a.require_both else "all segmented cells"}
    cell mask expanded by {a.expand_cells} px (expand_labels; 0 = no expansion)
    intensities read from raw images; processed images used only for masks
    empty M1/M2 = denominator zero (no signal in that channel), undefined
    """
    write_csv(rows, base + "_manders.csv", header)
    if excluded:
        write_csv(excluded, base + "_excluded.csv", header)
    save_labels(cells, base + "_cells.tif")
    if a.qc:
        save_overlay(wall_raw, green_raw, red_raw, cells, base + "_overlay.tif")
        save_numbered(wall_raw, cells, base + "_numbered.png")
        save_scatter(r_vals, g_vals, t_red, t_green, base + "_scatter.png")

    print(f"\n  M1 (red on green) : median {np.median(m1):.3f}  "
          f"mean {m1.mean():.3f}  IQR {np.percentile(m1, 25):.3f}-"
          f"{np.percentile(m1, 75):.3f}  (n={len(m1)}, {n_undef1} undefined)")
    print(f"  M2 (green on red) : median {np.median(m2):.3f}  "
          f"mean {m2.mean():.3f}  IQR {np.percentile(m2, 25):.3f}-"
          f"{np.percentile(m2, 75):.3f}  (n={len(m2)}, {n_undef2} undefined)")
    near1 = int(((m1 > 0.98).sum() + (m2 > 0.98).sum()))
    if near1 > 0.3 * (len(m1) + len(m2)):
        print("  ! many coefficients are at ~1.0, which usually means the "
              "threshold is too permissive rather than that colocalization "
              "is perfect (Manders et al. 1993, Fig. 2D-F).")
    print(f"  -> {base}_*")


if __name__ == "__main__":
    main()

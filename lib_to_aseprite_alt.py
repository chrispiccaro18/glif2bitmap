#!/usr/bin/env python3
import argparse, os, sys, subprocess, tempfile, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import plistlib
import math

# -------------------- small utils --------------------

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def which_any(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

def find_aseprite():
    # prefer app bundle (reliable on macOS if PATH is quirky)
    default = "/Applications/Aseprite.app/Contents/MacOS/aseprite"
    if Path(default).exists(): return default
    env = os.environ.get("ASEPRITE")
    if env and Path(env).exists(): return env
    w = shutil.which("aseprite")
    if w: return w
    return None

# -------------------- geometry helpers --------------------

def lerp(a, b, t): return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)

def dist2(a, b):
    dx, dy = a[0]-b[0], a[1]-b[1]
    return dx*dx + dy*dy

def flat_quad(p0, p1, p2, eps2, out):
    x0,y0=p0; x1,y1=p1; x2,y2=p2
    vx, vy = x2-x0, y2-y0
    if vx==0 and vy==0:
        if dist2(p0,p1) > eps2: out.append(p1)
        out.append(p2); return
    t = ((x1-x0)*vx + (y1-y0)*vy)/(vx*vx+vy*vy)
    proj = (x0+vx*t, y0+vy*t)
    dev2 = dist2(p1, proj)
    if dev2 <= eps2:
        out.append(p2); return
    p01 = lerp(p0,p1,0.5); p12 = lerp(p1,p2,0.5); p012 = lerp(p01,p12,0.5)
    flat_quad(p0, p01, p012, eps2, out)
    flat_quad(p012, p12, p2, eps2, out)

def flat_cubic(p0, p1, p2, p3, eps2, out):
    x0,y0=p0; x1,y1=p1; x2,y2=p2; x3,y3=p3
    vx, vy = x3-x0, y3-y0
    def dev2(p):
        if vx==0 and vy==0: return dist2(p0,p)
        t = ((p[0]-x0)*vx + (p[1]-y0)*vy)/(vx*vx+vy*vy)
        proj = (x0+vx*t, y0+vy*t)
        return dist2(p, proj)
    if max(dev2(p1), dev2(p2)) <= eps2:
        out.append(p3); return
    p01=lerp(p0,p1,0.5); p12=lerp(p1,p2,0.5); p23=lerp(p2,p3,0.5)
    p012=lerp(p01,p12,0.5); p123=lerp(p12,p23,0.5); p0123=lerp(p012,p123,0.5)
    flat_cubic(p0, p01, p012, p0123, eps2, out)
    flat_cubic(p0123, p123, p23, p3, eps2, out)

# -------------------- UFO / GLIF helpers --------------------

def read_contents_mapping(glyphs_dir: Path):
    p = glyphs_dir / "contents.plist"
    if not p.exists(): return {}
    with p.open("rb") as f:
        return plistlib.load(f)  # {glyphName: fileName.glif}

def resolve_glif_path(glyph_name: str, glyphs_dir: Path, contents_map: dict, allow_fallback=True):
    if glyph_name in contents_map:
        p = glyphs_dir / contents_map[glyph_name]
        return (p if p.exists() else None, "contents")
    p = glyphs_dir / f"{glyph_name}.glif"
    if p.exists(): return (p, "name")
    if not allow_fallback: return (None, "missing")
    p2 = glyphs_dir / f"{glyph_name}_.glif"  # common uppercase convention
    if p2.exists(): return (p2, "underscore")
    return (None, "missing")

def read_glyph_order(lib_plist: Path):
    with lib_plist.open('rb') as f:
        data = plistlib.load(f)
    order = data.get("public.glyphOrder", [])
    return [n for n in order if isinstance(n, str)]

def read_fontinfo(ufo_dir: Path):
    fi = (ufo_dir / "fontinfo.plist")
    if not fi.exists():
        # try sibling of lib.plist if --lib points into UFO subdir
        fi = (ufo_dir.parent / "fontinfo.plist")
        if not fi.exists(): return (1000, 800.0, -200.0, None)
    with fi.open("rb") as f:
        d = plistlib.load(f)
    upm = int(d.get("unitsPerEm", 1000))
    asc = float(d.get("ascender", 800))
    desc = float(d.get("descender", -200))
    cap = d.get("capHeight", None)
    cap = float(cap) if cap is not None else None
    return (upm, asc, desc, cap)

def get_monospace_advance(glyphs_dir: Path, contents_map: dict, order):
    # try common names first then fall back to first N ordered glyphs
    sample = [n for n in (["space","A","zero","a"] + list(order)) if n in order][:48]
    advs = []
    for name in sample:
        p,_ = resolve_glif_path(name, glyphs_dir, contents_map, True)
        if not p: continue
        try:
            tree = ET.parse(p)
            adv = tree.getroot().find("./advance")
            if adv is not None and "width" in adv.attrib:
                advs.append(float(adv.attrib["width"]))
        except Exception:
            pass
    if not advs: return None
    advs.sort()
    return advs[len(advs)//2]  # median

def parse_glif(glif_path: Path):
    """
    Return (contours, advance_width|None)
    contours: list[list[(x,y,type)]], type in {"line","curve","qcurve","offcurve"}
    """
    tree = ET.parse(glif_path)
    root = tree.getroot()
    adv = None
    adv_el = root.find("./advance")
    if adv_el is not None and "width" in adv_el.attrib:
        try: adv = float(adv_el.attrib["width"])
        except ValueError: adv = None
    outline = root.find("./outline")
    contours = []
    if outline is not None:
        for contour in outline.findall("./contour"):
            pts = []
            for pt in contour.findall("./point"):
                x = float(pt.attrib["x"]); y = float(pt.attrib["y"])
                t = pt.attrib.get("type", "offcurve")
                pts.append((x, y, t))
            if pts: contours.append(pts)
    return contours, adv

def bbox(contours):
    xs, ys = [], []
    for pts in contours:
        for x, y, _ in pts:
            xs.append(x); ys.append(y)
    if not xs: return (0,0,0,0)
    return (min(xs), min(ys), max(xs), max(ys))

# -------------------- path & svg --------------------

def contours_to_svg_path(contours, eps_px=0.5):
    """Flatten cubic & quadratic curves to line segments; flip Y for SVG."""
    eps2 = eps_px * eps_px
    def flipY(y): return -y
    parts = []
    for pts in contours:
        if not pts: continue
        # rotate to start at first on-curve
        start = 0
        for i,(_,_,t) in enumerate(pts):
            if t != "offcurve":
                start = i; break
        ordered = pts[start:] + pts[:start]
        n = len(ordered)
        if n == 0 or ordered[0][2] == "offcurve": continue

        curr = (ordered[0][0], ordered[0][1])
        poly = [curr]
        i = 0
        while i < n:
            off = []
            j = (i+1) % n
            while True:
                xj,yj,tj = ordered[j]
                if tj == "offcurve":
                    off.append((xj,yj))
                    j = (j+1) % n
                    if j == i: break
                else:
                    next_on = (xj,yj); next_type = tj
                    break
            if next_type == "line" or len(off) == 0:
                poly.append(next_on)
            elif next_type == "curve":
                if len(off) == 2:
                    out = []
                    flat_cubic(curr, off[0], off[1], next_on, eps2, out)
                    poly.extend(out)
                else:
                    poly.append(next_on)
            elif next_type == "qcurve":
                qpts = off[:] + [next_on]
                prev = curr; k = 0
                while k < len(qpts)-1:
                    c = qpts[k]
                    if k+1 < len(qpts)-1:
                        nctrl = qpts[k+1]
                        implied = ((c[0]+nctrl[0])/2.0, (c[1]+nctrl[1])/2.0)
                        out = []
                        flat_quad(prev, c, implied, eps2, out)
                        poly.extend(out)
                        prev = implied; k += 1
                    else:
                        onp = qpts[-1]
                        out = []
                        flat_quad(prev, c, onp, eps2, out)
                        poly.extend(out)
                        prev = onp; k += 1
            else:
                poly.append(next_on)
            curr = poly[-1]
            if j == i: break
            i = j
            if i == 0: break

        # dedupe
        dedup = [poly[0]]
        for p in poly[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        if len(dedup) >= 2:
            seg = [f"M{dedup[0][0]} {-dedup[0][1]}"]
            for (x,y) in dedup[1:]:
                seg.append(f"L{x} {-y}")
            seg.append("Z")
            parts.append(" ".join(seg))
    return " ".join(parts)

def write_svg_cell(svg_path: Path, contours, *,
                   asc: float, desc: float, cap: float|None, cap_height_px: float|None,
                   cell_w: int, cell_h: int,
                   pad_px: float, align: str, eps_px: float,
                   debug: bool=False, debug_baseline: bool=False,
                   fit: str="em", preset_scale: float|None=None):
    """Compose an SVG cell with baseline y=0, glyph path filled white on transparent."""
    d = contours_to_svg_path(contours, eps_px=eps_px)

    # uniform scale
    if preset_scale is not None:
        scale = preset_scale
    else:
        usable_h_px = max(1.0, cell_h - 2.0*pad_px)
        if fit == "cap" and cap:
            target_cap_px = cap_height_px or usable_h_px
            scale = float(target_cap_px) / float(cap)
        else:
            em_h = asc - desc
            scale = usable_h_px / float(em_h)

    pad_fu = pad_px / scale
    vw_fu = (cell_w - 2.0*pad_px) / scale
    vh_fu = (cell_h - 2.0*pad_px) / scale

    # baseline anchoring: viewBox y origin at -(asc + pad)
    vx0_fu = -pad_fu
    vy0_fu = -(asc + pad_fu)

    # horizontal alignment
    if contours:
        x0,_,x1,_ = bbox(contours)
    else:
        x0=x1=0.0
    gx_mid = 0.5*(x0+x1)
    if align == "left":
        vx0_fu = x0 - pad_fu
    elif align == "center":
        usable_center_fu = (vx0_fu + pad_fu) + vw_fu*0.5
        dx = gx_mid - usable_center_fu
        vx0_fu += dx

    # baseline debug line
    stroke_fu = 1.0 / max(1e-6, scale)
    baseline = ""
    if debug_baseline:
        x2_fu = vx0_fu + vw_fu + 2*pad_fu
        baseline = (f'<line x1="{vx0_fu}" y1="0" x2="{x2_fu}" y2="0" '
                    f'stroke="yellow" stroke-opacity="0.9" stroke-width="{stroke_fu}"/>')

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{cell_w}" height="{cell_h}"
     viewBox="{vx0_fu} {vy0_fu} {vw_fu + 2*pad_fu} {vh_fu + 2*pad_fu}">
  {baseline}
  <path d="{d}" fill="white" fill-rule="evenodd"/>
</svg>
'''
    svg_path.write_text(svg, encoding="utf-8")

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Convert lib.plist + .glif folder into an Aseprite sheet (pixel grid).")
    ap.add_argument("--lib", required=True, help="Path to lib.plist")
    ap.add_argument("--glyphs-dir", required=True, help="Folder containing .glif files")
    ap.add_argument("--out", required=True, help="Output .aseprite or .png path")
    ap.add_argument("--png-size", type=int, default=64, help="Default cell size if cell-* not given")
    ap.add_argument("--cell-width", type=int, default=None, help="Cell width (px)")
    ap.add_argument("--cell-height", type=int, default=None, help="Cell height (px)")
    ap.add_argument("--align", choices=["left","center"], default="left")
    ap.add_argument("--pad-px", type=float, default=0.0, help="Padding inside each cell")
    ap.add_argument("--flatten-eps", type=float, default=0.5, help="Bezier flatten tolerance in px")
    ap.add_argument("--oversample", type=int, default=1, help="Rasterize N× larger then downscale")
    ap.add_argument("--hard-threshold", action="store_true", help="After downscale, threshold alpha to hard pixels")
    ap.add_argument("--stroke-px", type=float, default=0.0, help="Optional morphological dilate radius in px")
    ap.add_argument("--map-out", help="Write Aseprite JSON/stdout to this file")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-baseline", action="store_true")
    # font fit & monospaced
    ap.add_argument("--fit", choices=["em","cap"], default="em", help="Vertical fit target (em default)")
    ap.add_argument("--cap-height-px", type=float, default=None, help="If --fit=cap, cap height in px")
    ap.add_argument("--derive-cell-from-advance", action="store_true",
                    help="For monospaced fonts, set cell width = round(advance_fu * scale)")
    ap.add_argument("--letterspace-px", type=float, default=0.0, help="Extra tracking to add to cell width")
    args = ap.parse_args()

    lib_path = Path(args.lib)
    glifs_dir = Path(args.glyphs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    contents_map = read_contents_mapping(glifs_dir)
    order = read_glyph_order(lib_path)
    if not order:
        print("ERROR: No public.glyphOrder found in lib.plist", file=sys.stderr); sys.exit(1)

    ufo_dir = lib_path.parent
    UPEM, ASC, DESC, CAP = read_fontinfo(ufo_dir)

    cell_h = args.cell_height or args.png_size
    cell_w = args.cell_width or args.png_size

    # vertical scale (uniform)
    def compute_scale():
        usable_h_px = max(1.0, cell_h - 2.0*args.pad_px)
        if args.fit == "cap" and CAP:
            target_cap_px = args.cap_height_px or usable_h_px
            return float(target_cap_px) / float(CAP)
        em_h = ASC - DESC
        return usable_h_px / float(em_h)

    SCALE = compute_scale()

    # monospaced: derive a single cell width from advance width
    if args.derive_cell_from_advance:
        adv_fu = get_monospace_advance(glifs_dir, contents_map, order)
        if adv_fu:
            cell_w = max(1, int(round(adv_fu * SCALE + args.letterspace_px)))
            if args.debug:
                print(f"[debug] derive-cell: upem={UPEM} asc={ASC} desc={DESC} cap={CAP}")
                print(f"[debug] derive-cell: ADV={adv_fu} fu, scale={SCALE:.4f} -> cell_w={cell_w}px")
        elif args.debug:
            print("[debug] derive-cell: no <advance width> found; keeping requested cell_w")

    # tool checks
    rsvg = which_any("rsvg-convert")
    if not rsvg:
        print("ERROR: rsvg-convert not found. Install with: brew install librsvg", file=sys.stderr); sys.exit(1)
    aseprite_bin = find_aseprite()
    if not aseprite_bin:
        print("ERROR: aseprite not found. Set ASEPRITE env or install.", file=sys.stderr); sys.exit(1)
    magick = which_any("magick") or which_any("convert")  # IM7 vs IM6

    tmp = Path(tempfile.mkdtemp(prefix="glif2ase_"))
    svgs_dir = tmp / "svgs"; svgs_dir.mkdir(parents=True, exist_ok=True)
    pngs_dir = tmp / "pngs"; pngs_dir.mkdir(parents=True, exist_ok=True)

    included_pngs = []
    try:
        seen = set()
        for name in order:
            glif_path, how = resolve_glif_path(name, glifs_dir, contents_map, True)
            if args.debug:
                print(f"[debug] resolve: {name} via {how} -> {glif_path}")
            if not glif_path:
                continue

            real = glif_path.resolve()
            if real in seen:
                if args.debug: print(f"[debug] de-dupe: {name} -> {real}")
                continue
            seen.add(real)

            contours, _adv = parse_glif(glif_path)
            if args.debug:
                cc = len(contours); pts = sum(len(c) for c in contours)
                has_c = any(any(t=="curve" for _,_,t in c) for c in contours)
                has_q = any(any(t=="qcurve" for _,_,t in c) for c in contours)
                print(f"[debug] glyph {name}: contours={cc}, points={pts}, cubic={has_c}, quad={has_q}")

            # write per-glyph SVG cell
            svg_path = svgs_dir / f"{name}.svg"
            write_svg_cell(svg_path, contours,
                           asc=ASC, desc=DESC, cap=CAP, cap_height_px=args.cap_height_px,
                           cell_w=cell_w, cell_h=cell_h,
                           pad_px=args.pad_px, align=args.align,
                           eps_px=args.flatten_eps,
                           debug=args.debug, debug_baseline=args.debug_baseline,
                           fit=args.fit, preset_scale=SCALE)

            # rasterize to PNG (transparent)
            png_big = pngs_dir / f"{name}_x{args.oversample}.png"
            w_big = cell_w * max(1, args.oversample)
            cmd = [rsvg, "--width", str(w_big), "--keep-aspect-ratio",
                   "-o", str(png_big), str(svg_path)]
            rc, out, err = run(cmd)
            if rc != 0:
                print(f"ERROR: rsvg-convert failed for {name}: {err}", file=sys.stderr); sys.exit(1)

            # downscale & threshold / stroke (ImageMagick if present)
            png_final = pngs_dir / f"{name}.png"
            if args.oversample > 1 or args.hard_threshold or args.stroke_px > 0:
                if not magick:
                    # fallback: just copy; you won't get threshold/ stroke
                    shutil.copyfile(png_big, png_final)
                else:
                    # build IM command
                    im = [magick, str(png_big)]
                    # downscale back to cell_w with nearest-neighbor (preserve pixels)
                    im += ["-filter", "point", "-resize", f"{cell_w}x{cell_h}!"]
                    if args.hard_threshold:
                        # force alpha to hard edges: anything >0 becomes 100%
                        im += ["-alpha", "extract", "-threshold", "0", "-alpha", "on"]
                    if args.stroke_px > 0:
                        # IM7 syntax prefers: -morphology Dilate Octagon:N
                        # approximate radius via Octagon:N (N ~ pixels)
                        N = max(1, int(round(args.stroke_px)))
                        im += ["-morphology", f"Dilate:1", f"Octagon:{N}"]
                    im += [str(png_final)]
                    rc, out, err = run(im)
                    if rc != 0:
                        print(f"ERROR: ImageMagick failed for {name}: {err}", file=sys.stderr); sys.exit(1)
            else:
                shutil.copyfile(png_big, png_final)

            included_pngs.append(str(png_final))

        if not included_pngs:
            print("ERROR: No glyphs were included.", file=sys.stderr); sys.exit(1)

        # Build horizontal sheet with no gaps (transparent)
        tmp_sheet = tmp / "sheet.png"
        sheet_cmd = [aseprite_bin, "-b", *included_pngs,
                     "--sheet", str(tmp_sheet),
                     "--sheet-type", "horizontal",
                     "--sheet-columns", str(len(included_pngs)),
                     "--sheet-rows", "1",
                     "--sheet-border-padding", "0",
                     "--sheet-inner-padding", "0",
                     "--sheet-spacing", "0",
                     "--sheet-transparent-color", "0", "0", "0", "0"]
        rc, out, err = run(sheet_cmd)
        if rc != 0:
            print(f"ERROR: aseprite sheet failed: {err}", file=sys.stderr); sys.exit(1)

        # Write .aseprite or .png
        if out_path.suffix.lower() == ".aseprite":
            rc, out2, err2 = run([aseprite_bin, "-b", str(tmp_sheet), "--save-as", str(out_path)])
            if rc != 0:
                print(f"ERROR: saving .aseprite failed: {err2}", file=sys.stderr); sys.exit(1)
        else:
            shutil.copyfile(tmp_sheet, out_path)

        if args.map_out:
            Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.map_out).write_text(out + "\n", encoding="utf-8")

        print(f"✅ Wrote {out_path}")
        if args.debug:
            print("aseprite cmd:", " ".join(sheet_cmd))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()

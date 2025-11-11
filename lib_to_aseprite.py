#!/usr/bin/env python3
import argparse, os, sys, subprocess, tempfile, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import plistlib

os.environ["CAIRO_ANTIALIAS"] = "none"

def find_aseprite():
    # 3) common macOS app bundle path
    default = "/Applications/Aseprite.app/Contents/MacOS/aseprite"
    if Path(default).exists():
        return default
    # 1) honor env var
    env = os.environ.get("ASEPRITE")
    if env and Path(env).exists():
        return env
    # 2) PATH lookup
    w = shutil.which("aseprite")
    if w:
        return w
    return None

# -------- helpers --------

def lerp(a, b, t): return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)

def dist2(a, b):
    dx, dy = a[0]-b[0], a[1]-b[1]
    return dx*dx + dy*dy

def flat_quad(p0, p1, p2, eps2, out):
    # recursive subdivision until deviation is within eps
    # deviation estimate: distance of control point from line p0-p2
    # using perpendicular distance squared
    # project p1 to line p0->p2
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
    # subdivide
    p01 = lerp(p0,p1,0.5)
    p12 = lerp(p1,p2,0.5)
    p012 = lerp(p01,p12,0.5)
    flat_quad(p0, p01, p012, eps2, out)
    flat_quad(p012, p12, p2, eps2, out)

def flat_cubic(p0, p1, p2, p3, eps2, out):
    # Use control polygon flatness (max distance of p1,p2 from line p0-p3)
    x0,y0=p0; x1,y1=p1; x2,y2=p2; x3,y3=p3
    vx, vy = x3-x0, y3-y0
    def dev2(p):
        if vx==0 and vy==0:
            return dist2(p0,p)
        t = ((p[0]-x0)*vx + (p[1]-y0)*vy)/(vx*vx+vy*vy)
        proj = (x0+vx*t, y0+vy*t)
        return dist2(p, proj)
    if max(dev2(p1), dev2(p2)) <= eps2:
        out.append(p3); return
    # subdivide with De Casteljau
    p01 = lerp(p0,p1,0.5); p12 = lerp(p1,p2,0.5); p23 = lerp(p2,p3,0.5)
    p012 = lerp(p01,p12,0.5); p123 = lerp(p12,p23,0.5)
    p0123 = lerp(p012,p123,0.5)
    flat_cubic(p0, p01, p012, p0123, eps2, out)
    flat_cubic(p0123, p123, p23, p3, eps2, out)

def read_contents_mapping(glyphs_dir: Path):
    """Return {glyphName: fileName.glif} from glyphs/contents.plist if present."""
    contents = glyphs_dir / "contents.plist"
    if not contents.exists():
        return {}
    with contents.open("rb") as f:
        return plistlib.load(f)  # dict: glyphName -> fileName (with .glif)

def read_font_metrics(ufo_root: Path):
    fi = ufo_root / "fontinfo.plist"
    asc, desc, upm, cap = 800, -200, 1000, None
    if fi.exists():
        with fi.open("rb") as f:
            d = plistlib.load(f)
        asc  = int(d.get("ascender", asc))
        desc = int(d.get("descender", desc))
        upm  = int(d.get("unitsPerEm", upm))
        cap  = d.get("capHeight", None)
        if cap is not None: cap = float(cap)
    return asc, desc, upm, cap


def resolve_glif_path(glyph_name: str, glyphs_dir: Path, contents_map: dict, allow_fallback=True):
    # 1) contents.plist mapping
    if glyph_name in contents_map:
        p = glyphs_dir / contents_map[glyph_name]
        return (p if p.exists() else None, "contents")

    # 2) naive name.glif
    p = glyphs_dir / f"{glyph_name}.glif"
    if p.exists():
        return (p, "name")

    if not allow_fallback:
        return (None, "missing")

    # 3) uppercase stored as Name_.glif
    p2 = glyphs_dir / f"{glyph_name}_.glif"
    if p2.exists():
        return (p2, "underscore")

    return (None, "missing")

def read_glyph_order(lib_plist: Path):
    with lib_plist.open('rb') as f:
        data = plistlib.load(f)
    order = data.get("public.glyphOrder", [])
    return [n for n in order if isinstance(n, str)]

def parse_glif(glif_path: Path):
    tree = ET.parse(glif_path)
    root = tree.getroot()
    outline = root.find("./outline")
    contours = []
    if outline is None:
        return contours

    for contour in outline.findall("./contour"):
        pts = []
        for pt in contour.findall("./point"):
            x = float(pt.attrib["x"])
            y = float(pt.attrib["y"])
            t = pt.attrib.get("type", "offcurve")  # line|curve|qcurve|offcurve
            pts.append((x, y, t))
        if pts:
            contours.append(pts)
    return contours

def bbox(contours):
    xs, ys = [], []
    for pts in contours:
        for x, y, _ in pts:
            xs.append(x); ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))

def contours_to_svg_path(contours, eps_px=0.5):
    """
    Convert GLIF-style contours into a single SVG path string composed only of
    M/L/Z commands by flattening all curves (cubic & quadratic) with tolerance eps_px.
    Y is flipped for SVG.
    """
    eps2 = eps_px * eps_px
    def flipY(y): return -y

    dparts = []
    for pts in contours:
        if not pts:
            continue

        # rotate so first is on-curve
        start_idx = 0
        for i, (_, _, t) in enumerate(pts):
            if t != "offcurve":
                start_idx = i; break
        ordered = pts[start_idx:] + pts[:start_idx]
        n = len(ordered)
        if n == 0: continue
        if ordered[0][2] == "offcurve":  # degenerate
            continue

        # Current on-curve
        curr = (ordered[0][0], ordered[0][1])
        poly = [curr]

        i = 0
        while i < n:
            # collect off-curves until the next on-curve (wrap)
            off = []
            j = (i + 1) % n
            while True:
                xj, yj, tj = ordered[j]
                if tj == "offcurve":
                    off.append((xj, yj))
                    j = (j + 1) % n
                    if j == i: break
                else:
                    next_on = (xj, yj); next_type = tj
                    break

            # Emit segment curr -> next_on
            if next_type == "line" or len(off) == 0:
                poly.append(next_on)
            elif next_type == "curve":
                if len(off) == 2:
                    out = []
                    flat_cubic(curr, off[0], off[1], next_on, eps2, out)
                    poly.extend(out)
                else:
                    # unexpected count; degrade to straight line
                    poly.append(next_on)
            elif next_type == "qcurve":
                # One or more quadratic off-curves before the on-curve:
                # break them into successive quads: curr->off0->(implied or oncurve),
                # handling chains of off-curves (UFO qcurve rules)
                quads = []
                qpts = off[:] + [next_on]
                prev = curr
                k = 0
                while k < len(qpts)-1:
                    c = qpts[k]
                    # Next on-curve is either explicit or implied midpoint
                    if k+1 < len(qpts)-1:
                        # implied on-curve between c and next control
                        nctrl = qpts[k+1]
                        implied = ((c[0]+nctrl[0])/2.0, (c[1]+nctrl[1])/2.0)
                        quads.append((prev, c, implied))
                        prev = implied
                        k += 1
                    else:
                        # last is the real on-curve
                        onp = qpts[-1]
                        quads.append((prev, c, onp))
                        prev = onp
                        k += 1
                for (p0, p1, p2) in quads:
                    out = []
                    flat_quad(p0, p1, p2, eps2, out)
                    poly.extend(out)
            else:
                # unknown -> line
                poly.append(next_on)

            curr = poly[-1]
            if j == i: break
            i = j
            if i == 0: break

        # Deduplicate consecutive identical points
        dedup = [poly[0]]
        for p in poly[1:]:
            if p != dedup[-1]:
                dedup.append(p)

        # Build SVG commands
        if len(dedup) >= 2:
            x0,y0 = dedup[0]
            seg = [f"M{x0} {flipY(y0)}"]
            for (x,y) in dedup[1:]:
                seg.append(f"L{x} {flipY(y)}")
            seg.append("Z")
            dparts.append(" ".join(seg))

    return " ".join(dparts)

def write_svg(svg_path: Path, contours, *,
                   asc: int, desc: int,
                   cap: float | None,
                   cap_height_px: float | None,
                   cell_w: int, cell_h: int,
                   pad_px: float, align: str, eps_px: float,
                   debug: bool = False):

    d = contours_to_svg_path(contours, eps_px=eps_px)  # flips Y already

    # Scale: prefer capHeight if provided, else em box
    if cap_height_px and cap:
        scale = cap_height_px / float(cap)
        if debug: print(f"[debug] scale by cap: {cap} → {scale:.4f}")
    else:
        em_h = asc - desc
        usable_h_px = max(1.0, cell_h - 2.0*pad_px)
        scale = usable_h_px / float(em_h)
        if debug: print(f"[debug] scale by em: {em_h} → {scale:.4f}")

    pad_fu = pad_px / scale
    usable_w_px = max(1.0, cell_w - 2.0*pad_px)
    vw_fu = usable_w_px / scale
    vh_fu = (cell_h - 2.0*pad_px) / scale

    # Vertical: put baseline at fixed position: pad + asc from the top
    vx0_fu = -pad_fu
    vy0_fu = -(asc + pad_fu)

    # Horizontal: align bbox left or center
    if contours:
        x0, _, x1, _ = bbox(contours)  # font units
    else:
        x0 = x1 = 0.0
    gx_mid = 0.5 * (x0 + x1)

    if align == "left":
        # left edge of glyph bbox sits at pixel 0 (+pad if any)
        vx0_fu = x0 - pad_fu
    else:
        usable_center_fu = (vx0_fu + pad_fu) + vw_fu * 0.5
        dx = gx_mid - usable_center_fu
        vx0_fu += dx

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{cell_w}" height="{cell_h}"
     viewBox="{vx0_fu} {vy0_fu} {vw_fu + 2*pad_fu} {vh_fu + 2*pad_fu}">
  <path d="{d}" fill="white" fill-rule="evenodd" />
</svg>
'''
    svg_path.write_text(svg, encoding="utf-8")

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

# -------- main pipeline --------

def main():
    ap = argparse.ArgumentParser(description="Convert lib.plist + .glif folder into one .aseprite sheet")
    ap.add_argument("--lib", required=True, help="Path to lib.plist")
    ap.add_argument("--glyphs-dir", required=True, help="Folder containing .glif files")
    ap.add_argument("--out", required=True, help="Output .aseprite path (e.g., out/font.aseprite)")
    ap.add_argument("--png-size", type=int, default=64, help="Target PNG width (px) for each glyph (default: 64)")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing glyphs instead of failing")
    ap.add_argument("--map-out", help="Write Aseprite --sheet JSON/stdout to this file (e.g. out/sheet_map.json)")
    ap.add_argument("--debug", action="store_true", help="Verbose logs: mapping, resolves, and segment stats")
    ap.add_argument("--flatten-eps", type=float, default=0.5, help="Max deviation in pixels when flattening curves (default: 0.5)")
    ap.add_argument("--oversample", type=int, default=1,
    help="Rasterize at N× width, then downscale with nearest (default 1)")
    ap.add_argument("--stroke-px", type=float, default=None,
    help="Grow the glyph by N pixels at the final size. "
         "If omitted, auto-scales with --png-size (png_size/128).")
    ap.add_argument("--hard-threshold", action="store_true",
    help="Force 1-bit alpha (no gray) after ops")
    ap.add_argument("--cell-width",  type=int, help="Fixed cell width in px (default: --png-size)")
    ap.add_argument("--cell-height", type=int, help="Fixed cell height in px (default: --png-size)")
    ap.add_argument("--align", choices=["left","center"], default="center",
                help="Horizontal alignment inside the cell (default: center)")
    ap.add_argument("--pad-px", type=float, default=0.0,
                help="Inner padding in px on all sides (default: 0)")
    ap.add_argument("--cap-height-px", type=float, default=None,
                help="If set, scale so capHeight maps to this many pixels (else scale by em box)")


    args = ap.parse_args()

    ufo_root = Path(args.lib).parent           # lib.plist lives at UFO root
    ASC, DESC, UPM, CAP = read_font_metrics(ufo_root)
    if args.debug:
        print(f"[debug] font metrics: asc={ASC} desc={DESC} upm={UPM} cap={CAP}")

    auto_stroke = round(args.png_size / 128.0, 3)  # 64->0.5, 32->0.25, 16->0.125
    stroke_px = auto_stroke if args.stroke_px is None else float(args.stroke_px)
    if args.debug:
        print(f"[debug] stroke_px(final)={stroke_px} (auto={auto_stroke}, user={args.stroke_px})")

    lib_path = Path(args.lib)
    glifs_dir = Path(args.glyphs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load once (moved out of the loop)
    contents_map = read_contents_mapping(glifs_dir)

    if args.debug:
        print(f"[debug] contents.plist entries: {len(contents_map)}")
        # Show only entries we’ll actually touch, to keep noise down:
        order_peek = [n for n in read_glyph_order(lib_path)]
        for n in order_peek:
            if n in contents_map:
                print(f"[debug] contents: {n} -> {contents_map[n]}")

    order = read_glyph_order(lib_path)
    if not order:
        print("ERROR: No public.glyphOrder found in lib.plist", file=sys.stderr)
        sys.exit(1)

    tmp = Path(tempfile.mkdtemp(prefix="glif2ase_"))
    svgs_dir = tmp / "svgs"
    pngs_dir = tmp / "pngs"
    svgs_dir.mkdir(parents=True, exist_ok=True)
    pngs_dir.mkdir(parents=True, exist_ok=True)

    included_pngs = []
    skipped = []

    try:
        seen_files = set()  # absolute paths to avoid duplicates

        for name in order:
            candidate, how = resolve_glif_path(name, glifs_dir, contents_map, allow_fallback=True)
            if args.debug:
                print(f"[debug] resolve: {name} via {how} -> {candidate}")

            if candidate is None:
                if args.skip_missing:
                    skipped.append((name, "missing"))
                    continue
                else:
                    print(f"ERROR: Missing GLIF for '{name}' (no contents.plist entry or file not found)", file=sys.stderr)
                    sys.exit(1)

            real = candidate.resolve()
            if real in seen_files:
                if args.debug:
                    print(f"[debug] de-dupe: {name} maps to already included file {real}")
                continue
            seen_files.add(real)

            cell_w = args.cell_width  or args.png_size
            cell_h = args.cell_height or args.png_size

            contours = parse_glif(candidate)
            if args.debug:
                # quick stats
                cc = len(contours)
                pts_count = sum(len(c) for c in contours)
                has_q = any(any(t == "qcurve" for _,_,t in c) for c in contours)
                has_c = any(any(t == "curve" for _,_,t in c) for c in contours)
                print(f"[debug] glyph {name}: contours={cc}, points={pts_count}, cubic={has_c}, quad={has_q}")
                mn = min((p[1] for c in contours for p in c), default=None)
                mx = max((p[1] for c in contours for p in c), default=None)
                print(f"[debug] {name}: yMin={mn} yMax={mx}")

            base = Path(candidate).stem
            svg_path = svgs_dir / f"{base}.svg"
            write_svg(svg_path, contours,
                asc=ASC, desc=DESC,
                cap=CAP, cap_height_px=args.cap_height_px,
                cell_w=cell_w, cell_h=cell_h,
                pad_px=args.pad_px, align=args.align,
                eps_px=args.flatten_eps,
                debug=args.debug)

            png_path = pngs_dir / f"{base}.png"
            os.environ["CAIRO_ANTIALIAS"] = "none"
            rc, out, err = run([
                "rsvg-convert",
                "-w", str(args.png_size),
                # "--without-background",
                # "--property", "shape-rendering=crispEdges",
                # "--property", "text-rendering=geometricPrecision",
                # "--property", "image-rendering=optimizeSpeed",
                # "--disable-antialias",
                "-b", "transparent",
                "-a",
                "-o", str(png_path),
                str(svg_path),
            ])

            # Oversample by rendering larger, then shrink with nearest neighbor
            overs = args.oversample
            if overs > 1:
                big = str(png_path).replace(".png", f"@{overs}x.png")
                # render big
                rc, _, err = run([
                    "rsvg-convert",
                    "-w", str(args.png_size * overs),
                    "-a",
                    "-o", big,
                    str(svg_path),
                ])
                if rc != 0:
                    print(f"ERROR: rsvg-convert (oversample) failed for {name}: {err}", file=sys.stderr)
                    sys.exit(1)
                # shrink with nearest neighbor (pixel-perfect)
                rc, _, err = run([
                    "magick", "PNG32:" + str(big),
                    "-filter", "point", "-resize", f"{100/overs}%",
                    "PNG32:" + str(png_path)
                ])
                if rc != 0:
                    print(f"ERROR: magick resize failed for {name}: {err}", file=sys.stderr)
                    sys.exit(1)

            if rc != 0:
                print(f"ERROR: rsvg-convert failed for {name}: {err}", file=sys.stderr)
                sys.exit(1)

            run(["magick", str(png_path), "-posterize", "2", "-threshold", "50%", str(png_path)])

            k = max(1, int(round(args.stroke_px))) if args.stroke_px else 0
            if k > 0:
                rc, _, err = run([
                    "magick", "PNG32:" + str(png_path),
                    "(", "+clone", "-alpha", "extract",
                    "-morphology", "Dilate", f"Octagon:{k}", ")",
                    "-compose", "copy-opacity", "-composite",
                    "PNG32:" + str(png_path)
                ])
                if rc != 0:
                    print(f"ERROR: stroke step failed for {name}: {err}", file=sys.stderr)
                    sys.exit(1)

            if args.hard_threshold:
                rc, _, err = run([
                    "magick", "PNG32:" + str(png_path),
                    "(", "+clone", "-alpha", "extract", "-threshold", "50%", ")",
                    "-compose", "copy-opacity", "-composite",
                    "PNG32:" + str(png_path)
                ])
                if rc != 0:
                    print(f"ERROR: magick threshold failed for {name}: {err}", file=sys.stderr)
                    sys.exit(1)

            included_pngs.append(str(png_path))

        if not included_pngs:
            print("ERROR: No glyphs were included (all missing/curves?).", file=sys.stderr)
            sys.exit(1)

        aseprite_bin = find_aseprite()
        if not aseprite_bin:
            print('ERROR: aseprite not found. Either:\n'
                ' - brew/itch/steam install and ensure it is on PATH, or\n'
                ' - set ASEPRITE=/Applications/Aseprite.app/Contents/MacOS/aseprite, or\n'
                ' - create a symlink in /opt/homebrew/bin/aseprite',
                file=sys.stderr)
            sys.exit(1)

        tmp_sheet_png = tmp / "sheet.png"
        cols = len(included_pngs)
        sheet_cmd = [
            aseprite_bin, "-b", *included_pngs,
            "--sheet", str(tmp_sheet_png),
            f"--sheet-columns={cols}",
            "--sheet-rows=1",
            "--sheet-type=horizontal",
            "--sheet-border-padding", "0",
            "--sheet-spacing", "0",
            "--sheet-inner-padding", "0",
            "--sheet-transparent-color", "0", "0", "0", "0"  # RGBA
            # (do NOT pass --trim for fixed grid)
            ]


        rc, sheet_stdout, err = run(sheet_cmd)
        if rc != 0:
            print(f"ERROR: aseprite sheet failed: {err}", file=sys.stderr)
            sys.exit(1)
        print(sheet_stdout)

        if out_path.suffix.lower() == ".aseprite":
            rc, out, err = run([aseprite_bin, "-b", str(tmp_sheet_png),
                                "--save-as", str(out_path)])
            if rc != 0:
                print(f"ERROR: saving .aseprite failed: {err}", file=sys.stderr)
                sys.exit(1)
        else:
            # Otherwise write a PNG directly to the requested location
            shutil.copyfile(tmp_sheet_png, out_path)

        if args.map_out:
            Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.map_out).write_text(sheet_stdout + "\n", encoding="utf-8")


        print(f"✅ Wrote {out_path}")
        print("aseprite cmd:", " ".join(sheet_cmd))
        if skipped:
            print("ℹ️ Skipped glyphs:", ", ".join([f"{n}({why})" for n, why in skipped]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()

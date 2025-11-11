import sys, xml.etree.ElementTree as ET
from pathlib import Path

def parse_glif(glif_path: Path):
    tree = ET.parse(glif_path)
    root = tree.getroot()
    name = root.attrib.get("name", "glyph")
    advance = root.find("./advance")
    width = int(advance.attrib.get("width", "0")) if advance is not None else 0

    contours = []
    for contour in root.findall(".//contour"):
        pts = []
        for pt in contour.findall("./point"):
            x = float(pt.attrib["x"])
            y = float(pt.attrib["y"])
            t = pt.attrib.get("type", "offcurve")
            pts.append((x, y, t))
        if pts:
            contours.append(pts)

    return name, width, contours

def contours_bbox(contours):
    xs, ys = [], []
    for pts in contours:
        for x, y, _ in pts:
            xs.append(x); ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))

def contours_to_svg_path(contours):
    # lines-only: move to first point, then L to each next, and Z close
    segs = []
    for pts in contours:
        if not pts: 
            continue
        x0, y0, _ = pts[0]
        seg = [f"M{x0} {-(y0)}"]  # flip Y for SVG (GLIF up, SVG down)
        for x, y, _ in pts[1:]:
            seg.append(f"L{x} {-(y)}")
        seg.append("Z")
        segs.append(" ".join(seg))
    return " ".join(segs)

def write_svg(svg_path: Path, glyph_name, width, contours):
    x0, y0, x1, y1 = contours_bbox(contours)
    # Add small padding
    pad = 50
    vx0, vy0 = x0 - pad, -(y1 + pad)
    vw, vh = (x1 - x0) + 2*pad, (y1 - y0) + 2*pad

    d = contours_to_svg_path(contours)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{vw}" height="{vh}"
     viewBox="{vx0} {vy0} {vw} {vh}">
  <title>{glyph_name}</title>
  <path d="{d}" fill="white" />
</svg>
'''
    svg_path.write_text(svg, encoding="utf-8")

def main():
    if len(sys.argv) < 3:
        print("Usage: python glif_to_svg.py input.glif output.svg")
        sys.exit(1)
    glif_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    glyph_name, width, contours = parse_glif(glif_path)
    if not contours:
        print("No contours found.")
        sys.exit(1)

    write_svg(out_path, glyph_name, width, contours)
    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    main()

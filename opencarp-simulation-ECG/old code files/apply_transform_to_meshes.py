#!/usr/bin/env python3
"""
Apply a 4x4 transformation matrix to mesh files (.vtu, .vtk, .vtp).

Usage examples:
  # save the transform from your Open3D script:
  # np.save('transform.npy', transformation)

  python apply_transform_to_meshes.py \
    -t transform.npy \
    -i /path/to/input_meshes \
    -o /path/to/output_meshes

The script reads a 3x3 or 4x4 matrix (npy or whitespace-delimited text),
applies it to point coordinates and writes meshes preserving topology and data.
"""
import argparse
from pathlib import Path
import sys

try:
    import numpy as np
    import pyvista as pv
except Exception:
    print("Required modules missing. Install with: pip install pyvista vtk numpy")
    sys.exit(1)


def load_transform(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    try:
        T = np.load(str(path))
    except Exception:
        T = np.loadtxt(str(path))
    T = np.asarray(T, dtype=float)
    if T.shape == (3, 3):
        T4 = np.eye(4, dtype=float)
        T4[:3, :3] = T
        return T4
    if T.shape == (4, 4):
        return T
    raise ValueError("Transform must be a 3x3 or 4x4 matrix")


def apply_transform(mesh: pv.DataSet, T: np.ndarray) -> None:
    pts = mesh.points
    if pts is None or pts.size == 0:
        raise RuntimeError("Mesh has no points to transform")
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts.astype(float), ones])
    pts_t = (T @ pts_h.T).T
    mesh.points = pts_t[:, :3]


def process_files(input_dir: Path, output_dir: Path, T: np.ndarray,
                  exts=(".vtu", ".vtk", ".vtp"), recursive=True,
                  suffix="_aligned", overwrite=False):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    glob = input_dir.rglob if recursive else input_dir.glob
    files = []
    for ext in exts:
        files.extend(list(glob(f"*{ext}")))
    if not files:
        print("No mesh files found in the input directory.")
        return
    for f in files:
        rel = f.relative_to(input_dir)
        out_parent = output_dir / rel.parent
        out_parent.mkdir(parents=True, exist_ok=True)
        out_name = (f.stem + suffix + f.suffix) if not overwrite else f.name
        out_path = out_parent / out_name
        print(f"Processing {f} -> {out_path}")
        mesh = pv.read(str(f))
        if getattr(mesh, 'n_points', 0) == 0:
            print(f"Skipping {f}: no points")
            continue
        apply_transform(mesh, T)
        mesh.save(str(out_path))
    print("All done.")


def main():
    p = argparse.ArgumentParser(description="Apply 4x4 transform to mesh files (.vtu/.vtk/.vtp)")
    p.add_argument('-t', '--transform', required=True, help='Path to .npy or text matrix')
    p.add_argument('-i', '--input', required=True, help='Input directory (searched recursively)')
    p.add_argument('-o', '--output', required=True, help='Output directory (created if needed)')
    p.add_argument('--suffix', default='_aligned', help='Suffix appended to filenames (default _aligned)')
    p.add_argument('--overwrite', action='store_true', help='Keep original filename in output dir')
    p.add_argument('--no-recursive', dest='recursive', action='store_false', help='Do not search input directory recursively')
    p.add_argument('--invert', action='store_true', help='Apply inverse of provided transform')
    args = p.parse_args()

    transform_path = Path(args.transform)
    T = load_transform(transform_path)
    if args.invert:
        T = np.linalg.inv(T)

    process_files(Path(args.input), Path(args.output), T,
                  recursive=args.recursive, suffix=args.suffix, overwrite=args.overwrite)


if __name__ == '__main__':
    main()

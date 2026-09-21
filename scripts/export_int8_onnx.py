#!/usr/bin/env python3
"""Export an Ultralytics YOLOv8 .pt checkpoint to INT8 ONNX (QDQ format).

This is a two-stage pipeline:

  1. Ultralytics exports the .pt checkpoint to F32 ONNX.
  2. `onnxruntime.quantization.quantize_static` performs PTQ with
     `QuantFormat.QDQ`, producing the standard ONNX QDQ-wrapped graph
     consumed by `dragonwing_onnx::fold_qdq_patterns`.

Ultralytics 8.x dropped first-party `int8=True` support for the `onnx`
target — INT8 there now lives in `tflite`, `openvino`, `engine`, etc.
The ONNX Runtime quantization toolchain is the de-facto path for
producing ONNX QDQ models and is the format we target.

Usage
-----

    scripts/export_int8_onnx.py \
        --pt artifacts/models/excavator_stone/truck.yolov8m.p640.20250512_best.pt \
        --calib path/to/calibration/images \
        --out artifacts/models/truck.yolov8m.p640.int8.onnx \
        --imgsz 640

    # Skip calibration — uses a small random-data set (DEBUG ONLY;
    # produces a structurally-valid QDQ graph with meaningless scales).
    scripts/export_int8_onnx.py --pt model.pt --out out.onnx --imgsz 640 \
        --calib-frac 0.05 --random-calib 32

Calibration image requirements
------------------------------

* ~100 representative images at the inference resolution.
* Diverse lighting, angles, and content distribution.
* Same domain as the deployment scenes.

Dependencies
------------

    pip install ultralytics onnx onnxruntime onnxsim pillow numpy
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


# -----------------------------------------------------------------------------
# Stage 1: Ultralytics → F32 ONNX
# -----------------------------------------------------------------------------


def export_f32_onnx(pt_path: Path, imgsz: int, simplify: bool, verbose: bool) -> Path:
    """Export Ultralytics .pt to an F32 ONNX file. Returns the produced path."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. Install with:\n"
            "    pip install ultralytics onnx onnxruntime pillow numpy\n"
            f"(original error: {e})"
        )

    if not pt_path.exists():
        raise SystemExit(f"input checkpoint not found: {pt_path}")

    if verbose:
        print(f"[export_int8] stage 1: ultralytics → F32 ONNX", flush=True)
        print(f"[export_int8]   loading {pt_path}", flush=True)
    model = YOLO(str(pt_path))

    f32_onnx_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            opset=13,  # exposes axis attribute on QuantizeLinear / DequantizeLinear
        )
    ).resolve()
    if verbose:
        print(f"[export_int8]   produced {f32_onnx_path}", flush=True)
    return f32_onnx_path


# -----------------------------------------------------------------------------
# Stage 2: F32 ONNX → INT8 QDQ
# -----------------------------------------------------------------------------


class _ImageDirCalibReader:
    """`CalibrationDataReader` that walks a directory of images.

    Loads RGB JPG/PNG/BMP files, resizes to `imgsz`, normalises to [0, 1],
    and yields `{input_name: NCHW float32 tensor}` dicts in batches.

    Falls back to random data when `random_calib > 0` (debug path).
    """

    def __init__(
        self,
        input_name: str,
        imgsz: int,
        calib_dir: Path | None,
        random_calib: int,
        calib_frac: float,
        verbose: bool,
    ):
        import numpy as np

        self._np = np
        self._input_name = input_name
        self._imgsz = imgsz
        self._verbose = verbose

        if random_calib > 0:
            # Random uniform data — useful for smoke-testing the pipeline.
            if verbose:
                print(
                    f"[export_int8]   using {random_calib} random calibration samples (DEBUG)",
                    flush=True,
                )
            self._batches = [
                {
                    input_name: np.random.uniform(0.0, 1.0, (1, 3, imgsz, imgsz)).astype(
                        np.float32
                    )
                }
                for _ in range(random_calib)
            ]
        elif calib_dir is not None:
            try:
                from PIL import Image
            except ImportError as e:
                raise SystemExit(f"pillow is required for image calibration: {e}")

            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = sorted(
                p for p in calib_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
            )
            if not files:
                raise SystemExit(f"no calibration images found under {calib_dir}")

            n = max(1, int(len(files) * calib_frac))
            files = files[:n]
            if verbose:
                print(
                    f"[export_int8]   loading {len(files)} calibration images from {calib_dir}",
                    flush=True,
                )

            self._batches = []
            for f in files:
                try:
                    img = Image.open(f).convert("RGB").resize((imgsz, imgsz))
                except Exception as e:
                    if verbose:
                        print(f"[export_int8]   skipping {f}: {e}", file=sys.stderr)
                    continue
                arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
                arr = arr.transpose(2, 0, 1)[None]  # NCHW
                self._batches.append({input_name: arr})
        else:
            raise SystemExit(
                "no calibration source: pass either --calib DIR or --random-calib N"
            )

        self._iter = iter(self._batches)

    def get_next(self):  # noqa: D401
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._batches)


def quantize_qdq(
    f32_onnx_path: Path,
    output_path: Path,
    imgsz: int,
    calib_dir: Path | None,
    random_calib: int,
    calib_frac: float,
    verbose: bool,
) -> Path:
    """Quantize the F32 ONNX to INT8 QDQ via ONNX Runtime PTQ."""
    try:
        import onnx
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise SystemExit(
            "onnxruntime + onnx are required for QDQ quantization. Install:\n"
            "    pip install onnx onnxruntime\n"
            f"(original error: {e})"
        )

    # Discover the model's input name (usually "images" for YOLOv8).
    model = onnx.load(str(f32_onnx_path))
    input_name = model.graph.input[0].name
    if verbose:
        print(f"[export_int8] stage 2: ORT static QDQ quantization", flush=True)
        print(f"[export_int8]   input tensor name: {input_name}", flush=True)

    reader = _ImageDirCalibReader(
        input_name=input_name,
        imgsz=imgsz,
        calib_dir=calib_dir,
        random_calib=random_calib,
        calib_frac=calib_frac,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[export_int8]   quantizing → {output_path}", flush=True)

    quantize_static(
        model_input=str(f32_onnx_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # INT8 symmetric for both activations and weights — matches the
        # dragonwing QDQ-fold pass which rejects non-zero zero-points.
        # ORT defaults to asymmetric activation quantization (zero_point=128)
        # even with QInt8; the extra_options below force symmetric.
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibTensorRangeSymmetric": True,
        },
    )

    if verbose:
        print(f"[export_int8]   done (pre-simplify)", flush=True)
    return output_path


def simplify_quantized(onnx_path: Path, verbose: bool) -> None:
    """Run `onnxsim.simplify` on a QDQ-quantized ONNX model in place.

    ORT's static quantizer inserts `Shape → Gather → Mul → Slice` chains
    in YOLO detection heads with constant arithmetic that the dragonwing
    `SliceBuilder` cannot resolve (it requires `starts/ends` to be true
    constants). `onnxsim` folds those constant chains and lets the
    detection head compile cleanly.

    Skipped quietly if `onnxsim` is not installed.
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        if verbose:
            print(
                "[export_int8] onnxsim not installed; skipping post-quant simplify",
                flush=True,
            )
        return

    if verbose:
        print(f"[export_int8] stage 3: onnxsim post-quant simplify", flush=True)
    model = onnx.load(str(onnx_path))
    n_before = len(model.graph.node)
    try:
        simplified, ok = simplify(model)
    except Exception as e:
        print(
            f"[export_int8] WARNING: onnxsim failed ({e}); keeping unsimplified model",
            file=sys.stderr,
        )
        return
    if not ok:
        print(
            "[export_int8] WARNING: onnxsim returned ok=False; keeping unsimplified model",
            file=sys.stderr,
        )
        return
    onnx.save(simplified, str(onnx_path))
    n_after = len(simplified.graph.node)
    if verbose:
        print(
            f"[export_int8]   simplified {n_before} → {n_after} nodes",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def summarise(onnx_path: Path) -> None:
    """Print a small QDQ summary so the user can sanity-check the export."""
    try:
        import onnx
    except ImportError:
        print(
            "[export_int8] warning: `onnx` not installed; skipping summary",
            file=sys.stderr,
        )
        return

    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    print(f"\n[export_int8] {onnx_path.name} summary:")
    print(f"  ir_version : {model.ir_version}")
    print(f"  opset      : {[o.version for o in model.opset_import]}")
    print(f"  inputs     : {[i.name for i in model.graph.input]}")
    print(f"  outputs    : {[o.name for o in model.graph.output]}")
    print(f"  nodes      : {len(model.graph.node)}")

    qlin = counts.get("QuantizeLinear", 0)
    dqlin = counts.get("DequantizeLinear", 0)
    conv = counts.get("Conv", 0)
    print(f"  QuantizeLinear   : {qlin}")
    print(f"  DequantizeLinear : {dqlin}")
    print(f"  Conv             : {conv}")

    file_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  file size  : {file_mb:.1f} MB")

    if qlin == 0 and dqlin == 0:
        print(
            "\n  WARNING: no QDQ nodes detected. Quantization probably failed."
        )
    else:
        ratio = (qlin + dqlin) / max(conv, 1)
        print(f"  QDQ-per-Conv ratio: {ratio:.2f}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pt", required=True, type=Path, help="Input .pt checkpoint")
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="Directory of calibration images (JPG/PNG/BMP). See top-of-file docs.",
    )
    parser.add_argument(
        "--random-calib",
        type=int,
        default=0,
        help=(
            "Number of random-data calibration samples. Use this for "
            "smoke-testing the QDQ pipeline (output is structurally valid "
            "but scales are meaningless). Mutually exclusive with --calib."
        ),
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output .onnx file path"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (must match training resolution). Default: 640.",
    )
    parser.add_argument(
        "--calib-frac",
        type=float,
        default=1.0,
        help="Fraction of the calibration set to use (0..1). Default: 1.0.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim post-pass on the F32 export.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the QDQ summary at the end.",
    )
    parser.add_argument(
        "--keep-f32",
        action="store_true",
        help="Keep the intermediate F32 ONNX file alongside the INT8 output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.calib is None and args.random_calib == 0:
        parser.error("either --calib DIR or --random-calib N must be provided")
    if args.calib is not None and args.random_calib > 0:
        parser.error("--calib and --random-calib are mutually exclusive")

    pt_path = args.pt.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ultralytics → F32 ONNX (in pt's directory by default).
    f32_path = export_f32_onnx(
        pt_path=pt_path,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
        verbose=args.verbose,
    )

    # Stage 2: F32 ONNX → INT8 QDQ ONNX.
    quantize_qdq(
        f32_onnx_path=f32_path,
        output_path=out_path,
        imgsz=args.imgsz,
        calib_dir=args.calib.resolve() if args.calib else None,
        random_calib=args.random_calib,
        calib_frac=args.calib_frac,
        verbose=args.verbose,
    )

    # Stage 3: post-quant simplify (folds the constant chains ORT
    # inserts in YOLO detection heads). Skipped if onnxsim is missing.
    if not args.no_simplify:
        simplify_quantized(out_path, verbose=args.verbose)

    # Optional cleanup of the intermediate F32 ONNX.
    if not args.keep_f32 and f32_path != out_path:
        try:
            f32_path.unlink()
            if args.verbose:
                print(f"[export_int8]   removed intermediate {f32_path}", flush=True)
        except OSError as e:
            print(f"[export_int8] warning: could not remove {f32_path}: {e}", file=sys.stderr)

    if not args.no_summary:
        summarise(out_path)


if __name__ == "__main__":
    main()

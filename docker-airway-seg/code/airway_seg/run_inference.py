# -*- coding: utf-8 -*-
"""
Batch inference script for Connectivity-Aware-Airway-Segmentaion.

Usage examples:
  python run_inference.py --input-dir ./sample --output-dir ./sample_out

The script will instantiate the model once and run prediction on every
.nii or .nii.gz file found in `--input-dir` (non-recursive by default).
"""
from pathlib import Path
import argparse
import sys
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "-i", required=True, help="Directory with input .nii or .nii.gz files")
    p.add_argument("--output-dir", "-o", required=True, help="Directory to write outputs")
    p.add_argument("--recursive", "-r", action="store_true", help="Recursively search subfolders")
    p.add_argument("--device", "-d", default=None, help="Override device (e.g. cpu or cuda:0)")
    return p.parse_args()


def find_images(folder: Path, recursive: bool):
    if recursive:
        patterns = ["**/*.nii", "**/*.nii.gz"]
    else:
        patterns = ["*.nii", "*.nii.gz"]
    files = []
    for pat in patterns:
        files.extend(list(folder.glob(pat)))
    # sort for deterministic order
    return sorted(files)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If user passed a device override, adjust config before importing model
    if args.device is not None:
        try:
            from configs import airway_config
            airway_config.config['device'] = args.device
        except Exception:
            pass

    # Import model and helpers
    from models.airway_model import AirwayExtractionModel
    from util.utils import load_itk_image, save_itk

    print("Instantiating model...")
    model = AirwayExtractionModel()

    img_files = find_images(input_dir, args.recursive)
    if not img_files:
        print(f"No .nii/.nii.gz files found in {input_dir}")
        return

    print(f"Found {len(img_files)} files. Starting inference...")
    failures = []
    for idx, fp in enumerate(img_files, 1):
        try:
            print(f"[{idx}/{len(img_files)}] {fp}")
            image, origin, spacing, direction = load_itk_image(str(fp))
            pred = model.predict(image)

            out_name = fp.stem
            # If filename ends with .nii.gz, Path.stem returns name without .gz; handle .nii.gz and .nii
            if fp.name.endswith('.nii.gz'):
                out_file = output_dir / (out_name + '_airway.nii.gz')
            else:
                out_file = output_dir / (out_name + '_airway.nii')

            save_itk(pred, str(out_file), origin, spacing, direction)
        except Exception as e:
            failures.append((fp, str(e)))
            traceback.print_exc()

    print("Done.")
    if failures:
        print(f"{len(failures)} files failed:")
        for f, err in failures:
            print(f" - {f}: {err}")


if __name__ == '__main__':
    main()

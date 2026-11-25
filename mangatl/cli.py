"""Run the pipeline on one input with a settings file (written by translate.py).

    python -m mangatl.cli --settings run/settings.json --input chapter/ --output run/ --batch
"""

import argparse
import os

# Suppress OpenBLAS NUM_THREADS warnings on machines with >24 logical cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import sys
import tempfile
import time
import zipfile
from pathlib import Path

from mangatl.inputs.archive import ARCHIVE_SUFFIXES, safe_extract_zip


def main():
    parser = argparse.ArgumentParser(description="Translate manga pages with a settings file.")
    parser.add_argument("--settings", required=True, help="settings.json (see mangatl/settings.py)")
    parser.add_argument("--input", required=True,
                        help="Image; with --batch: directory, ZIP/CBZ archive, or failed_paths.txt")
    parser.add_argument("--output", default=None, help="Output file; with --batch: output directory")
    parser.add_argument("--batch", action="store_true", help="Translate a folder / archive")
    args = parser.parse_args()

    from mangatl.common.log import init_file_logging

    init_file_logging()

    from mangatl.ml.model_manager import get_model_manager, migrate_model_dirs
    from mangatl.pipeline.page import batch_translate_images, translate_and_render
    from mangatl.settings import Settings
    from mangatl.common import eval_hooks
    from mangatl.common.log import log_message

    config = Settings.load(args.settings)
    eval_hooks.configure(config.debug)

    models_dir = config.runtime.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    migrate_model_dirs(models_dir)
    manager = get_model_manager()
    manager.set_models_dir(models_dir)
    if config.runtime.device == "cpu":
        manager.set_device(config.device)
    if config.runtime.hf_cache:
        manager.use_hf_cache = True
    log_message(f"Using {config.device.type.upper()} device.", always_print=True)

    if args.batch:
        input_path = Path(args.input)
        suffix = input_path.suffix.lower()
        glossary_name = (
            None
            if suffix == ".txt"
            else input_path.stem
            if suffix in ARCHIVE_SUFFIXES
            else input_path.resolve().name
        )
        zip_temp_dir_obj = None
        path_list_temp_dir_obj = None
        path_list_run_dir = None
        glossary_path = None
        preserve_structure = False
        source_path_map = None

        if input_path.is_file() and suffix in ARCHIVE_SUFFIXES:
            log_message(f"Detected ZIP archive: {input_path.name}", always_print=True)
            try:
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir_path = Path(temp_dir_obj.name)
                zip_temp_dir_obj = temp_dir_obj

                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    safe_extract_zip(zip_ref, temp_dir_path)

                log_message(
                    "Extracted ZIP archive to temporary directory",
                    always_print=True,
                )

                input_path = temp_dir_path
                preserve_structure = True
            except zipfile.BadZipFile:
                log_message(
                    f"Error: '{args.input}' is not a valid ZIP archive.",
                    always_print=True,
                )
                sys.exit(1)
            except Exception as e:
                log_message(
                    f"Error extracting ZIP archive: {e!s}",
                    always_print=True,
                )
                sys.exit(1)
        elif input_path.is_file() and input_path.suffix.lower() == ".txt":
            from mangatl.inputs.path_list import stage_path_list

            log_message(
                f"Detected failed-paths list: {input_path.name}",
                always_print=True,
            )
            path_list_temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(path_list_temp_dir_obj.name)
            source_path_map, path_list_run_dir, glossary_path = stage_path_list(
                input_path, temp_dir_path
            )
            if not source_path_map:
                log_message(
                    f"Error: no existing image paths found in '{args.input}'.",
                    always_print=True,
                )
                sys.exit(1)

            log_message(
                f"Prepared {len(source_path_map)} image(s) from path list",
                always_print=True,
            )
            input_path = temp_dir_path
            preserve_structure = path_list_run_dir is not None
        elif not input_path.is_dir():
            log_message(
                f"Error: --batch requires --input '{args.input}' to be a directory, "
                "ZIP/CBZ archive, or failed-paths .txt file.",
                always_print=True,
            )
            sys.exit(1)

        # A failed-paths list writes back into its original run unless --output is set
        output_dir = Path(args.output) if args.output else path_list_run_dir

        if args.output:
            if not output_dir.exists():
                log_message(
                    f"Creating output directory: {output_dir}", always_print=True
                )
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                log_message(
                    f"Error: Specified --output '{output_dir}' is not a directory.",
                    always_print=True,
                )
                sys.exit(1)

        try:
            results = batch_translate_images(
                input_path,
                config,
                output_dir,
                preserve_structure=preserve_structure,
                source_path_map=source_path_map,
                glossary_name=glossary_name,
                glossary_path=glossary_path,
            )
            failed_paths_file = results.get("failed_paths_file")
            if failed_paths_file:
                log_message(
                    f"Failed image paths saved to: {failed_paths_file}",
                    always_print=True,
                )
            if results.get("error_count", 0) > 0:
                raise SystemExit(1)
        finally:
            for temp_obj, label in (
                (zip_temp_dir_obj, "ZIP extraction"),
                (path_list_temp_dir_obj, "path list staging"),
            ):
                if not temp_obj:
                    continue
                try:
                    temp_obj.cleanup()
                    log_message(
                        f"Cleaned up {label} temporary directory",
                        always_print=True,
                    )
                except Exception as e_clean:
                    log_message(
                        f"Warning: Failed to clean up temporary directory: {e_clean}",
                        always_print=True,
                    )
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            log_message(
                f"Error: Input '{args.input}' is not a valid file.", always_print=True
            )
            sys.exit(1)

        output_path_arg = args.output
        original_ext = input_path.suffix.lower()
        output_format = config.output.format
        if output_format == "auto":
            output_ext = (
                original_ext
                if original_ext in [".jpg", ".jpeg", ".png", ".webp"]
                else ".png"
            )
        elif output_format == "png":
            output_ext = ".png"
        elif output_format == "jpeg":
            output_ext = ".jpg"
        else:
            output_ext = ".png"

        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{input_path.stem}_translated_{timestamp}{output_ext}"
            )
            log_message(
                f"--output not specified, using default: {output_path}",
                always_print=True,
            )
        else:
            output_path = Path(output_path_arg)
            if output_format != "auto" or not output_path.suffix:
                output_path = output_path.with_suffix(output_ext)
            if not output_path.parent.exists():
                log_message(
                    f"Creating directory for output file: {output_path.parent}",
                    always_print=True,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(
                f"Translation complete. Result saved to {output_path}",
                always_print=True,
            )
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)
            raise SystemExit(1) from e



if __name__ == "__main__":
    main()

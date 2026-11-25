"""Translating a folder of pages: order, previous-page context, glossary, parallel requests, retries."""

import asyncio
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from mangatl.common.coordinator import BatchRequestCoordinator
from mangatl.translate.glossary import default_glossary_path, load_terms
from mangatl.inputs.archive import copy_comic_info
from mangatl.common.exceptions import CancellationError, TranslationError, ValidationError
from mangatl.common.log import log_message
from mangatl.inputs.path_list import record_failed_images, resolve_source_path
from mangatl.settings import Settings
from mangatl.pipeline.page import translate_and_render
from mangatl.common.cancellation import CancellationManager


NATURAL_SORT_TOKEN_RE = re.compile(r"(\d+)")


def _natural_text_sort_key(text: str) -> tuple[tuple[int, int | str, str], ...]:
    return tuple(
        (0, int(part), part) if part.isdigit() else (1, part.lower(), part)
        for part in NATURAL_SORT_TOKEN_RE.split(text)
        if part
    )


def natural_path_sort_key(path: Path):
    return tuple(_natural_text_sort_key(part) for part in path.parts)


def _build_previous_context_texts(
    image_files: list[Path],
    image_index: int,
    config: Settings,
    ocr_text_history: dict[Path, list[str]] | None = None,
    ocr_text_history_lock: threading.Lock | None = None,
) -> list[list[str]]:
    """Collect OCR transcripts from up to N already-processed prior pages.

    Parallel callers that require deterministic prior-page text context should
    wait for the required previous pages before calling this helper.
    """
    requested_count = int(
        config.translation.context_pages
    )
    if requested_count <= 0 or ocr_text_history is None:
        return []

    start_index = max(0, image_index - requested_count)
    previous_paths = image_files[start_index:image_index]
    if not previous_paths:
        return []

    previous_texts: list[list[str]] = []
    if ocr_text_history_lock is not None:
        with ocr_text_history_lock:
            for previous_path in previous_paths:
                texts = ocr_text_history.get(previous_path)
                if texts:
                    previous_texts.append(list(texts))
    else:
        for previous_path in previous_paths:
            texts = ocr_text_history.get(previous_path)
            if texts:
                previous_texts.append(list(texts))
    return previous_texts


def _resolve_output_path(
    img_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: Settings,
    preserve_structure: bool,
) -> tuple[Path, str, str]:
    """Compute output path, display name, and error key for a single image."""
    if preserve_structure:
        relative_path = img_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f"{relative_path.stem}_translated"
        display_path = str(relative_path)
        error_key = str(relative_path)
    else:
        output_subdir = output_dir
        output_filename = f"{img_path.stem}_translated"
        display_path = img_path.name
        error_key = img_path.name

    original_ext = img_path.suffix.lower()
    desired_format = config.output.format
    if desired_format == "jpeg":
        output_ext = ".jpg"
    elif desired_format == "png":
        output_ext = ".png"
    elif desired_format == "auto":
        output_ext = original_ext
    else:
        output_ext = original_ext
        log_message(
            f"Warning: Invalid output_format '{desired_format}' in config. "
            f"Using original extension '{original_ext}'.",
            always_print=True,
        )

    return output_subdir / f"{output_filename}{output_ext}", display_path, error_key


def _check_glossary_path(path: Path) -> None:
    """Fail the batch once up front instead of on every page."""
    if path.is_dir():
        raise ValidationError(f"Glossary file '{path}' is a folder, not a file.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        load_terms(path)
    except (OSError, UnicodeDecodeError) as e:
        raise ValidationError(f"Cannot use glossary file '{path}': {e}") from e


def _translate_batch_page(*args, **kwargs) -> None:
    """Translate one batch page, failing it if any bubble was left empty.

    The partial result is still saved; a later retry of the page overwrites it.
    """
    empty_bubbles: list = []
    translate_and_render(*args, empty_bubbles_out=empty_bubbles, **kwargs)
    if empty_bubbles:
        raise TranslationError(f"{len(empty_bubbles)} bubble(s) left empty")


def _should_run_failed_retry(
    config: Settings,
    failed_jobs: list[dict[str, Any]],
    cancellation_manager: Optional["CancellationManager"] = None,
) -> bool:
    if not config.runtime.retry_failed_once:
        return False
    if not failed_jobs:
        return False
    return not (
        cancellation_manager is not None and cancellation_manager.is_cancelled()
    )


def _retry_failed_batch_images(
    failed_jobs: list[dict[str, Any]],
    results: dict[str, Any],
    config: Settings,
    input_dir: Path,
    output_dir: Path,
    preserve_structure: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    source_path_map: dict[str, str] | None = None,
) -> None:
    """Retry failed batch images once. Mutates *results* in place.

    Each entry in *failed_jobs* must include:
      - error_key: key used in results["errors"]
      - img_path: processing Path passed to translate_and_render
      - source_path: path stored in results["failed_image_paths"]
    """
    if not failed_jobs:
        results.setdefault("retry_attempted_count", 0)
        results.setdefault("retry_success_count", 0)
        results.setdefault("retry_failed_count", 0)
        return

    if cancellation_manager is not None and cancellation_manager.is_cancelled():
        results.setdefault("retry_attempted_count", 0)
        results.setdefault("retry_success_count", 0)
        results.setdefault("retry_failed_count", 0)
        return

    total = len(failed_jobs)
    attempted = 0
    retry_success = 0
    retry_failed = 0

    log_message(
        f"Retrying {total} failed image(s) once...",
        always_print=True,
    )

    for i, job in enumerate(failed_jobs):
        if cancellation_manager is not None and cancellation_manager.is_cancelled():
            log_message(
                "Batch retry cancelled by user; remaining retries skipped.",
                always_print=True,
            )
            break

        error_key = job["error_key"]
        img_path = Path(job["img_path"])
        source_path = job.get("source_path")
        if source_path is None:
            source_path = resolve_source_path(img_path, source_path_map)

        output_path, display_path, _ = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )

        attempted += 1
        try:
            if progress_callback:
                # Keep retries in the high progress band so UI still looks near-complete.
                frac = 0.95 + 0.05 * (i / max(total, 1))
                progress_callback(
                    frac,
                    f"Retrying failed image {i + 1}/{total}: {display_path}",
                )

            log_message(
                f"Retrying {i + 1}/{total}: {display_path}",
                always_print=True,
            )

            if cancellation_manager is not None and cancellation_manager.is_cancelled():
                attempted -= 1
                break
            _translate_batch_page(
                img_path,
                config,
                output_path,
                cancellation_manager=cancellation_manager,
            )
            results["success_count"] = int(results.get("success_count", 0)) + 1
            results["error_count"] = max(0, int(results.get("error_count", 0)) - 1)
            errors = results.setdefault("errors", {})
            errors.pop(error_key, None)
            failed_paths = list(results.get("failed_image_paths") or [])
            before_len = len(failed_paths)
            failed_paths = [p for p in failed_paths if p != source_path]
            if len(failed_paths) == before_len:
                log_message(
                    f"Retry success path cleanup: source_path not in "
                    f"failed_image_paths ({source_path!r})",
                    always_print=True,
                )
            results["failed_image_paths"] = failed_paths
            retry_success += 1
            log_message(
                f"Retry succeeded: {display_path}",
                always_print=True,
            )
        except CancellationError:
            attempted -= 1
            log_message(
                f"Batch retry cancelled during {display_path}; "
                "remaining retries skipped.",
                always_print=True,
            )
            break
        except Exception as e:
            results.setdefault("errors", {})[error_key] = str(e)
            retry_failed += 1
            log_message(
                f"Retry failed for {display_path}: {e}",
                always_print=True,
            )

    results["retry_attempted_count"] = attempted
    results["retry_success_count"] = retry_success
    results["retry_failed_count"] = retry_failed

    if progress_callback and attempted > 0:
        try:
            progress_callback(
                0.99,
                f"Retry complete: {retry_success} recovered, {retry_failed} still failed",
            )
        except CancellationError:
            log_message(
                "Batch retry progress callback cancelled after stats flush.",
                always_print=True,
            )


async def _batch_translate_pages(
    image_files: list[Path],
    input_dir: Path,
    config: Settings,
    output_dir: Path,
    preserve_structure: bool,
    progress_callback: Callable[[float, str], None] | None,
    cancellation_manager: Optional["CancellationManager"],
    source_path_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Process images with up to `config.runtime.parallel_requests` pages in flight.

    The first image is processed alone to warm up all ML models (triggering
    lazy loading and YOLO layer fusing on the main thread). Remaining images are
    then processed in parallel with models already initialized; with one worker
    they run in order.
    """
    total_images = len(image_files)
    n_workers = config.runtime.parallel_requests
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": {},
        "failed_image_paths": [],
        "_failed_jobs": [],
    }
    ocr_text_history: dict[Path, list[str]] = {}
    ocr_text_history_lock = threading.Lock()
    ocr_text_ready_events = [threading.Event() for _ in image_files]
    requested_text_context_count = int(
        config.translation.context_pages
    )

    log_message(
        f"Starting batch processing: {total_images} images, {n_workers} worker(s)",
        always_print=True,
    )

    def _record_failure(img_path: Path, display_path: str, error_key: str, error):
        log_message(f"Error processing {display_path}: {error!s}", always_print=True)
        source_path = resolve_source_path(img_path, source_path_map)
        results["error_count"] += 1
        results["errors"][error_key] = str(error)
        results["failed_image_paths"].append(source_path)
        results["_failed_jobs"].append(
            {"error_key": error_key, "img_path": img_path, "source_path": source_path}
        )

    def _remember_ocr_texts(img_path: Path, texts: list[str]) -> None:
        # Kept even when the page fails: one empty bubble doesn't void the rest
        if texts:
            with ocr_text_history_lock:
                ocr_text_history[img_path] = texts

    if n_workers > 1 and config.runtime.parallel_within_pages:
        request_coordinator = BatchRequestCoordinator(
            n_workers, cancellation_manager=cancellation_manager
        )
        config.request_coordinator = request_coordinator
        config.request_coordinator = request_coordinator
        log_message(
            "Intra-page parallel requests enabled",
            always_print=True,
        )
    else:
        config.request_coordinator = None
        config.request_coordinator = None

    # -- Phase 1: process the first image sequentially to warm up models --
    first_img = image_files[0]
    first_output, first_display, first_key = _resolve_output_path(
        first_img, input_dir, output_dir, config, preserve_structure
    )
    log_message(
        f"Processing 1/{total_images}: {first_display} (warming up models)",
        always_print=True,
    )
    try:
        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Batch process cancelled by user.")
        if progress_callback:
            progress_callback(
                0.0, f"Processing image 1/{total_images}: {first_display}"
            )
        first_previous_context_texts = _build_previous_context_texts(
            image_files,
            0,
            config,
            ocr_text_history,
            ocr_text_history_lock,
        )
        first_ocr_texts: list[str] = []
        try:
            _translate_batch_page(
                first_img,
                config,
                first_output,
                cancellation_manager=cancellation_manager,
                previous_context_texts=first_previous_context_texts,
                ocr_texts_out=first_ocr_texts,
            )
        finally:
            _remember_ocr_texts(first_img, first_ocr_texts)
        results["success_count"] += 1
    except CancellationError:
        raise
    except Exception as e:
        _record_failure(first_img, first_display, first_key, e)
    finally:
        ocr_text_ready_events[0].set()

    completed_count = 1
    if progress_callback:
        has_errors = results["error_count"] > 0
        suffix = " (with errors)" if has_errors else ""
        progress_callback(
            1 / total_images, f"Completed 1/{total_images} images{suffix}"
        )

    # -- Phase 2: process remaining images in parallel --
    remaining = image_files[1:]
    if not remaining:
        return results

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Batch process cancelled by user.")

    sem = asyncio.Semaphore(n_workers)
    results_lock = threading.Lock()
    cancelled = False

    def _wait_for_required_previous_ocr(index: int) -> None:
        if requested_text_context_count <= 0:
            return
        start_index = max(0, index - requested_text_context_count)
        for previous_index in range(start_index, index):
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")
            while not ocr_text_ready_events[previous_index].wait(timeout=0.2):
                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Batch process cancelled by user.")
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")

    def _process_single(img_path: Path, index: int) -> tuple[str, str]:
        """Run translate_and_render for a single image. Returns (display_path, error_key)."""
        output_path, display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )
        log_message(
            f"Processing {index + 1}/{total_images}: {display_path}",
            always_print=True,
        )
        def previous_context_texts_provider() -> list[list[str]]:
            _wait_for_required_previous_ocr(index)
            return _build_previous_context_texts(
                image_files,
                index,
                config,
                ocr_text_history,
                ocr_text_history_lock,
            )

        captured_ocr_texts: list[str] = []
        try:
            _translate_batch_page(
                img_path,
                config,
                output_path,
                cancellation_manager=cancellation_manager,
                previous_context_texts_provider=previous_context_texts_provider,
                ocr_texts_out=captured_ocr_texts,
            )
        finally:
            _remember_ocr_texts(img_path, captured_ocr_texts)
        return display_path, error_key

    async def _worker(img_path: Path, index: int, executor: ThreadPoolExecutor):
        nonlocal completed_count, cancelled
        try:
            if cancelled or (
                cancellation_manager and cancellation_manager.is_cancelled()
            ):
                cancelled = True
                return

            async with sem:
                if cancelled or (
                    cancellation_manager and cancellation_manager.is_cancelled()
                ):
                    cancelled = True
                    return

                if progress_callback:
                    _, display_name, _ = _resolve_output_path(
                        img_path, input_dir, output_dir, config, preserve_structure
                    )
                    progress_callback(
                        completed_count / total_images,
                        f"Processing image {index + 1}/{total_images}: {display_name}",
                    )

                loop = asyncio.get_running_loop()
                try:
                    if n_workers == 1:
                        # On the main thread, so Ctrl+C stops the batch at once
                        # instead of waiting for a worker thread's page to finish
                        _process_single(img_path, index)
                    else:
                        await loop.run_in_executor(
                            executor, _process_single, img_path, index
                        )
                    with results_lock:
                        results["success_count"] += 1
                        completed_count += 1
                        count = completed_count
                except CancellationError:
                    cancelled = True
                    raise
                except Exception as e:
                    _, display_path, error_key = _resolve_output_path(
                        img_path, input_dir, output_dir, config, preserve_structure
                    )
                    with results_lock:
                        _record_failure(img_path, display_path, error_key, e)
                        completed_count += 1
                        count = completed_count

                if progress_callback:
                    progress = count / total_images
                    has_errors = results["error_count"] > 0
                    suffix = " (with errors)" if has_errors else ""
                    progress_callback(
                        progress, f"Completed {count}/{total_images} images{suffix}"
                    )
        except CancellationError:
            cancelled = True
            raise
        finally:
            ocr_text_ready_events[index].set()

            if cancelled:
                for event in ocr_text_ready_events:
                    event.set()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        tasks = [_worker(img, i, executor) for i, img in enumerate(remaining, start=1)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for exc in gathered:
        if isinstance(exc, CancellationError):
            raise exc

    return results


def batch_translate_images(
    input_dir: str | Path,
    config: Settings,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    preserve_structure: bool = False,
    cancellation_manager: Optional["CancellationManager"] = None,
    source_path_map: dict[str, str] | None = None,
    glossary_name: str | None = None,
    glossary_path: str | None = None,
) -> dict[str, Any]:
    """
    Process all images in a directory using a configuration object.

    Args:
        input_dir (str or Path): Directory containing images to process
        config (Settings): Configuration object containing all settings.
        output_dir (str or Path, optional): Directory to save translated images.
                                            If None, uses input_dir / "output_translated".
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0, message).
        preserve_structure (bool): If True, recursively process subdirectories and preserve folder structure
                                   in the output. If False, only processes files in the root directory.
        source_path_map: Optional mapping from processing paths to original source paths
            (used when UI/CLI copies inputs into a temp directory).
        glossary_name: Name for a new terminology memory file (the input folder or ZIP
            name). Defaults to the first image's stem.
        glossary_path: Existing terminology memory file to reuse (a retried run's
            glossary) when the config doesn't set one; takes precedence over
            glossary_name.

    Returns:
        dict: Processing results with keys:
            - "success_count": Number of successfully processed images
            - "error_count": Number of images that failed to process
            - "errors": Dictionary mapping filenames to error messages
            - "failed_image_paths": Absolute source paths of failed images
            - "failed_paths_file": Path to failed_paths.txt when written
    """
    empty_results = {
        "success_count": 0,
        "error_count": 0,
        "errors": {},
        "failed_image_paths": [],
    }

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return empty_results

    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    if preserve_structure:
        image_files = []
        for root, _dirs, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

    def _batch_sort_key(path: Path):
        try:
            sort_path = (
                path.relative_to(input_dir) if preserve_structure else Path(path.name)
            )
        except ValueError:
            sort_path = path
        return natural_path_sort_key(sort_path)

    image_files.sort(key=_batch_sort_key)

    if not image_files:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return empty_results

    translates = not config.runtime.cleaning_only
    if not (translates and config.translation.glossary):
        glossary_path = None
    elif config.translation.glossary_path:
        glossary_path = config.translation.glossary_path
    else:
        glossary_path = glossary_path or str(
            default_glossary_path(glossary_name or image_files[0].stem)
        )
        config = config.updated(**{"translation.glossary_path": Path(glossary_path)})
    if glossary_path:
        _check_glossary_path(Path(glossary_path))
        log_message(f"Terminology memory: {glossary_path}", always_print=True)

    total_images = len(image_files)
    start_batch_time = time.time()

    created_output_dir = not output_dir.exists()
    os.makedirs(output_dir, exist_ok=True)
    try:
        if progress_callback:
            progress_callback(
                0.0, f"Starting batch processing of {total_images} images..."
            )

        results = asyncio.run(
            _batch_translate_pages(
                image_files=image_files,
                input_dir=input_dir,
                config=config,
                output_dir=output_dir,
                preserve_structure=preserve_structure,
                progress_callback=progress_callback,
                cancellation_manager=cancellation_manager,
                source_path_map=source_path_map,
            )
        )
        failed_jobs = results.pop("_failed_jobs", [])

        if _should_run_failed_retry(config, failed_jobs, cancellation_manager):
            _retry_failed_batch_images(
                failed_jobs=failed_jobs,
                results=results,
                config=config,
                input_dir=input_dir,
                output_dir=output_dir,
                preserve_structure=preserve_structure,
                progress_callback=progress_callback,
                cancellation_manager=cancellation_manager,
                source_path_map=source_path_map,
            )
    except (CancellationError, KeyboardInterrupt):
        # Only empty subfolders remain when no page was saved yet.
        if created_output_dir and not any(p.is_file() for p in output_dir.rglob("*")):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    copy_comic_info(
        input_dir, output_dir, config.translation.target_language, preserve_structure
    )

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"Batch complete: {results['success_count']}/{total_images} images in "
        f"{total_batch_time:.2f}s ({seconds_per_image:.2f}s/image)",
        always_print=True,
    )
    if results.get("retry_attempted_count"):
        log_message(
            f"Retry pass: {results.get('retry_success_count', 0)} recovered, "
            f"{results.get('retry_failed_count', 0)} still failed "
            f"(of {results['retry_attempted_count']} attempted)",
            always_print=True,
        )
    if results["error_count"] > 0:
        log_message(f"Failed: {results['error_count']} images", always_print=True)
        for filename, error_msg in results["errors"].items():
            log_message(f"  - {filename}: {error_msg}", always_print=True)

    if glossary_path:
        results["glossary_path"] = glossary_path

    still_failed = set(results.get("failed_image_paths") or [])
    failures = [
        (
            Path(job["img_path"]),
            Path(job["img_path"]).relative_to(input_dir)
            if preserve_structure
            else Path(Path(job["img_path"]).name),
        )
        for job in failed_jobs
        if job["source_path"] in still_failed
    ]
    succeeded = [
        source
        for source in (resolve_source_path(p, source_path_map) for p in image_files)
        if source not in still_failed
    ]
    failed_file = record_failed_images(output_dir, failures, succeeded, glossary_path)
    if failed_file:
        results["failed_paths_file"] = str(failed_file)
        log_message(
            f"Failed image paths saved to: {failed_file}",
            always_print=True,
        )

    return results

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import cv2
import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.dataset.core import Dataset, MOTChallengeDataset
from trackers.eval.metrics import (
    instantiate_metrics,
)
from trackers.eval.metrics.base_tracking_metric import TrackingMetric
from trackers.eval.utils.save_tracks import load_tracks_from_disk, save_tracks

# from trackers import SORTTracker

# --- Define MOT Constants at Module Level ---
MOT_PEDESTRIAN_ID = 1
MOT_DISTRACTOR_IDS = [
    2,
    7,
    8,
    12,
]  # person_on_vehicle, static_person, distractor, reflection
MOT_IGNORE_IDS = [2, 7, 8, 12, 13]  # Includes crowd (13) for ignore, adjust as needed
# Rule for zero_marked GTs: Check if confidence (column 7 in gt.txt) is effectively zero
# Use a small epsilon for float comparison instead of a larger threshold
ZERO_MARKED_EPSILON = 1e-5
# --- End MOT Constants ---


def _add_frame_metadata_to_detections(
    detections: sv.Detections, frame_idx: int, image_path: str
) -> sv.Detections:
    """
    Adds 'frame_idx' and 'image_path' to the `data` attribute of detections.

    Initializes `detections.data` if it's None. Handles empty detections by
    adding empty arrays for the keys.

    Args:
        detections (sv.Detections): The detections object to modify.
        frame_idx (int): The frame index to associate with these detections.
        image_path (str): The path to the image file for this frame.

    Returns:
        sv.Detections: The modified detections object with metadata added to its
        `data` attribute.
    """
    if len(detections) > 0:
        if not detections.data:  # Initialize data if None
            detections.data = {}
        # Create arrays matching the number of detections
        frame_indices = np.full(len(detections), frame_idx, dtype=int)
        image_paths = np.array([image_path] * len(detections), dtype=object)
        detections.data["frame_idx"] = frame_indices
        detections.data["image_path"] = image_paths
    else:
        # Ensure data dictionary exists even for empty detections
        if not detections.data:
            detections.data = {}
        detections.data["frame_idx"] = np.empty(0, dtype=int)
        detections.data["image_path"] = np.empty(0, dtype=object)
    return detections


def generate_tracks(
    dataset: Dataset,
    detection_source: Union[
        MOTChallengeDataset,
        sv.DetectionDataset,
        Callable[[Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    tracker_source: Union[
        BaseTracker,
        Callable[[sv.Detections, Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    output_dir: Optional[Union[str, Path]] = None,
    image_loader: Optional[Callable[[str], np.ndarray]] = None,
) -> Dict[str, sv.Detections]:
    """
    Generates tracking results for all sequences in a dataset.

    Iterates through each sequence and frame, obtains detections using the
    `detection_source`, runs the tracker specified by `tracker_source`, collects
    the results, and optionally saves tracks to disk.

    Args:
        dataset (Dataset): Provides sequences and frames.
        detection_source: Source providing `sv.Detections` per frame. Can be:
            - `MOTChallengeDataset`: Uses loaded public detections.
            - `sv.DetectionDataset`: Uses annotations keyed by image path.
            - Callback `(frame, frame_info) -> sv.Detections`: `frame` can be None.
        tracker_source: Source providing tracked `sv.Detections`. Can be:
            - `BaseTracker`: Implements `update(detections)`.
            - Callback `(detections, frame, frame_info) -> sv.Detections`: `frame`
              can be None. The callback must handle its own state reset if needed.
        output_dir (Optional[Union[str, Path]]): Directory to save tracking results
            as JSON files (one per sequence). Defaults to None (no saving).
        image_loader (Optional[Callable[[str], np.ndarray]]): Custom image loading
            function `(path) -> np.ndarray`. Defaults to `cv2.imread` (BGR)
            followed by conversion to RGB.

    Returns:
        Dict[str, sv.Detections]: Maps sequence names to `sv.Detections` objects
        containing all merged tracks for that sequence. If a sequence has no tracks
        or encounters an error, it maps to an empty `sv.Detections` object.

    Raises:
        ValueError: If the default image loader fails to load an image.
        TypeError: If `detection_source` or `tracker_source` is an unsupported type,
                   or if a callback returns an incorrect type.
    """
    if image_loader is None:

        def default_image_loader(path: str) -> np.ndarray:
            """Default loader using cv2, converting BGR to RGB."""
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_loader = default_image_loader

    # Handle different types of detection sources
    def get_detections(
        frame: Optional[np.ndarray], frame_info: Dict[str, Any]
    ) -> sv.Detections:
        """Internal helper to retrieve detections based on the source type."""
        image_path: Optional[str] = frame_info.get("image_path")  # type: ignore

        if isinstance(detection_source, MOTChallengeDataset):
            if detection_source.has_public_detections:
                # Ensure image_path is not None before calling
                if image_path is None:
                    print(
                        "MOTChallengeDataset source requires 'image_path' in frame_info."
                    )
                    return sv.Detections.empty()
                dets = detection_source.get_public_detections(image_path)
                # MOT public detections should not have class_id, ensure it's None
                if dets.class_id is not None:
                    print(
                        f"Public detections for {image_path} unexpectedly contain "
                        f"'class_id'. Setting to None."
                    )
                    dets.class_id = None
                return dets
            else:
                print(
                    "MOTChallengeDataset provided as detection_source, but public "
                    "detections not loaded. Returning empty detections."
                )
                return sv.Detections.empty()
        elif isinstance(detection_source, sv.DetectionDataset):
            # Ensure image_path is not None before dictionary lookup
            if image_path is None:
                print("sv.DetectionDataset source requires 'image_path' in frame_info.")
                return sv.Detections.empty()
            if image_path in detection_source.annotations:
                return detection_source.annotations[image_path]
            else:
                print(f"No detections found for {image_path} in sv.DetectionDataset.")
                return sv.Detections.empty()
        elif callable(detection_source):
            # It's a callback function
            dets = detection_source(frame, frame_info)
            if not isinstance(dets, sv.Detections):
                raise TypeError(
                    f"Detection source callback must return sv.Detections, "
                    f"but got {type(dets)}"
                )
            return dets
        else:
            raise TypeError(
                f"Unsupported detection_source type: {type(detection_source)}"
            )

    def _process_tracking(
        detections: sv.Detections,
        frame: Optional[np.ndarray],
        frame_info: Dict[str, Any],
    ) -> sv.Detections:
        """Internal helper to run the tracker and add metadata."""
        sequence_name: str = frame_info.get("sequence_name", "UnknownSeq")  # type: ignore
        frame_idx: int = frame_info.get("frame_idx", -1)  # type: ignore
        image_path: str = frame_info.get("image_path", "UnknownPath")  # type: ignore

        if callable(tracker_source):
            tracks = tracker_source(detections, frame, frame_info)
        elif isinstance(tracker_source, BaseTracker):
            tracks = tracker_source.update(detections)
            tracks = tracks[tracks.tracker_id != -1]
        else:
            raise TypeError(f"Unsupported tracker_source type: {type(tracker_source)}")

        # --- Validate Tracker Output ---
        if not isinstance(tracks, sv.Detections):
            # Raise error immediately as this is a fundamental issue
            raise TypeError(
                f"Tracker source (type: {type(tracker_source)}) must return an "
                f"sv.Detections object, but returned type {type(tracks)} for "
                f"sequence {sequence_name}, frame {frame_idx}."
            )

        if tracks.tracker_id is None and len(tracks) > 0:
            print(
                f"Tracker output for sequence {sequence_name}, frame {frame_idx} "
                f"is missing 'tracker_id'. Assigning dummy ID -1. Evaluation might fail."
            )
            tracks.tracker_id = np.full(len(tracks), -1, dtype=int)
        # --- End Validation ---

        # Add frame metadata AFTER tracker has processed
        tracks_with_metadata = _add_frame_metadata_to_detections(
            tracks, frame_idx, image_path
        )

        return tracks_with_metadata

    all_tracks: Dict[str, sv.Detections] = {}
    sequence_names: List[str] = dataset.get_sequence_names()

    for seq_name in sequence_names:
        print(f"--- Generating tracks for sequence: {seq_name} ---")

        sequence_detections_list: List[sv.Detections] = []
        frame_iterator: Iterator[Dict[str, Any]] = dataset.get_frame_iterator(seq_name)
        sequence_had_frames = False  # Track if iterator yields anything

        try:
            for frame_info in frame_iterator:
                sequence_had_frames = True
                frame_idx: int = frame_info["frame_idx"]
                frame_info["sequence_name"] = seq_name  # Add sequence name for context

                if isinstance(tracker_source, BaseTracker) and frame_idx == 0:
                    # Reset tracker state for the first frame of each sequence
                    tracker_source.reset()


                # Load the frame image
                frame: Optional[np.ndarray] = None
                image_path: Optional[str] = frame_info.get("image_path")
                if image_path:
                    try:
                        frame = image_loader(image_path)
                    except Exception as e:
                        print(
                            f"Failed to load image for frame {frame_idx} ({image_path}): {e}. "
                            f"Proceeding without image."
                        )
                        # Tracker/detector must handle None frame if this occurs
                else:
                    print(
                        f"Frame {frame_idx} in sequence {seq_name} missing 'image_path'. "
                        f"Proceeding without image."
                    )

                # Get detections for the frame
                detections = get_detections(frame, frame_info)

                # Process tracking for the frame
                tracker_output = _process_tracking(detections, frame, frame_info)
                sequence_detections_list.append(tracker_output)

        except (
            TypeError
        ) as e:  # Catch type errors from _process_tracking or get_detections
            print(f"Type error during processing sequence {seq_name}: {e}")
            print(f"Skipping remaining frames for sequence {seq_name} due to error.")
            sequence_detections_list = []  # Discard partial results for this sequence
        except Exception as e:  # Catch other unexpected errors
            print(f"Unexpected error during processing sequence {seq_name}: {e}")
            print(f"Skipping remaining frames for sequence {seq_name} due to error.")
            sequence_detections_list = []  # Discard partial results

        # --- Merge and Store/Save Results for the Sequence ---
        if (
            sequence_detections_list
        ):  # Only merge if list is not empty (i.e., no error occurred)
            merged_detections = sv.Detections.merge(sequence_detections_list)
            if merged_detections is not None and len(merged_detections) > 0:
                all_tracks[seq_name] = merged_detections
                print(f"Generated {len(merged_detections)} tracks for {seq_name}.")
                # Save to disk if requested
                if output_dir:
                    saved = save_tracks(merged_detections, seq_name, output_dir)
                    if saved:
                        print(f"Saved tracks for {seq_name} to {output_dir}")
            else:
                # Merging resulted in empty detections (shouldn't happen if list wasn't empty)
                print(f"Merging detections for {seq_name} resulted in empty object.")
                all_tracks[seq_name] = sv.Detections.empty()
        elif sequence_had_frames:
            # Sequence had frames, but list is empty (due to error or no tracks generated)
            print(f"No tracks generated or saved for sequence {seq_name}.")
            all_tracks[seq_name] = sv.Detections.empty()
        else:
            # Sequence iterator yielded nothing
            print(f"Sequence {seq_name} appears to be empty.")
            all_tracks[seq_name] = sv.Detections.empty()
        # --- End Merge and Store/Save ---

    return all_tracks


def _evaluate_single_sequence(
    seq_name: str,
    seq_tracks: sv.Detections,
    dataset: Dataset,
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
    preprocess_remove_distractor_matches: bool = True,
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Evaluates tracking metrics for a single sequence.

    Loads ground truth, optionally applies dataset-specific preprocessing,
    validates tracks, computes requested metrics, and handles placeholders and errors.

    Args:
        seq_name (str): The name of the sequence being evaluated.
        seq_tracks (sv.Detections): Tracking result for the sequence. Expected
            to have `tracker_id` and `data['frame_idx']`.
        dataset (Dataset): Used to load ground truth, sequence info, and perform
                           preprocessing via its `preprocess` method.
        metrics_to_compute (Dict[str, TrackingMetric]): Instantiated metric objects.
        placeholder_metrics (List[str]): Metric names requested but not implemented.
        preprocess_remove_distractor_matches (bool): Passed to
            `dataset.preprocess` if applicable. Defaults to True.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: Maps metric names to dictionaries
        containing computed results or error/info messages for this sequence.
    """
    print(f"--- Evaluating sequence: {seq_name} ---")

    # --- Load Ground Truth ---
    try:
        gt_data_raw = dataset.load_ground_truth(seq_name)
        if gt_data_raw is None:
            print(
                f"No ground truth found for sequence {seq_name}. Skipping evaluation."
            )
            # Create error results for all expected metrics
            error_result = {"error": "Ground truth not found"}
            return {
                metric_name: error_result
                for metric_name in list(metrics_to_compute.keys()) + placeholder_metrics
            }
    except Exception as e:
        print(f"Error loading ground truth for {seq_name}: {e}")
        error_result = {"error": f"Failed to load ground truth: {e}"}
        return {
            metric_name: error_result
            for metric_name in list(metrics_to_compute.keys()) + placeholder_metrics
        }
    # --- End Load Ground Truth ---

    # --- Apply Dataset-Specific Preprocessing ---
    # Check if the dataset object has a 'preprocess' method
    can_preprocess = hasattr(dataset, "preprocess") and callable(dataset.preprocess)

    gt_data = gt_data_raw  # Default to raw GT
    seq_tracks_processed = seq_tracks  # Default to raw predictions

    if can_preprocess:
        print(f"Applying dataset preprocessing for sequence: {seq_name}")
        try:
            # Call the dataset's preprocess method
            gt_data, seq_tracks_processed = dataset.preprocess(
                gt_data_raw,
                seq_tracks,
                iou_threshold=0.5,  # Standard threshold for MOT preprocessing matching
                remove_distractor_matches=preprocess_remove_distractor_matches,
            )
            print(
                f"Preprocessing complete. GT: {len(gt_data_raw)} -> {len(gt_data)}, "
                f"Preds: {len(seq_tracks)} -> {len(seq_tracks_processed)}"
            )
        except Exception as e:
            print(
                f"Error during dataset preprocessing for {seq_name}: {e}. "
                f"Attempting evaluation with raw data.",
            )
            # Fallback to using raw data if preprocessing fails
            gt_data = gt_data_raw
            seq_tracks_processed = seq_tracks
    else:
        print(
            f"Skipping dataset preprocessing for sequence: {seq_name} (method not found)"
        )
    # --- End Preprocessing ---

    # Load sequence info (optional, might be needed by some metrics)
    seq_info: Optional[Dict[str, Any]] = None
    try:
        seq_info = dataset.get_sequence_info(seq_name)
    except Exception as e:
        print(f"Could not load sequence info for {seq_name}: {e}")

    # --- Validate Processed Tracks ---
    # Check the tracks *after* potential preprocessing
    if len(seq_tracks_processed) > 0:
        if seq_tracks_processed.tracker_id is None:
            print(
                f"Processed tracks for sequence {seq_name} are missing 'tracker_id'. "
                f"Evaluation might fail or be incorrect."
            )
        if (
            seq_tracks_processed.data is None
            or "frame_idx" not in seq_tracks_processed.data
        ):
            print(
                f"Processed tracks for sequence {seq_name} are missing 'frame_idx' in data. "
                f"Evaluation might fail or be incorrect."
            )
    # --- End Validation ---

    # --- Compute Metrics ---
    seq_results_for_this_seq: Dict[str, Dict[str, Union[float, str]]] = {}
    for metric_name, metric_instance in metrics_to_compute.items():
        try:
            # Pass the PROCESSED gt_data and seq_tracks_processed
            metric_output = metric_instance.compute(
                ground_truth=gt_data,
                predictions=seq_tracks_processed,
                sequence_info=seq_info,
            )
            # Ensure output is a dictionary
            if not isinstance(metric_output, dict):
                print(
                    f"Metric '{metric_name}' compute returned non-dict: {metric_output}. "
                    f"Wrapping in default key '{metric_name}'."
                )
                # Attempt to wrap if possible, otherwise report error
                try:
                    seq_results_for_this_seq[metric_name] = {
                        metric_name: float(metric_output)
                    }
                except (ValueError, TypeError):
                    seq_results_for_this_seq[metric_name] = {
                        "error": f"Invalid non-dict output: {metric_output}"
                    }
            else:
                seq_results_for_this_seq[metric_name] = metric_output
            # Log summary of results or just confirmation
            # Use repr for concise logging of dict content
            print(
                f"  Computed {metric_name}: {seq_results_for_this_seq[metric_name]!r}"
            )
        except Exception as e:
            error_msg = (
                f"Error computing metric '{metric_name}' for sequence {seq_name}: {e}"
            )
            print(error_msg)  # Log traceback
            seq_results_for_this_seq[metric_name] = {"error": str(e)}
    # --- End Compute Metrics ---

    # Add placeholder results
    for metric_name in placeholder_metrics:
        seq_results_for_this_seq[metric_name] = {
            "value": 0.0,
            "message": "Placeholder - Not implemented",
        }
        print(f"  Added placeholder for {metric_name}")

    return seq_results_for_this_seq


def _aggregate_results(
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]],
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Aggregates metric results across all evaluated sequences.

    Calls the `aggregate` method of each `TrackingMetric` instance for implemented
    metrics and handles placeholders and errors appropriately.

    Args:
        metric_results_by_name (Dict[str, List[Dict[str, Union[float, str]]]]):
            Maps metric names to lists of per-sequence result dictionaries.
        metrics_to_compute (Dict[str, TrackingMetric]): Instantiated metric objects.
        placeholder_metrics (List[str]): Placeholder metric names.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: Maps metric names to dictionaries
        containing aggregated results or error/info messages.
    """
    overall_results: Dict[str, Dict[str, Union[str, float]]] = {}
    print("\n--- Aggregating results across sequences ---")

    # --- Aggregate Implemented Metrics ---
    for metric_name, metric_instance in metrics_to_compute.items():
        raw_seq_outputs = metric_results_by_name.get(metric_name, [])
        # Filter out results that contain an 'error' key
        valid_seq_outputs: List[Dict[str, float]] = []
        num_errors = 0
        for res in raw_seq_outputs:
            if isinstance(res, dict):
                if "error" in res:
                    num_errors += 1
                else:
                    # Attempt to convert values to float for aggregation, skip dict if fails
                    try:
                        float_res = {
                            k: float(v)
                            for k, v in res.items()
                            if isinstance(v, (int, float))
                        }
                        # Check if essential keys expected by aggregate are present (optional, depends on metric)
                        valid_seq_outputs.append(float_res)
                    except (ValueError, TypeError):
                        print(
                            f"Could not convert sequence result to float dict for {metric_name}: {res}"
                        )
                        num_errors += 1
            else:
                print(f"Invalid sequence result format for {metric_name}: {res}")
                num_errors += 1

        if num_errors > 0:
            print(f"{num_errors} sequence(s) had errors for metric '{metric_name}'.")

        if not valid_seq_outputs:
            message = f"No valid sequence results to aggregate for {metric_name}"
            if num_errors > 0:
                message += " (due to errors in all sequences)"
            overall_results[metric_name] = {"message": message}
            print(message)
            continue

        try:
            # Pass only the valid, float-converted results to aggregate
            aggregated_result = metric_instance.aggregate(valid_seq_outputs)
            overall_results[metric_name] = aggregated_result
            print(f"  Aggregated {metric_name}: {aggregated_result!r}")
        except Exception as e:
            error_msg = f"Error during aggregation for metric '{metric_name}': {e}"
            print(error_msg)
            overall_results[metric_name] = {"error": f"Aggregation failed: {e}"}
    # --- End Aggregate Implemented Metrics ---

    # --- Handle Placeholder Metrics ---
    for metric_name in placeholder_metrics:
        raw_seq_outputs = metric_results_by_name.get(metric_name, [])
        has_errors = any(
            isinstance(res, dict) and "error" in res for res in raw_seq_outputs
        )
        message = f"Placeholder ({metric_name}) - Not implemented"
        if has_errors:
            message += " (Errors occurred in some sequence results)"

        overall_results[metric_name] = {"value": 0.0, "message": message}
        print(f"  Handled placeholder {metric_name}: {message}")
    # --- End Handle Placeholder Metrics ---

    return overall_results


def evaluate_tracks(
    dataset: Dataset,
    tracks: Optional[Dict[str, sv.Detections]] = None,
    tracks_path: Optional[Union[str, Path]] = None,
    metrics: List[str] = ["HOTA", "CLEAR", "Count"],
    preprocess_remove_distractor_matches: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates tracking results against ground truth using specified metrics.

    Loads tracks either directly via the `tracks` dictionary or from JSON files
    in `tracks_path`. It then evaluates each sequence using the provided `dataset`
    for ground truth and finally aggregates the results across all sequences.

    Args:
        dataset (Dataset): Contains ground truth and sequence information.
        tracks (Optional[Dict[str, sv.Detections]]): Maps sequence names to
            `sv.Detections` objects containing tracks. If provided, `tracks_path`
            is ignored. Defaults to None.
        tracks_path (Optional[Union[str, Path]]): Path to a directory containing
            saved tracking results (JSON files, one per sequence). Used if `tracks`
            is None. Defaults to None.
        metrics (List[str]): Tracking metric names (case-insensitive) to compute
            (e.g., ["Count", "HOTA", "CLEAR"]). Defaults to ["HOTA", "CLEAR", "Count"].
        preprocess_remove_distractor_matches (bool): If True (default), removes
            predictions matched to GT distractors during MOT preprocessing (mimics
            TrackEval default). Set to False to keep these predictions. This is
            relevant primarily for MOTChallenge datasets and CLEAR/HOTA metrics.

    Returns:
        Dict[str, Any]: Contains evaluation results, structured as:
        ```
        {
            "per_sequence": {
                "seq_name_1": {"metric_1": {...}, "metric_2": {...}, ...},
                "seq_name_2": {...},
                ...
            },
            "overall": {
                "metric_1": {...}, "metric_2": {...}, ...
            }
        }
        ```
        Metric results dictionaries contain computed values or error/info messages.

    Raises:
        ValueError: If neither `tracks` nor `tracks_path` is provided.
    """
    if tracks is None and tracks_path is None:
        raise ValueError(
            "Either tracks (Dict[str, sv.Detections]) or tracks_path (str/Path) must be provided"
        )
    if tracks is not None and tracks_path is not None:
        print("Both 'tracks' and 'tracks_path' provided. Using 'tracks' dictionary.")

    # --- Metric Instantiation ---
    try:
        metrics_to_compute, placeholder_metrics = instantiate_metrics(metrics)
        if not metrics_to_compute and not placeholder_metrics:
            print("No valid metrics specified or found. Evaluation will be empty.")
            return {"per_sequence": {}, "overall": {}}
        print(f"Metrics to compute: {list(metrics_to_compute.keys())}")
        if placeholder_metrics:
            print(f"Placeholder metrics: {placeholder_metrics}")
    except Exception as e:
        print(f"Failed to instantiate metrics: {e}")
        return {
            "per_sequence": {},
            "overall": {"error": f"Metric instantiation failed: {e}"},
        }
    # --- End Metric Instantiation ---

    # --- Load Tracks ---
    loaded_tracks: Dict[str, sv.Detections] = {}
    if tracks is not None:
        loaded_tracks = tracks
        print(f"Using provided tracks dictionary for {len(loaded_tracks)} sequences.")
    else:
        sequence_names = dataset.get_sequence_names()
        if not sequence_names:
            print("No sequences found in the dataset to load tracks for.")
            # Return empty results if no sequences exist
            return {
                "per_sequence": {},
                "overall": {
                    metric_name: {
                        "message": f"No sequences in dataset for {metric_name}"
                    }
                    for metric_name in list(metrics_to_compute.keys())
                    + placeholder_metrics
                },
            }
        try:
            # Ensure tracks_path is not None here (checked by initial ValueError)
            loaded_tracks = load_tracks_from_disk(tracks_path or ".", sequence_names)  # type: ignore
        except Exception as e:
            print(f"Failed to load tracks from disk ({tracks_path}): {e}")
            # Return error if loading fails critically
            return {
                "per_sequence": {},
                "overall": {"error": f"Track loading failed: {e}"},
            }
    # --- End Load Tracks ---

    # --- Evaluate Per Sequence ---
    results: Dict[str, Any] = {
        "per_sequence": {},
        "overall": {},
    }
    # Use defaultdict for easier appending of sequence results
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]] = defaultdict(
        list
    )

    if not loaded_tracks:
        print("No tracks loaded or provided for evaluation.")
        # Populate overall results with 'no tracks' messages
        for metric_name in list(metrics_to_compute.keys()) + placeholder_metrics:
            msg_key = "message" if metric_name in placeholder_metrics else "error"
            results["overall"][metric_name] = {
                msg_key: f"No tracks available for evaluation for {metric_name}"
            }
        return results

    # Evaluate only the sequences for which tracks were loaded/provided
    sequences_to_evaluate = list(loaded_tracks.keys())
    print(f"Evaluating {len(sequences_to_evaluate)} sequences: {sequences_to_evaluate}")

    for seq_name in sequences_to_evaluate:
        seq_tracks = loaded_tracks[seq_name]
        # Call the single sequence evaluation function
        seq_results_for_this_seq = _evaluate_single_sequence(
            seq_name=seq_name,
            seq_tracks=seq_tracks,
            dataset=dataset,
            metrics_to_compute=metrics_to_compute,
            placeholder_metrics=placeholder_metrics,
            preprocess_remove_distractor_matches=preprocess_remove_distractor_matches,
        )

        # Store per-sequence results
        results["per_sequence"][seq_name] = seq_results_for_this_seq
        # Collect results by metric name for aggregation
        for metric_name, metric_output in seq_results_for_this_seq.items():
            metric_results_by_name[metric_name].append(metric_output)
    # --- End Evaluate Per Sequence ---

    # --- Aggregate Overall Metrics ---
    if metric_results_by_name:
        results["overall"] = _aggregate_results(
            metric_results_by_name=metric_results_by_name,
            metrics_to_compute=metrics_to_compute,
            placeholder_metrics=placeholder_metrics,
        )
    else:
        # This case should ideally not be reached if loaded_tracks was not empty,
        # but handle defensively.
        print("No metric results collected across sequences for aggregation.")
        for metric_name in list(metrics_to_compute.keys()) + placeholder_metrics:
            results["overall"][metric_name] = {
                "message": "No sequence results collected"
            }
    # --- End Aggregate Overall Metrics ---

    return results


def evaluate_tracker(
    dataset: Dataset,
    detection_source: Union[
        MOTChallengeDataset,
        sv.DetectionDataset,
        Callable[[Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    tracker_source: Union[
        BaseTracker,
        Callable[[sv.Detections, Optional[np.ndarray], Dict[str, Any]], sv.Detections],
    ],
    cache_tracks: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    metrics: List[str] = ["HOTA", "CLEAR", "Count"],
    image_loader: Optional[Callable[[str], np.ndarray]] = None,
    preprocess_remove_distractor_matches: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end tracker evaluation: generates tracks and then evaluates them.

    This function first calls `generate_tracks` to produce tracking results for
    all sequences in the `dataset` using the specified `detection_source` and
    `tracker_source`. It can optionally cache these tracks. Then, it calls
    `evaluate_tracks` to compute the specified `metrics` against the ground truth
    provided by the `dataset`.

    Args:
        dataset (Dataset): Provides sequences, frames, and ground truth.
        detection_source: Source providing `sv.Detections` per frame. Passed to
            `generate_tracks`. See `generate_tracks` docstring for options.
        tracker_source: Source providing tracked `sv.Detections`. Passed to
            `generate_tracks`. See `generate_tracks` docstring for options.
        cache_tracks (bool): If True, saves generated tracks to `cache_dir`.
            Defaults to False.
        cache_dir (Optional[Union[str, Path]]): Directory to save/load cached
            tracking results if `cache_tracks` is True. Defaults to None.
        metrics (List[str]): Tracking metric names (case-insensitive) to compute
            (e.g., ["Count", "HOTA", "CLEAR"]). Passed to `evaluate_tracks`.
            Defaults to ["HOTA", "CLEAR", "Count"].
        image_loader (Optional[Callable[[str], np.ndarray]]): Custom image loading
            function. Passed to `generate_tracks`. Defaults to None (uses default).
        preprocess_remove_distractor_matches (bool): If True (default), removes
            predictions matched to GT distractors during MOT preprocessing. Passed
            to `evaluate_tracks`. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation metrics, structured
        as returned by `evaluate_tracks`.

    Examples:
        >>> from pathlib import Path
        >>> from trackers.dataset.core import MOTChallengeDataset
        >>> from trackers.sort_tracker import SORTTracker # Example tracker
        >>>
        >>> # Setup dataset (assuming MOT17 data is in ./data/mot/MOT17/train)
        >>> dataset_path = Path("./data/mot/MOT17/train")
        >>> if dataset_path.exists():
        ...     mot_dataset = MOTChallengeDataset(dataset_path=dataset_path)
        ...     # Load public detections to use as input for the tracker
        ...     mot_dataset.load_public_detections(min_confidence=0.1)
        ...
        ...     # Instantiate the tracker
        ...     tracker = SORTTracker()
        ...
        ...     # Run the full evaluation pipeline
        ...     results = evaluate_tracker(
        ...         dataset=mot_dataset,
        ...         detection_source=mot_dataset, # Use public detections from dataset
        ...         tracker_source=tracker,
        ...         metrics=["CLEAR", "Count"], # Evaluate CLEAR and Count metrics
        ...         cache_tracks=True,          # Cache the generated tracks
        ...         cache_dir="./eval_cache",     # Directory for caching
        ...         preprocess_remove_distractor_matches=True # Use default preprocessing
        ...     )
        ...     # Print the overall CLEAR results
        ...     print(results.get("overall", {}).get("CLEAR", "CLEAR results not found")) # doctest: +SKIP
        ... else:
        ...     print(f"Dataset path not found: {dataset_path}") # doctest: +SKIP

    Raises:
        FileNotFoundError: If the dataset path is invalid.
        ValueError: If detection/tracker sources are misconfigured or required
                    data (like public detections) is not loaded.
        TypeError: If sources return unexpected types.
        Exception: For other unexpected errors during generation or evaluation.
    """
    print("Starting end-to-end tracker evaluation...")

    # --- Generate Tracks ---
    output_dir = Path(cache_dir) if cache_tracks and cache_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Track caching enabled. Cache directory: {output_dir}")

    try:
        tracks = generate_tracks(
            dataset=dataset,
            detection_source=detection_source,
            tracker_source=tracker_source,
            output_dir=output_dir,
            image_loader=image_loader,
        )
    except Exception as e:
        print(f"Track generation failed: {e}")
        # Return an error structure consistent with evaluate_tracks output
        return {
            "per_sequence": {},
            "overall": {"error": f"Track generation failed: {e}"},
        }
    # --- End Generate Tracks ---

    # --- Evaluate Tracks ---
    try:
        results = evaluate_tracks(
            dataset=dataset,
            tracks=tracks,  # Pass the generated tracks dictionary
            metrics=metrics,
            preprocess_remove_distractor_matches=preprocess_remove_distractor_matches,
        )
    except Exception as e:
        print(f"Track evaluation failed: {e}")
        # Return an error structure consistent with evaluate_tracks output
        return {
            "per_sequence": {},
            "overall": {"error": f"Track evaluation failed: {e}"},
        }
    # --- End Evaluate Tracks ---

    print("End-to-end tracker evaluation finished.")
    return results


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    from trackers import MOTChallengeDataset, SORTTracker

    # 1. Instantiate a dataset object
    try:
        mot_dataset_path = Path("./data/mot/MOT17/train")  # Example path
        if not mot_dataset_path.exists():
            print(f"Dataset path does not exist: {mot_dataset_path}")
            print("Please update the path in the __main__ block.")
            exit()
        mot_dataset = MOTChallengeDataset(dataset_path=mot_dataset_path)

        # --- Example: Accessing dataset info ---
        available_sequences = mot_dataset.get_sequence_names()
        print(f"Available sequences: {available_sequences}")
        if available_sequences:
            seq_name_example = available_sequences[0]
            print(f"\nInfo for sequence '{seq_name_example}':")
            print(mot_dataset.get_sequence_info(seq_name_example))

            print(f"\nLoading GT for '{seq_name_example}':")
            gt_data = mot_dataset.load_ground_truth(seq_name_example)
            if gt_data is not None:
                print(f"Loaded GT with {len(gt_data)} entries.")
                # print(f"First 5 GT entries:\n{gt_data[:5]}") # Can be verbose
            else:
                print("GT data not loaded.")

            print(f"\nFirst 5 frames for '{seq_name_example}':")
            frame_count = 0
            for frame_info in mot_dataset.get_frame_iterator(seq_name_example):
                print(frame_info)
                frame_count += 1
                if frame_count >= 5:
                    break
        # --- End example access ---

        # 2. Load or create detection source
        # Example: Using MOTChallengeDataset with public detections
        mot_dataset.load_public_detections(min_confidence=0.1)  # Load detections

        # 3. Instantiate a tracker object
        tracker = SORTTracker()  # Use SORT tracker

        # --- Run evaluation (with default preprocessing) ---
        print("\n--- Starting Evaluation (Default Preprocessing) ---")
        cache_dir_default = Path("./cached_tracks_sv_default")
        results_default = evaluate_tracker(
            dataset=mot_dataset,
            detection_source=mot_dataset,  # Use public detections from dataset
            tracker_source=tracker,
            metrics=["Count", "HOTA", "CLEAR"],
            cache_tracks=True,
            cache_dir=cache_dir_default,
            preprocess_remove_distractor_matches=True,  # Explicitly True (default)
        )
        print("\n--- Evaluation Results (Default Preprocessing) ---")
        # Pretty print the JSON results
        print(json.dumps(results_default, indent=2))
        # --- End default evaluation ---

        # --- Run evaluation (without removing distractor matches) ---
        print("\n--- Starting Evaluation (Keep Distractor Matches) ---")
        # Instantiate a new tracker instance to ensure fresh state
        tracker_keep = SORTTracker()
        results_keep = evaluate_tracker(
            dataset=mot_dataset,
            detection_source=mot_dataset,  # Use public detections again
            tracker_source=tracker_keep,  # Use new tracker instance
            metrics=["Count", "HOTA", "CLEAR"],
            cache_tracks=False,  # Don't cache this run to avoid overwriting
            preprocess_remove_distractor_matches=False,  # Set to False
        )
        print("\n--- Evaluation Results (Keep Distractor Matches) ---")
        print(json.dumps(results_keep, indent=2))
        # --- End keep evaluation ---

        # --- Example of evaluating from cached tracks ---
        print("\n--- Evaluating Cached Tracks (Default Preprocessing Run) ---")
        if cache_dir_default.exists():
            results_from_cache = evaluate_tracks(
                dataset=mot_dataset,
                tracks_path=cache_dir_default,  # Point to the cache directory
                metrics=["Count", "CLEAR"],  # Evaluate subset from cache
                preprocess_remove_distractor_matches=True,  # Match the preprocessing used when generating cache
            )
            print("\n--- Cached Evaluation Results (Count, CLEAR only) ---")
            print(json.dumps(results_from_cache, indent=2))
        else:
            print(
                f"Cache directory '{cache_dir_default}' not found, skipping cached evaluation."
            )
        # --- End cached evaluation ---

    except FileNotFoundError as e:
        print(f"File/Directory not found: {e}")
    except ImportError as e:
        print(f"Import Error: {e}. Make sure all dependencies are installed.")
    except Exception as e:
        # Log the full traceback for unexpected errors
        print(f"An unexpected error occurred: {e}")

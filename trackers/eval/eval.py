import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.dataset.core import Dataset, MOTChallengeDataset
from trackers.eval.metrics import (
    instantiate_metrics,
)
from trackers.eval.metrics.base_tracking_metric import TrackingMetric
from trackers.eval.utils.save_tracks import load_tracks_from_disk, save_tracks
from trackers.sort_tracker import SORTTracker

# --- Define MOT Constants at Module Level ---
MOT_PEDESTRIAN_ID = 1
# Adjusted based on common MOT classes, ensure these match your dataset's conventions
MOT_DISTRACTOR_IDS = [
    2,
    7,
    8,
    12,
]  # person_on_vehicle, static_person, distractor, reflection
MOT_IGNORE_IDS = [2, 7, 8, 12, 13]  # Includes crowd (13) for ignore, adjust as needed
# Rule for zero_marked GTs (often confidence=0 in gt.txt, or specific ignore classes)
ZERO_MARKED_CONF_THRESHOLD = 0.01  # Proxy based on confidence column in gt.txt
# --- End MOT Constants ---


def _add_frame_metadata_to_detections(
    detections: sv.Detections, frame_idx: int, image_path: str
) -> sv.Detections:
    """
    Helper function to add 'frame_idx' and 'image_path' to detections.data.

    If detections.data is None, it initializes it. If detections is empty,
    it initializes data with empty arrays for these keys.

    Args:
        detections: The sv.Detections object to modify.
        frame_idx: The frame index to associate with these detections.
        image_path: The path to the image file for this frame.

    Returns:
        The modified sv.Detections object with metadata added to its `data` attribute.
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

    Iterates through each sequence and frame, obtains detections, runs the tracker,
    and collects the results. Optionally saves tracks to disk.

    Args:
        dataset: A Dataset object providing sequences and frames.
        detection_source: Source providing sv.Detections per frame. Can be:
                          - MOTChallengeDataset (uses loaded public detections).
                          - sv.DetectionDataset (uses annotations keyed by image path).
                          - A callback function: `(frame, frame_info) -> sv.Detections`.
                            The frame can be None if image loading fails.
        tracker_source: Source providing tracked sv.Detections. Can be:
                        - A BaseTracker object (implements update method).
                        - A callback function:
                          `(detections, frame, frame_info) -> sv.Detections`.
                          The frame can be None if image loading fails.
                          *Note: The callback function should handle resetting
                          the tracker state internally if needed (e.g., at frame 1).*
        output_dir: Optional directory path (str or Path) to save tracking results
                    as JSON files (one per sequence).
        image_loader: Optional custom image loading function `(path) -> np.ndarray`.
                      Defaults to using `cv2.imread` and converting to RGB.

    Returns:
        A dictionary mapping sequence names (str) to sv.Detections objects
        containing all merged tracks for that sequence. If a sequence has no tracks
        or encounters an error during processing, it will map to an empty
        sv.Detections object.
    """
    if image_loader is None:

        def default_image_loader(path):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        image_loader = default_image_loader

    # Handle different types of detection sources
    def get_detections(
        frame: Optional[np.ndarray], frame_info: Dict[str, Union[str, int]]
    ) -> sv.Detections:
        image_path = frame_info.get("image_path")

        if isinstance(detection_source, MOTChallengeDataset):
            if detection_source.has_public_detections:
                dets = detection_source.get_public_detections(image_path)
                if dets.class_id is not None:
                    print(f"Warning: Detected class_id is not None for {image_path}")
                    # Ensure class_id is None
                    dets.class_id = None
                return dets
            else:
                print(
                    "Warning: MOTChallengeDataset provided as detection_source, \
                        but public detections not loaded. Returning empty."
                )
                return sv.Detections.empty()
        elif isinstance(detection_source, sv.DetectionDataset):
            if image_path in detection_source.annotations:
                dets = detection_source.annotations[image_path]
                return dets
            else:
                print(
                    f"Warning: No detections found for {image_path} in \
                        sv.DetectionDataset"
                )
                return sv.Detections.empty()
        elif callable(detection_source):
            # It's a callback function
            dets = detection_source(frame, frame_info)
            if not isinstance(dets, sv.Detections):
                raise TypeError(
                    f"Detection source callback must return sv.Detections, \
                        but got {type(dets)}"
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
        sequence_name = frame_info.get("sequence_name")
        frame_idx = frame_info.get("frame_idx")
        image_path = frame_info.get("image_path")  # Get image path for metadata

        if callable(tracker_source):
            # It's a callback function
            tracks = tracker_source(detections, frame, frame_info)
        elif isinstance(tracker_source, BaseTracker):
            # It's a Tracker object
            tracks = tracker_source.update(detections)
        else:
            raise TypeError(f"Unsupported tracker_source type: {type(tracker_source)}")

        # --- Check tracker output type ---
        if not isinstance(tracks, sv.Detections):
            raise TypeError(
                f"Tracker source (type: {type(tracker_source)}) must return an \
                    sv.Detections object, "
                f"but returned type {type(tracks)} for sequence {sequence_name}, \
                    frame {frame_idx}."
            )
        # --- End check ---

        # Ensure tracker_id is present
        if tracks.tracker_id is None and len(tracks) > 0:
            print(
                f"Warning: Tracker output for sequence {sequence_name}, \
                    frame {frame_idx} is missing 'tracker_id'. Evaluation might fail."
            )
            # Optionally assign dummy IDs or raise error depending on strictness needed
            tracks.tracker_id = np.full(len(tracks), -1)  # Example: assign dummy ID

        # Add frame metadata AFTER tracker has processed
        tracks_with_metadata = _add_frame_metadata_to_detections(
            tracks, frame_idx, image_path
        )

        return tracks_with_metadata

    # Generate tracks for all sequences
    all_tracks: Dict[str, sv.Detections] = {}
    sequence_names = dataset.get_sequence_names()

    for seq_name in sequence_names:
        print(f"\n--- Generating tracks for sequence: {seq_name} ---")

        sequence_detections_list: List[
            sv.Detections
        ] = []  # Store sv.Detections per frame
        frame_iterator = dataset.get_frame_iterator(seq_name)

        for frame_info in frame_iterator:
            frame_idx = frame_info["frame_idx"]
            frame_info["sequence_name"] = seq_name  # Add sequence name for context

            # Load the frame image
            frame = None
            if "image_path" in frame_info:
                try:
                    frame = image_loader(frame_info["image_path"])
                except Exception as e:
                    print(
                        f"Warning: Failed to load image for frame {frame_idx}: {e}. \
                            Proceeding without image."
                    )
                    # Decide how to handle - skip frame? pass None to detector/tracker?
                    # Passing None might cause errors downstream if not handled.
                    # For now, we proceed, but detector/tracker must handle None frame.

            detections = get_detections(frame, frame_info)

            # Process tracking
            try:
                tracker_output = _process_tracking(detections, frame, frame_info)
            except TypeError as e:  # Catch the type error from process_tracking
                print(f"Error during tracking: {e}")

                print(
                    f"Skipping remaining frames for sequence \
                        {seq_name} due to tracker output error."
                )
                sequence_detections_list = []  # Clear tracks for this sequence
                break

            sequence_detections_list.append(tracker_output)

        # --- Merge detections for the sequence ---
        merged_detections = sv.Detections.merge(sequence_detections_list)
        # --- End merge ---

        # Only add sequence if processing didn't break early and tracks exist
        if merged_detections is not None and len(merged_detections) > 0:
            all_tracks[seq_name] = merged_detections

            # Save to disk if requested
            save_tracks(merged_detections, seq_name, output_dir)

        elif not sequence_detections_list and any(dataset.get_frame_iterator(seq_name)):
            # Sequence had frames but no tracks were generated (or error occurred)
            print(f"No tracks generated or saved for sequence {seq_name}.")
            all_tracks[seq_name] = (
                sv.Detections.empty()
            )  # Store empty detections object
        elif not any(dataset.get_frame_iterator(seq_name)):
            # Sequence was empty
            print(f"Sequence {seq_name} is empty.")
            all_tracks[seq_name] = sv.Detections.empty()

    return all_tracks


def _relabel_ids(detections: sv.Detections) -> sv.Detections:
    """Relabels tracker_ids to be contiguous integers starting from 0."""
    if len(detections) == 0 or detections.tracker_id is None:
        return detections

    # --- Robust ID Handling ---
    # 1. Filter out potential NaN values first
    valid_ids_mask = ~np.isnan(detections.tracker_id)
    if not np.any(valid_ids_mask):
        # All IDs were NaN or array was empty after filtering
        return detections # Nothing to relabel

    # 2. Get unique integer IDs
    try:
        # Attempt conversion to int, np.unique handles the rest
        unique_ids = np.unique(detections.tracker_id[valid_ids_mask].astype(int))
    except ValueError:
        print("Warning: Could not convert tracker IDs to integers during relabeling. Skipping.")
        return detections

    if len(unique_ids) == 0:
        # Should not happen if valid_ids_mask passed, but as a safeguard
        return detections
    # --- End Robust ID Handling ---


    # Now unique_ids contains only valid integers
    max_id = np.max(unique_ids)
    min_id = np.min(unique_ids)

    # The previous type check is no longer needed as we ensured integer types
    # if not np.issubdtype(type(max_id), np.integer): # Removed check
    #     print(
    #         f"Warning: Non-integer max unique ID found during relabeling: {max_id}. Skipping relabel."
    #     )
    #     return detections

    offset = 0
    if min_id < 0:
        print(
            f"Warning: Negative tracker IDs found ({min_id}). Shifting IDs for relabeling."
        )
        offset = -min_id
        max_id += offset # Adjust max_id after offset calculation

    # Check max_id validity after potential offset adjustment
    if np.isnan(max_id):
        print("Warning: Max ID is NaN during relabeling after offset. Skipping.")
        return detections

    # Ensure id_map size is correct integer
    map_size = int(max_id) + 1
    id_map = np.full(map_size, fill_value=-1, dtype=int)
    new_id_counter = 0
    # Initialize new_ids based on the original tracker_id shape and type
    new_ids = np.full_like(detections.tracker_id, fill_value=-1, dtype=int)

    # Iterate through the original positions where IDs were valid
    original_indices = np.where(valid_ids_mask)[0]
    for i in original_indices:
        original_id = int(detections.tracker_id[i]) + offset # Apply offset
        if original_id >= map_size or original_id < 0: # Bounds check
             print(f"Warning: Original ID {original_id-offset} out of bounds for map during relabeling. Skipping ID.")
             continue

        if id_map[original_id] == -1:
            id_map[original_id] = new_id_counter
            new_ids[i] = new_id_counter
            new_id_counter += 1
        else:
            new_ids[i] = id_map[original_id]

    # Handle potential -1s if any IDs were invalid/NaN or out of bounds
    if np.any(new_ids[valid_ids_mask] == -1): # Check only where IDs were originally valid
        print(
            "Warning: Some valid tracker IDs could not be relabeled (check bounds warnings)."
        )

    detections.tracker_id = new_ids
    return detections


def _preprocess_mot_sequence(
    gt_dets: sv.Detections, pred_dets: sv.Detections, iou_threshold: float = 0.5
) -> Tuple[sv.Detections, sv.Detections]:
    """
    Applies MOT specific preprocessing based on TrackEval logic.
    Removes tracker detections matching GT distractors.
    Removes GT distractors and zero-marked GTs.
    Relabels IDs.
    """
    gt_out_list = []
    pred_out_list = []

    # --- Input Validation ---
    if (
        "frame_idx" not in gt_dets.data
        or gt_dets.tracker_id is None
        or gt_dets.class_id is None
        or gt_dets.confidence is None
    ):
        print(
            "Warning: GT detections missing required fields (frame_idx, tracker_id, class_id, confidence) for MOT preprocessing. Skipping."
        )
        return gt_dets, pred_dets
    if "frame_idx" not in pred_dets.data or pred_dets.tracker_id is None:
        print(
            "Warning: Prediction detections missing required fields (frame_idx, tracker_id) for MOT preprocessing. Skipping."
        )
        return gt_dets, pred_dets

    all_frame_indices = sorted(
        list(set(gt_dets.data["frame_idx"]).union(set(pred_dets.data["frame_idx"])))
    )

    for frame_idx in all_frame_indices:
        gt_dets_t = gt_dets[gt_dets.data["frame_idx"] == frame_idx]
        pred_dets_t = pred_dets[pred_dets.data["frame_idx"] == frame_idx]

        # --- TrackEval Preprocessing Step 1 & 2: Remove tracker dets matching distractor GTs ---
        to_remove_tracker_indices = np.array([], dtype=int)
        if len(gt_dets_t) > 0 and len(pred_dets_t) > 0:
            # Match all preds against all GTs for this frame
            similarity = sv.detection.utils.box_iou_batch(
                gt_dets_t.xyxy, pred_dets_t.xyxy
            )
            match_scores = similarity.copy()
            match_scores[match_scores < iou_threshold - np.finfo("float").eps] = 0

            match_rows, match_cols = linear_sum_assignment(
                -match_scores
            )  # Maximize score
            valid_match_mask = (
                match_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
            )
            match_rows = match_rows[valid_match_mask]
            match_cols = match_cols[valid_match_mask]

            # Identify matches where GT is a distractor
            matched_gt_classes = gt_dets_t.class_id[match_rows]
            is_distractor_match = np.isin(matched_gt_classes, MOT_DISTRACTOR_IDS)
            to_remove_tracker_indices = match_cols[is_distractor_match]

        # Filter tracker detections for the frame
        if len(to_remove_tracker_indices) > 0:
            pred_keep_mask = np.ones(len(pred_dets_t), dtype=bool)
            pred_keep_mask[to_remove_tracker_indices] = False
            pred_dets_t_filtered = pred_dets_t[pred_keep_mask]
        else:
            pred_dets_t_filtered = pred_dets_t

        # --- TrackEval Preprocessing Step 4: Remove unwanted GT dets ---
        # Keep only pedestrian class (ID 1) and remove zero_marked
        gt_is_pedestrian = gt_dets_t.class_id == MOT_PEDESTRIAN_ID
        # gt_is_zero_marked = (gt_dets_t.confidence < ZERO_MARKED_CONF_THRESHOLD) | np.isin(gt_dets_t.class_id, MOT_IGNORE_IDS)
        # Let's use TrackEval's logic more directly: zero_marked is column 6 (confidence in MOT format) == 0
        # Assuming confidence field holds this value from parsing gt.txt
        gt_is_zero_marked = (
            gt_dets_t.confidence < ZERO_MARKED_CONF_THRESHOLD
        )  # Use threshold as proxy

        gt_keep_mask = gt_is_pedestrian & ~gt_is_zero_marked
        gt_dets_t_filtered = gt_dets_t[gt_keep_mask]

        # Append filtered detections for the frame
        if len(gt_dets_t_filtered) > 0:
            gt_out_list.append(gt_dets_t_filtered)
        if len(pred_dets_t_filtered) > 0:
            pred_out_list.append(pred_dets_t_filtered)

    # Merge filtered detections across all frames
    gt_processed = (
        sv.Detections.merge(gt_out_list) if gt_out_list else sv.Detections.empty()
    )
    pred_processed = (
        sv.Detections.merge(pred_out_list) if pred_out_list else sv.Detections.empty()
    )

    # --- TrackEval Preprocessing Step 6: Relabel IDs ---
    gt_processed = _relabel_ids(gt_processed)
    pred_processed = _relabel_ids(pred_processed)

    return gt_processed, pred_processed


def _evaluate_single_sequence(
    seq_name: str,
    seq_tracks: sv.Detections,
    dataset: Dataset,
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Evaluates tracking metrics for a single sequence.

    Loads ground truth, validates tracks, computes requested metrics, and handles
    placeholders and errors.

    Args:
        seq_name: The name of the sequence being evaluated.
        seq_tracks: The sv.Detections object containing tracking result for the sequence
                    Expected to have `tracker_id` and `data['frame_idx']`.
        dataset: The Dataset object used to load ground truth and sequence info.
        metrics_to_compute: A dictionary mapping metric names to their instantiated
                            TrackingMetric objects.
        placeholder_metrics: A list of metric names requested but not implemented.

    Returns:
        A dictionary where keys are metric names (including placeholders) and
        values are dictionaries containing the computed metric results or error/info
        messages for this sequence.
    """
    print(f"\n--- Evaluating sequence: {seq_name} ---")

    # Load ground truth
    gt_data_raw = dataset.load_ground_truth(seq_name)  # Load raw GT
    if gt_data_raw is None:
        print(f"Warning: No ground truth for sequence {seq_name}, skipping evaluation")
        seq_results_error = {
            metric_name: {"error": "Ground truth not found"}
            for metric_name in metrics_to_compute
        }
        seq_results_error.update(
            {
                metric_name: {"error": "Ground truth not found"}
                for metric_name in placeholder_metrics
            }
        )
        return seq_results_error

    # --- Apply MOT Preprocessing ---
    # Check if the metric is CLEAR or if the dataset is MOTChallenge based
    # This check might need refinement depending on how datasets/metrics are identified
    is_mot_eval = isinstance(dataset, MOTChallengeDataset) and any(
        m.name == "CLEAR" for m in metrics_to_compute.values()
    )

    if is_mot_eval:
        print(f"Applying MOT preprocessing for sequence: {seq_name}")
        try:
            # Assuming default IoU threshold of 0.5 for preprocessing matching step
            gt_data, seq_tracks_processed = _preprocess_mot_sequence(
                gt_data_raw, seq_tracks, iou_threshold=0.5
            )
            print(
                f"Preprocessing complete. GT: {len(gt_data_raw)} -> {len(gt_data)}, Preds: {len(seq_tracks)} -> {len(seq_tracks_processed)}"
            )
        except Exception as e:
            print(
                f"Error during MOT preprocessing for {seq_name}: {e}. Skipping preprocessing."
            )
            # Fallback to using raw data? Or return error? For now, fallback.
            gt_data = gt_data_raw
            seq_tracks_processed = seq_tracks
    else:
        # No preprocessing applied for other datasets/metrics
        gt_data = gt_data_raw
        seq_tracks_processed = seq_tracks
    # --- End Preprocessing ---

    # Load sequence info
    seq_info = dataset.get_sequence_info(seq_name)

    # --- Validate sequence tracks before passing to metric ---
    # Validate the *processed* tracks
    if len(seq_tracks_processed) > 0:
        if seq_tracks_processed.tracker_id is None:
            print(
                f"Warning: Processed tracks for sequence {seq_name} are missing \
                    'tracker_id'. Evaluation might fail or be incorrect."
            )
        if "frame_idx" not in seq_tracks_processed.data:
            print(
                f"Warning: Processed tracks for sequence {seq_name} are missing \
                    'frame_idx' in data. Evaluation might fail or be incorrect."
            )
    # --- End Validation ---

    # Compute metrics for this sequence using processed data
    seq_results_for_this_seq: Dict[str, Dict[str, Union[float, str]]] = {}
    for metric_name, metric_instance in metrics_to_compute.items():
        try:
            # Pass the PROCESSED gt_data and seq_tracks_processed
            metric_output = metric_instance.compute(
                ground_truth=gt_data,  # Use processed GT
                predictions=seq_tracks_processed,  # Use processed predictions
                sequence_info=seq_info,
            )
            if not isinstance(metric_output, dict):
                print(
                    f"Warning: Metric '{metric_name}' compute returned non-dict: \
                        {metric_output}. Wrapping."
                )
                seq_results_for_this_seq[metric_name] = {metric_name: metric_output}
            else:
                seq_results_for_this_seq[metric_name] = metric_output
            print(f"  {metric_name}: {seq_results_for_this_seq[metric_name]}")
        except Exception as e:
            error_msg = (
                f"Error computing metric '{metric_name}' for sequence {seq_name}: {e}"
            )
            print(error_msg)
            seq_results_for_this_seq[metric_name] = {"error": str(e)}

    # Add placeholder results
    for metric_name in placeholder_metrics:
        seq_results_for_this_seq[metric_name] = {
            "value": 0.0,
            "message": "Placeholder - Not implemented",
        }

    return seq_results_for_this_seq


def _aggregate_results(
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]],
    metrics_to_compute: Dict[str, TrackingMetric],
    placeholder_metrics: List[str],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Aggregates metric results across all evaluated sequences.

    Calls the `aggregate` method of each TrackingMetric instance for implemented
    metrics and handles placeholders and errors.

    Args:
        metric_results_by_name: A dictionary mapping metric names to lists of
                                per-sequence result dictionaries.
        metrics_to_compute: A dictionary mapping metric names to their instantiated
                            TrackingMetric objects.
        placeholder_metrics: A list of placeholder metric names.

    Returns:
        A dictionary where keys are metric names (including placeholders) and
        values are dictionaries containing the aggregated metric results or
        error/info messages.
    """
    overall_results: Dict[str, Dict[str, Union[str, float]]] = {}

    for metric_name, metric_instance in metrics_to_compute.items():
        raw_seq_outputs = metric_results_by_name[metric_name]
        valid_seq_outputs: List[Dict[str, float]] = [
            res
            for res in raw_seq_outputs
            if isinstance(res, dict) and "error" not in res
        ]

        if not valid_seq_outputs:
            if any(isinstance(res, dict) and "error" in res for res in raw_seq_outputs):
                overall_results[metric_name] = {
                    "message": f"Aggregation skipped due to errors in sequence \
                        results for {metric_name}"
                }
            else:
                overall_results[metric_name] = {
                    "message": f"No valid sequence results to aggregate for \
                        {metric_name}"
                }
            print(overall_results[metric_name].get("message", "Aggregation issue"))
            continue

        try:
            aggregated_result = metric_instance.aggregate(valid_seq_outputs)
            overall_results[metric_name] = aggregated_result
        except Exception as e:
            print(f"Error during aggregation for metric '{metric_name}': {e}")
            overall_results[metric_name] = {"error": f"Aggregation failed: {e}"}

    for metric_name in placeholder_metrics:
        raw_seq_outputs = metric_results_by_name[metric_name]
        has_errors = any(
            isinstance(res, dict) and "error" in res for res in raw_seq_outputs
        )
        if has_errors:
            overall_results[metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) \
                    - Errors in sequence results",
            }
        else:
            overall_results[metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) - Not implemented",
            }

    return overall_results


def evaluate_tracks(
    dataset: Dataset,
    tracks: Optional[Dict[str, sv.Detections]] = None,
    tracks_path: Optional[Union[str, Path]] = None,
    metrics: List[str] = ["HOTA", "CLEAR", "Count"],
) -> Dict[str, Any]:
    """
    Evaluates tracking results against ground truth using specified metrics.

    Loads tracks either directly or from disk, evaluates each sequence, and then
    aggregates the results.

    Args:
        dataset: The Dataset object containing ground truth and sequence information.
        tracks: Optional dictionary mapping sequence names (str) to sv.Detections
                objects containing all tracks for the sequence. If provided,
                `tracks_path` is ignored.
        tracks_path: Optional path (str or Path) to a directory containing saved
                     tracking results (JSON files, one per sequence, generated by
                     `save_tracks`). Used if `tracks` is None.
        metrics: A list of tracking metric names (case-insensitive) to compute
                 (e.g., ["Count", "HOTA", "CLEAR"]).

    Returns:
        A dictionary containing evaluation results, structured as:
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
        Metric results dictionaries contain computed values or error/info messages.
    """
    if tracks is None and tracks_path is None:
        raise ValueError("Either tracks or tracks_path must be provided")

    # --- Metric Instantiation ---
    metrics_to_compute, placeholder_metrics = instantiate_metrics(metrics)

    # Load tracks from disk if not provided
    if tracks is None:
        sequence_names = dataset.get_sequence_names()
        if not sequence_names:
            print("Warning: No sequences found in the dataset to load tracks for.")
            return {"per_sequence": {}, "overall": {}}

        tracks = load_tracks_from_disk(tracks_path or ".", sequence_names)  # type: ignore

    # Compute metrics for each sequence
    results: Dict[str, Dict[str, Any]] = {
        "per_sequence": {},
        "overall": {},
    }
    metric_results_by_name: Dict[str, List[Dict[str, Union[float, str]]]] = defaultdict(
        list
    )

    if not tracks:
        print("Warning: No tracks loaded or provided for evaluation.")
        for metric_name in placeholder_metrics:
            results["overall"][metric_name] = {
                "value": 0.0,
                "message": f"Placeholder ({metric_name}) - No tracks evaluated",
            }
        for metric_name in metrics_to_compute:
            results["overall"][metric_name] = {
                "message": f"No tracks evaluated for {metric_name}"
            }
        return results

    for seq_name, seq_tracks in tracks.items():
        seq_results_for_this_seq = _evaluate_single_sequence(
            seq_name, seq_tracks, dataset, metrics_to_compute, placeholder_metrics
        )

        results["per_sequence"][seq_name] = seq_results_for_this_seq
        for metric_name, metric_output in seq_results_for_this_seq.items():
            metric_results_by_name[metric_name].append(metric_output)

    # --- Aggregate overall metrics ---
    if metric_results_by_name:
        results["overall"] = _aggregate_results(
            metric_results_by_name, metrics_to_compute, placeholder_metrics
        )

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
) -> Dict[str, Any]:
    """
    End-to-end tracker evaluation function.

    Combines track generation (`generate_tracks`) and evaluation (`evaluate_tracks`).

    Args:
        dataset: The Dataset object providing sequences, frames, and ground truth.
        detection_source: Source providing sv.Detections per frame. Can be:
                          - MOTChallengeDataset (uses loaded public detections).
                          - sv.DetectionDataset (uses annotations keyed by image path).
                          - A callback function: `(frame, frame_info) -> sv.Detections`.
                            The frame can be None if image loading fails.
        tracker_source: Source providing tracked sv.Detections. Can be:
                        - A BaseTracker object (implements update method).
                        - A callback function:
                          `(detections, frame, frame_info) -> sv.Detections`.
                          The frame can be None if image loading fails.
                          *Note: The callback function should handle resetting
                          the tracker state internally if needed (e.g., at frame 1).*
        cache_tracks: If True, saves the generated tracks to disk using `cache_dir`.
        cache_dir: Directory path (str or Path) to save tracking results if
                   `cache_tracks` is True.
        metrics: A list of tracking metric names (case-insensitive) to compute
                 (e.g., ["Count", "HOTA", "CLEAR"]).
        image_loader: Optional custom image loading function (passed to
                      `generate_tracks`).

    Returns:
        A dictionary containing the evaluation metrics, structured as returned by
        `evaluate_tracks`.
    """
    # Generate tracks
    output_dir = cache_dir if cache_tracks else None
    tracks = generate_tracks(
        dataset=dataset,
        detection_source=detection_source,
        tracker_source=tracker_source,
        output_dir=output_dir,
        image_loader=image_loader,
    )

    # Evaluate tracks
    results = evaluate_tracks(
        dataset=dataset,
        tracks=tracks,  # Pass the Dict[str, sv.Detections]
        metrics=metrics,
    )

    return results


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # 1. Instantiate a dataset object
    try:
        mot_dataset_path = Path("./data/mot/MOT17/train")  # Example path
        if not mot_dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {mot_dataset_path}")
            print("Please update the path in the __main__ block.")
            exit()
        mot_dataset = MOTChallengeDataset(dataset_path=mot_dataset_path)

        # --- Example: Accessing dataset info ---
        print("Available sequences:", mot_dataset.get_sequence_names())
        if mot_dataset.get_sequence_names():
            seq_name_example = mot_dataset.get_sequence_names()[0]
            print(f"\nInfo for sequence '{seq_name_example}':")
            print(mot_dataset.get_sequence_info(seq_name_example))

            print(f"\nLoading GT for '{seq_name_example}':")
            gt_data = mot_dataset.load_ground_truth(seq_name_example)
            if gt_data is not None:
                print(f"Loaded GT with {len(gt_data)} entries. First 5:")
                print(gt_data[:5])  # Print first 5 entries of the list
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

        # Run evaluation
        print("\n--- Starting Evaluation ---")
        results = evaluate_tracker(
            dataset=mot_dataset,
            detection_source=mot_dataset,  # Using detections from MOTChallengeDataset
            tracker_source=tracker,
            metrics=["Count", "HOTA", "CLEAR"],  # Specify desired metrics
            cache_tracks=True,  # Example: cache the tracks
            cache_dir="./cached_tracks_sv",  # Example cache directory
        )
        print("\n--- Evaluation Results ---")
        print(json.dumps(results, indent=2))

        # Example of evaluating from cached tracks
        print("\n--- Evaluating Cached Tracks ---")
        cache_path = "./cached_tracks_sv_merged"
        if Path(cache_path).exists():
            results_from_cache = evaluate_tracks(
                dataset=mot_dataset,
                tracks_path=cache_path,  # Point to the cache directory
                metrics=["Count"],  # Evaluate only Count from cache for demo
            )
            print("\n--- Cached Evaluation Results (Count only) ---")
            print(json.dumps(results_from_cache, indent=2))
        else:
            print(
                f"Cache directory '{cache_path}' not found, skipping cached evaluation."
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"Import Error: {e}. Make sure all dependencies are installed.")
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())

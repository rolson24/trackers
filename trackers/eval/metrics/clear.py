from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.eval.metrics.base_tracking_metric import TrackingMetric

# --- Constants ---
_CONTINUITY_BONUS: float = 1000.0  # Bonus for matching the same ID as the previous step
_MT_THRESHOLD: float = 0.8  # Threshold for Mostly Tracked (MT)
_ML_THRESHOLD: float = 0.2  # Threshold for Mostly Lost (ML)


class CLEARMetric(TrackingMetric):
    """
    Calculates CLEAR metrics (MOTA, MOTP, IDSW, MT, ML, PT, Frag, etc.).

    This implementation computes standard CLEAR metrics based on matching ground
    truth objects to predicted objects frame-by-frame using the Hungarian algorithm
    on an IoU-based cost matrix, potentially augmented with a continuity bonus.

    It requires ground truth and prediction `sv.Detections` objects to have
    `tracker_id` (integer IDs) and `data['frame_idx']` (integer frame indices)
    attributes populated. Ground truth IDs and prediction IDs are expected to be
    pre-processed (e.g., via `_preprocess_mot_sequence`) to be 0-based and
    contiguous for internal calculations, although the metric calculation itself
    handles the mapping.

    References:
        - K. Bernardin, R. Stiefelhagen, "Evaluating Multiple Object Tracking
          Performance: The CLEAR MOT Metrics", 2008.
          https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309
        - Implementation details inspired by TrackEval:
          https://github.com/JonathonLuiten/TrackEval [MIT License]

    Attributes:
        iou_threshold (float): The IoU threshold for considering a match.
        integer_fields (List[str]): Names of metrics that are counts (integers).
        float_fields (List[str]): Names of metrics that are ratios/averages (floats).
        summed_fields (List[str]): Names of fields summed during aggregation.
        fields (List[str]): All metric field names produced.
    """

    @staticmethod
    def _compute_final_fields(res: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates derived CLEAR metrics from base counts.

        Computes metrics like MOTA, MOTP, Recall, Precision, MTR, MLR, PTR, F1,
        sMOTA, MOTAL, etc., based on the accumulated counts (TP, FP, FN, IDSW,
        MOTP_sum, MT, ML, PT, Frag, CLR_Frames). Ensures all expected fields
        exist in the output dictionary, defaulting to 0.0 if a calculation
        results in NaN or if a component is missing.

        Args:
            res (Dict[str, float]): Dictionary containing the base counts
                accumulated over a sequence or aggregated across sequences.
                Expected keys include 'CLR_TP', 'FN', 'FP', 'IDSW', 'MOTP_sum',
                'MT', 'ML', 'PT', 'Frag', 'CLR_Frames'. Missing keys will be
                treated as 0.

        Returns:
            Dict[str, float]: The input dictionary `res` updated with the
            calculated derived metrics. All values are guaranteed to be floats.
        """
        # Ensure necessary base counts are present, default to 0.0 if missing
        required_counts: List[str] = [
            "CLR_TP",
            "FN",
            "FP",
            "IDSW",
            "MOTP_sum",
            "MT",
            "ML",
            "PT",
            "Frag",
            "CLR_Frames",
        ]
        for key in required_counts:
            res.setdefault(key, 0.0)

        # --- Calculate intermediate sums (used in multiple metrics) ---
        # Note: Using res[...] directly as requested, avoiding intermediate vars like num_gt_dets
        # num_gt_dets = res["CLR_TP"] + res["FN"]
        # num_tracker_dets = res["CLR_TP"] + res["FP"]
        num_gt_ids = res["MT"] + res["ML"] + res["PT"]

        # --- Calculate Ratios/Derived Metrics ---
        # Use np.maximum(1.0, ...) to avoid division by zero
        res["MTR"] = res["MT"] / np.maximum(1.0, num_gt_ids)
        res["MLR"] = res["ML"] / np.maximum(1.0, num_gt_ids)
        res["PTR"] = res["PT"] / np.maximum(1.0, num_gt_ids)
        res["Recall"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["FN"])
        res["Precision"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["FP"])

        res["MODA"] = (res["CLR_TP"] - res["FP"]) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )
        res["MOTA"] = (res["CLR_TP"] - res["FP"] - res["IDSW"]) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )
        res["MOTP"] = res["MOTP_sum"] / np.maximum(1.0, res["CLR_TP"])

        # sMOTA (Scalable MOTA - incorporates MOTP)
        res["sMOTA"] = (res["MOTP_sum"] - res["FP"] - res["IDSW"]) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )

        # MOTAL (MOTA Log - reduces impact of high IDSW counts)
        # Use safe log10 calculation
        safe_log_idsw = np.log10(res["IDSW"]) if res["IDSW"] > 0 else 0.0
        res["MOTAL"] = (res["CLR_TP"] - res["FP"] - safe_log_idsw) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )

        # F1 Score (using standard definition: 2*TP / (2*TP + FP + FN))
        # Which simplifies to TP / (TP + 0.5*FP + 0.5*FN)
        res["CLR_F1"] = res["CLR_TP"] / np.maximum(
            1.0, res["CLR_TP"] + 0.5 * res["FN"] + 0.5 * res["FP"]
        )
        res["FP_per_frame"] = res["FP"] / np.maximum(1.0, res["CLR_Frames"])

        # --- Ensure all expected float fields exist ---
        # List includes all derived float metrics calculated above
        float_fields: List[str] = [
            "MOTA",
            "MOTP",
            "MODA",
            "Recall",
            "Precision",
            "MTR",
            "PTR",
            "MLR",
            "CLR_F1",
            "sMOTA",
            "MOTAL",
            "FP_per_frame",
        ]
        for field in float_fields:
            # Ensure field exists and replace potential NaN with 0.0
            res[field] = float(np.nan_to_num(res.get(field, 0.0)))

        return res

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initializes the CLEARMetric calculator.

        Args:
            iou_threshold (float): The Intersection over Union (IoU) threshold
                for considering a detection match (True Positive). Matches below
                this threshold are ignored unless a continuity bonus applies.
                Defaults to 0.5.
        """
        self.iou_threshold: float = iou_threshold

        # Define fields following TrackEval naming convention where possible
        # These lists define the structure of the output dictionaries.
        self.integer_fields: List[str] = [
            "CLR_TP",
            "FN",
            "FP",
            "IDSW",
            "MT",
            "PT",
            "ML",
            "Frag",
            "CLR_Frames",
        ]
        self.float_fields: List[str] = [
            "MOTA",
            "MOTP",
            "MODA",
            "Recall",
            "Precision",
            "MTR",
            "PTR",
            "MLR",
            "CLR_F1",
            "sMOTA",
            "MOTAL",
            "FP_per_frame",
            "MOTP_sum",  # Include MOTP_sum here for completeness
        ]
        # MOTP_sum is summed during aggregation before the final MOTP calculation
        self.summed_fields: List[str] = [
            *self.integer_fields,
            "MOTP_sum",
        ]
        # Combined list of all fields expected in the final output
        self.fields: List[str] = self.float_fields + self.integer_fields

    @property
    def name(self) -> str:
        """Returns the standard name of the metric."""
        return "CLEAR"

    def compute(
        self,
        ground_truth: sv.Detections,
        predictions: sv.Detections,
        sequence_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Computes CLEAR metrics for a single sequence.

        Processes frame by frame, matching predictions to ground truth using the
        Hungarian algorithm based on IoU and continuity. Accumulates counts
        (TP, FP, FN, IDSW, Frag, MOTP_sum) and track statistics (MT, ML, PT).
        Finally, calculates derived metrics like MOTA, MOTP, etc.

        Args:
            ground_truth (sv.Detections): Ground truth annotations. Requires
                `tracker_id` (int) and `data['frame_idx']` (int). Assumes IDs
                are 0-based and contiguous after potential preprocessing.
            predictions (sv.Detections): Tracker predictions. Requires
                `tracker_id` (int) and `data['frame_idx']` (int). Assumes IDs
                are 0-based and contiguous after potential preprocessing.
            sequence_info (Optional[Dict[str, Any]]): Optional sequence metadata
                (unused by CLEAR). Defaults to None.

        Returns:
            Dict[str, float]: A dictionary containing computed CLEAR metric
            components (integer counts like 'CLR_TP', 'FP', etc.) and final
            derived metrics (floats like 'MOTA', 'MOTP', etc.) for the sequence.
            All values are returned as floats.

        Raises:
            ValueError: If `ground_truth` or `predictions` are missing the
                required `tracker_id` or `data['frame_idx']`.
        """
        # Initialize results dictionary with all expected fields set to 0.0
        res: Dict[str, float] = {field: 0.0 for field in self.fields}
        res["MOTP_sum"] = 0.0  # Ensure MOTP_sum is also initialized

        # --- Input Validation ---
        if ground_truth.data is None or "frame_idx" not in ground_truth.data:
            raise ValueError("Ground truth detections must have 'frame_idx' in data.")
        if predictions.data is None or "frame_idx" not in predictions.data:
            raise ValueError("Prediction detections must have 'frame_idx' in data.")
        if ground_truth.tracker_id is None:
            raise ValueError("Ground truth detections must have 'tracker_id'.")
        if predictions.tracker_id is None:
            raise ValueError("Prediction detections must have 'tracker_id'.")
        # --- End Input Validation ---

        # --- Quick exit for empty sequences ---
        if len(ground_truth) == 0 and len(predictions) == 0:
            res["CLR_Frames"] = 0.0
            # Compute final fields even for empty sequence to ensure all keys exist
            return self._compute_final_fields(res)
        # --- End Quick exit ---

        # Determine frame range
        gt_frame_indices: Set[int] = set(ground_truth.data["frame_idx"])
        pred_frame_indices: Set[int] = set(predictions.data["frame_idx"])
        all_frame_indices: List[int] = sorted(
            list(gt_frame_indices.union(pred_frame_indices))
        )

        if (
            not all_frame_indices
        ):  # Should not happen if len > 0 check passed, but safety
            res["CLR_Frames"] = 0.0
            return self._compute_final_fields(res)

        res["CLR_Frames"] = float(len(all_frame_indices))

        # --- Data Structures for Tracking State ---
        # Determine the number of unique GT IDs based on the maximum ID present
        # (Assumes IDs are 0-based and contiguous after preprocessing)
        num_gt_ids: int = 0
        if len(ground_truth) > 0:
            unique_gt_ids_in_input = np.unique(ground_truth.tracker_id)
            if len(unique_gt_ids_in_input) > 0:
                # Add 1 because IDs are 0-based indices
                max_gt_id = np.max(unique_gt_ids_in_input)
                if np.isnan(max_gt_id):
                    print(
                        "NaN found in ground truth tracker IDs. Results may be inaccurate."
                    )
                    # Attempt to filter NaNs if necessary, though preprocessing should handle this
                    valid_ids = unique_gt_ids_in_input[
                        ~np.isnan(unique_gt_ids_in_input)
                    ]
                    num_gt_ids = int(np.max(valid_ids)) + 1 if len(valid_ids) > 0 else 0
                else:
                    num_gt_ids = int(max_gt_id) + 1

        # Per GT ID tracking stats (using 0-based index)
        # Size arrays based on the maximum possible ID + 1
        gt_id_frame_count: np.ndarray = np.zeros(num_gt_ids, dtype=int)
        gt_id_matched_count: np.ndarray = np.zeros(num_gt_ids, dtype=int)
        gt_id_frag_count: np.ndarray = np.zeros(
            num_gt_ids, dtype=int
        )  # Counts fragmentation starts

        # Track matching history for IDSW and Fragmentation calculation
        # Stores the *tracker* ID last matched to each *GT* ID (index)
        prev_tracker_id: np.ndarray = np.full(num_gt_ids, np.nan)
        # Stores the *tracker* ID matched in the *immediately preceding* timestep
        prev_timestep_tracker_id: np.ndarray = np.full(num_gt_ids, np.nan)
        # Boolean flag indicating if a GT ID was matched in the previous timestep
        matched_in_prev_step: np.ndarray = np.zeros(num_gt_ids, dtype=bool)
        # --- End Data Structures ---

        # --- Frame-by-Frame Processing ---
        print(f"Processing {len(all_frame_indices)} frames for CLEAR metrics...")
        for frame_idx in all_frame_indices:
            # Get detections for the current frame
            gt_dets_t: sv.Detections = ground_truth[
                ground_truth.data["frame_idx"] == frame_idx
            ]
            pred_dets_t: sv.Detections = predictions[
                predictions.data["frame_idx"] == frame_idx
            ]

            # Get corresponding IDs (assumed to be 0-based indices)
            gt_ids_t: np.ndarray = gt_dets_t.tracker_id
            pred_ids_t: np.ndarray = pred_dets_t.tracker_id

            # Update frame count for GT IDs present in this frame
            if len(gt_ids_t) > 0:
                # Check bounds before indexing
                valid_gt_ids_mask = (gt_ids_t >= 0) & (gt_ids_t < num_gt_ids)
                if not np.all(valid_gt_ids_mask):
                    print(
                        f"Frame {frame_idx}: GT IDs out of expected range [0, {num_gt_ids - 1}]. Clamping."
                    )
                    gt_ids_t = np.clip(
                        gt_ids_t, 0, num_gt_ids - 1
                    )  # Clamp to valid range

                gt_id_frame_count[gt_ids_t] += 1

            # --- Handle cases with no GT or no predictions in the frame ---
            if len(gt_ids_t) == 0:
                res["FP"] += len(pred_ids_t)  # All predictions are False Positives
                # Reset previous timestep state as no GTs were present
                prev_timestep_tracker_id[:] = np.nan
                matched_in_prev_step[:] = False
                continue  # Move to next frame
            if len(pred_ids_t) == 0:
                res["FN"] += len(gt_ids_t)  # All GTs are False Negatives
                # Reset previous timestep state as no predictions were present
                prev_timestep_tracker_id[:] = np.nan
                matched_in_prev_step[:] = False
                continue  # Move to next frame
            # --- End Handle empty cases ---

            # --- Calculate Similarity Matrix ---
            # IoU between all GT and prediction boxes in the current frame
            # Shape: (num_gt_dets_t, num_pred_dets_t)
            similarity: np.ndarray = sv.detection.utils.box_iou_batch(
                gt_dets_t.xyxy, pred_dets_t.xyxy
            )
            # --- End Similarity Calculation ---

            # --- Matching Logic (Hungarian Algorithm based on TrackEval) ---
            # Build score matrix incorporating IoU and continuity bonus

            # 1. Calculate continuity bonus matrix
            # Bonus applied if a prediction ID matches the ID tracked by the GT in the previous step
            current_timestep_pred_ids: np.ndarray = pred_ids_t[
                np.newaxis, :
            ]  # Shape (1, num_pred)
            # Get the tracker ID associated with each GT ID in the *previous* timestep
            gt_prev_timestep_ids: np.ndarray = prev_timestep_tracker_id[
                gt_ids_t[:, np.newaxis]
            ]  # Shape (num_gt, 1)
            # Boolean matrix (num_gt, num_pred): True where pred ID matches GT's previous timestep ID
            matches_prev_step: np.ndarray = (
                current_timestep_pred_ids == gt_prev_timestep_ids
            ) & (~np.isnan(gt_prev_timestep_ids))  # Ensure previous ID was valid

            # 2. Combine bonus and similarity
            # Score is high if continuity bonus applies, otherwise it's just IoU
            score_mat: np.ndarray = matches_prev_step * _CONTINUITY_BONUS + similarity

            # 3. Apply IoU threshold
            # Zero out entries where IoU is below threshold (unless continuity bonus applied)
            # Use epsilon for float comparison robustness (like TrackEval)
            score_mat[similarity < self.iou_threshold - np.finfo("float").eps] = 0.0

            # 4. Solve assignment problem using Hungarian algorithm
            # linear_sum_assignment finds the minimum cost assignment. Since we want
            # to maximize the score, we negate the score matrix.
            row_ind: np.ndarray
            col_ind: np.ndarray
            row_ind, col_ind = linear_sum_assignment(-score_mat)

            # 5. Filter matches: Only assignments with a positive score are valid
            # (Positive score means IoU >= threshold OR continuity bonus was applied)
            actual_scores: np.ndarray = score_mat[row_ind, col_ind]
            # Use epsilon for float comparison
            valid_assignment_mask: np.ndarray = (
                actual_scores > 0.0 + np.finfo("float").eps
            )
            match_rows: np.ndarray = row_ind[
                valid_assignment_mask
            ]  # Indices into gt_dets_t
            match_cols: np.ndarray = col_ind[
                valid_assignment_mask
            ]  # Indices into pred_dets_t
            # --- End Matching Logic ---

            # --- Update Metrics based on Matches ---
            num_matches: int = len(match_rows)
            res["CLR_TP"] += num_matches
            res["FN"] += len(gt_ids_t) - num_matches
            res["FP"] += len(pred_ids_t) - num_matches

            # Get the GT indices and Tracker IDs for the valid matches
            matched_gt_indices: np.ndarray = np.array([], dtype=int)
            matched_tracker_ids: np.ndarray = np.array([], dtype=int)
            if num_matches > 0:
                # MOTP sum uses the raw IoU similarity (without bonus) of matched pairs
                res["MOTP_sum"] += similarity[match_rows, match_cols].sum()

                # Get the actual GT IDs (0-based indices) and Tracker IDs involved in matches
                matched_gt_indices = gt_ids_t[match_rows]
                matched_tracker_ids = pred_ids_t[match_cols]

                # --- ID Switch Calculation ---
                # Compare current matched tracker ID with the *last known* matched ID for this GT ID
                prev_matched_tracker_ids: np.ndarray = prev_tracker_id[
                    matched_gt_indices
                ]
                # IDSW occurs if:
                # 1. The GT ID *was* matched before (prev_tracker_id is not NaN)
                # 2. The current matched tracker ID is *different* from the previous one
                is_idsw: np.ndarray = (
                    np.logical_not(np.isnan(prev_matched_tracker_ids))
                ) & (matched_tracker_ids != prev_matched_tracker_ids)
                res["IDSW"] += np.sum(is_idsw)
                # --- End ID Switch Calculation ---

                # --- Update Tracking State for Next Frame ---
                # Fragmentation: Increment count if GT ID is matched now but was *not* matched in the previous step
                was_not_matched_prev: np.ndarray = np.logical_not(
                    matched_in_prev_step[matched_gt_indices]
                )
                gt_id_frag_count[matched_gt_indices] += was_not_matched_prev

                # Update matched count for this GT ID (used for MT/ML/PT)
                gt_id_matched_count[matched_gt_indices] += 1

                # Update the *last known* tracker ID for this GT ID (used for future IDSW checks)
                prev_tracker_id[matched_gt_indices] = matched_tracker_ids
                # --- End Update Tracking State ---

            # --- Prepare State for the *Next* Iteration ---
            # Reset the state tracking for the *immediately preceding* timestep
            prev_timestep_tracker_id[:] = np.nan
            matched_in_prev_step[:] = False
            # Set the state for GT IDs that were matched *in this current timestep*
            if num_matches > 0:
                prev_timestep_tracker_id[matched_gt_indices] = matched_tracker_ids
                matched_in_prev_step[matched_gt_indices] = True
            # --- End Prepare State ---
        # --- End Frame-by-Frame Processing ---

        # --- Calculate MT/ML/PT/Frag based on accumulated counts ---
        # Filter GT IDs that actually appeared in the sequence
        valid_gt_indices_mask: np.ndarray = gt_id_frame_count > 0
        if np.any(valid_gt_indices_mask):
            # Calculate the ratio of frames a GT ID was matched vs. frames it appeared
            # Use np.divide to handle potential division by zero safely (though mask should prevent it)
            tracked_ratio: np.ndarray = np.divide(
                gt_id_matched_count[valid_gt_indices_mask],
                gt_id_frame_count[valid_gt_indices_mask],
                out=np.zeros_like(
                    gt_id_matched_count[valid_gt_indices_mask], dtype=float
                ),
                where=gt_id_frame_count[valid_gt_indices_mask] != 0,
            )

            # Mostly Tracked (MT): Ratio > threshold
            res["MT"] = np.sum(tracked_ratio > _MT_THRESHOLD)
            # Partially Tracked (PT): Ratio between thresholds
            res["PT"] = np.sum(
                (tracked_ratio >= _ML_THRESHOLD) & (tracked_ratio <= _MT_THRESHOLD)
            )
            # Mostly Lost (ML): Ratio < threshold
            res["ML"] = np.sum(tracked_ratio < _ML_THRESHOLD)

            # Fragmentation (Frag): Sum of (number of fragments - 1) for each GT track
            # gt_id_frag_count holds the number of times a track *started* being matched
            # (either first time, or after a gap)
            res["Frag"] = np.sum(
                np.maximum(0, gt_id_frag_count[valid_gt_indices_mask] - 1)
            )
        else:
            # Handle case where GT IDs existed (num_gt_ids > 0) but never appeared
            res["MT"] = 0.0
            res["PT"] = 0.0
            # All GT IDs are considered Mostly Lost if they exist but never appear
            res["ML"] = float(num_gt_ids)
            res["Frag"] = 0.0
        # --- End MT/ML/PT/Frag Calculation ---

        # --- Compute Final Derived Metrics (MOTA, MOTP, etc.) ---
        res = self._compute_final_fields(res)
        # --- End Final Metrics ---

        # Ensure final result dictionary contains exactly the expected fields as floats
        # Filter results based on self.fields and ensure float type
        final_res: Dict[str, float] = {}
        all_defined_fields = set(self.integer_fields) | set(self.float_fields)
        for k in all_defined_fields:
            # Use get with default 0.0, convert potential NaN to 0.0, ensure float
            final_res[k] = float(np.nan_to_num(res.get(k, 0.0)))

        return final_res

    def aggregate(
        self, per_sequence_results: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Aggregates CLEAR results across multiple sequences.

        Sums the base counts (TP, FP, FN, IDSW, MOTP_sum, MT, ML, PT, Frag,
        CLR_Frames) across all provided sequence results. Then, recalculates the
        final derived metrics (MOTA, MOTP, Recall, Precision, etc.) based on
        these aggregated counts.

        Args:
            per_sequence_results (List[Dict[str, float]]): A list where each
                element is a dictionary returned by `compute` for a single
                sequence. Invalid entries (non-dicts or dicts missing expected
                keys) will be skipped with a warning.

        Returns:
            Dict[str, Union[float, str]]: A dictionary containing the aggregated
            CLEAR metrics (both summed counts and recalculated derived metrics).
            Values are floats. May contain a 'message' key (str) if aggregation
            is not possible (e.g., no valid input).
        """
        if not per_sequence_results:
            return {"message": "No sequence results to aggregate for CLEAR"}

        # Initialize dictionary to store summed counts
        aggregated_counts: Dict[str, float] = {
            field: 0.0 for field in self.summed_fields
        }
        valid_sequences: int = 0

        # Iterate through sequence results and sum the base counts
        for seq_res in per_sequence_results:
            if isinstance(seq_res, dict):
                # Check if essential keys seem present (basic validation)
                if all(
                    isinstance(seq_res.get(f, 0.0), (int, float))
                    for f in self.summed_fields
                ):
                    valid_sequences += 1
                    for field in self.summed_fields:
                        # Add the value, defaulting to 0.0 if key is missing
                        aggregated_counts[field] += seq_res.get(field, 0.0)
                else:
                    print(
                        f"Skipping sequence result due to missing or non-numeric "
                        f"summed fields during CLEAR aggregation: {seq_res}"
                    )
            else:
                print(
                    f"Skipping invalid sequence result format (expected dict) "
                    f"during CLEAR aggregation: {seq_res}"
                )

        if valid_sequences == 0:
            return {"message": "No valid sequence results found for CLEAR aggregation"}

        # Recalculate final derived metrics from the aggregated counts
        final_aggregated_metrics: Dict[str, float] = self._compute_final_fields(
            aggregated_counts
        )

        # Prepare the final dictionary, ensuring all expected fields are present
        result_dict: Dict[str, Union[float, str]] = {}
        all_defined_fields = set(self.integer_fields) | set(self.float_fields)

        for field in all_defined_fields:
            if field in self.integer_fields:
                # Get summed integer count
                result_dict[field] = aggregated_counts.get(field, 0.0)
            elif field in self.float_fields:
                # Get recalculated float metric
                # Use get with default 0.0, convert potential NaN to 0.0
                result_dict[field] = float(
                    np.nan_to_num(final_aggregated_metrics.get(field, 0.0))
                )
            else:
                # Should not happen if fields lists are correct
                print(
                    f"Field '{field}' not found in integer or float field lists during aggregation."
                )

        # Ensure MOTP_sum (which is in float_fields but also summed) is correctly represented
        result_dict["MOTP_sum"] = aggregated_counts.get("MOTP_sum", 0.0)

        return result_dict

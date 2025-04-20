from typing import Any, Dict, List, Optional, Union

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.eval.metrics.base_tracking_metric import TrackingMetric


class CLEARMetric(TrackingMetric):
    """
    Calculates CLEAR metrics (MOTA, MOTP, IDSW, MT, ML, PT, Frag, etc.).

    Requires ground truth and prediction `sv.Detections` objects to have
    `tracker_id` and `data['frame_idx']` attributes populated.

    Reference:
        K. Bernardin, R. Stiefelhagen,
        "Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics", 2008.
        https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309

        Implementation details inspired by TrackEval:
        https://github.com/JonathonLuiten/TrackEval [MIT License]
    """

    @staticmethod
    def _compute_final_fields(res: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate derived CLEAR metrics (MOTA, MOTP, Recall, Precision, etc.)
        from base counts (TP, FP, FN, IDSW, etc.).

        Args:
            res: Dictionary containing the base counts accumulated over a sequence
                 or aggregated across sequences. Expected keys include 'CLR_TP',
                 'FN', 'FP', 'IDSW', 'MOTP_sum', 'MT', 'ML', 'PT', 'Frag',
                 'CLR_Frames'.

        Returns:
            The input dictionary `res` updated with the calculated derived metrics.
        """
        # Ensure necessary counts are present, default to 0.0 if missing
        required_counts = [
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

        num_gt_dets = res["CLR_TP"] + res["FN"]
        num_tracker_dets = res["CLR_TP"] + res["FP"]
        num_gt_ids = res["MT"] + res["ML"] + res["PT"]

        res["MTR"] = res["MT"] / np.maximum(1.0, num_gt_ids)
        res["MLR"] = res["ML"] / np.maximum(1.0, num_gt_ids)
        res["PTR"] = res["PT"] / np.maximum(1.0, num_gt_ids)
        res["Recall"] = res["CLR_TP"] / np.maximum(1.0, num_gt_dets)  # Recall
        res["Precision"] = res["CLR_TP"] / np.maximum(1.0, num_tracker_dets)  # Precision

        #         res['MODA'] = (res['CLR_TP'] - res['FP']) / np.maximum(1.0, res['CLR_TP'] + res['FN'])
        # res['MOTA'] = (res['CLR_TP'] - res['FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['FN'])
        res["MODA"] = (res["CLR_TP"] - res["FP"]) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )
        res["MOTA"] = (res["CLR_TP"] - res["FP"] - res["IDSW"]) / np.maximum(
            1.0, res["CLR_TP"] + res["FN"]
        )

        # res["MODA"] = (res["CLR_TP"] - res["FP"]) / np.maximum(1.0, num_gt_dets)
        # res["MOTA"] = (res["CLR_TP"] - res["FP"] - res["IDSW"]) / np.maximum(
        #     1.0, num_gt_dets
        # )
        res["MOTP"] = res["MOTP_sum"] / np.maximum(1.0, res["CLR_TP"])
        # Note: sMOTA and MOTAL are sometimes defined, but MOTA/MOTP are primary
        res["CLR_F1"] = res["CLR_TP"] / np.maximum(
            1.0, num_tracker_dets + res["FN"]
        )  # F1 = TP / (TP + 0.5*FP + 0.5*FN) = TP / (NumPred + FN)
        res["FP_per_frame"] = res["FP"] / np.maximum(1.0, res["CLR_Frames"])

        # Ensure all expected float fields exist, even if calculated as 0
        float_fields = [
            "MOTA",
            "MOTP",
            "MODA",
            "Recall",
            "Precision",
            "MTR",
            "PTR",
            "MLR",
            "CLR_F1",
            "FP_per_frame",
        ]
        for field in float_fields:
            res.setdefault(field, 0.0)

        return res

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initializes the CLEARMetric calculator.

        Args:
            iou_threshold: The Intersection over Union (IoU) threshold for
                           considering a detection match (True Positive).
                           Defaults to 0.5.
        """
        self.iou_threshold = iou_threshold
        # Define fields following trackeval naming convention where possible
        self.integer_fields = [
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
        self.float_fields = [
            "MOTA",
            "MOTP",
            "MODA",
            "Recall",
            "Precision",
            "MTR",
            "PTR",
            "MLR",
            "CLR_F1",
            "FP_per_frame",
        ]
        # MOTP_sum is summed for aggregation before final MOTP calculation
        self.summed_fields = [
            *self.integer_fields,
            "MOTP_sum",
        ]
        self.fields = self.float_fields + self.integer_fields

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

        Args:
            ground_truth: Ground truth annotations as an sv.Detections object.
                          Requires `tracker_id` attribute and `data['frame_idx']`.
            predictions: Tracker predictions as an sv.Detections object.
                         Requires `tracker_id` attribute and `data['frame_idx']`.
            sequence_info: Optional sequence metadata (unused by CLEAR).

        Returns:
            A dictionary containing computed CLEAR metric components and final
            derived metrics (MOTA, MOTP, etc.) for the sequence.
        """
        res: Dict[str, float] = {field: 0.0 for field in [*self.fields, "MOTP_sum"]}

        # --- Input Validation ---
        if "frame_idx" not in ground_truth.data:
            raise ValueError("Ground truth detections must have 'frame_idx' in data.")
        if "frame_idx" not in predictions.data:
            raise ValueError("Prediction detections must have 'frame_idx' in data.")
        if ground_truth.tracker_id is None:
            raise ValueError("Ground truth detections must have 'tracker_id'.")
        if predictions.tracker_id is None:
            raise ValueError("Prediction detections must have 'tracker_id'.")

        # --- Quick exit for empty sequences ---
        if len(ground_truth) == 0 and len(predictions) == 0:
            res["CLR_Frames"] = 0.0
            return self._compute_final_fields(res)  # Return all zeros

        gt_frame_indices = set(ground_truth.data["frame_idx"])
        pred_frame_indices = set(predictions.data["frame_idx"])
        all_frame_indices = sorted(list(gt_frame_indices.union(pred_frame_indices)))

        if not all_frame_indices:
            res["CLR_Frames"] = 0.0
            return self._compute_final_fields(res)  # Return all zeros

        res["CLR_Frames"] = float(len(all_frame_indices))

        # --- Data Structures for Tracking State ---
        # GT IDs are assumed to be 0-based and contiguous after preprocessing
        unique_gt_ids_in_input = np.unique(ground_truth.tracker_id)
        if len(unique_gt_ids_in_input) > 0:
            # Add 1 because IDs are 0-based indices
            num_gt_ids = int(np.max(unique_gt_ids_in_input)) + 1
        else:
            num_gt_ids = 0

        # Per GT ID tracking stats (using 0-based index from preprocessed input)
        # Size arrays based on the maximum possible ID + 1
        gt_id_frame_count = np.zeros(num_gt_ids, dtype=int)
        gt_id_matched_count = np.zeros(num_gt_ids, dtype=int)
        gt_id_frag_count = np.zeros(num_gt_ids, dtype=int)

        # Track last matched tracker ID for each GT ID (using 0-based index)
        prev_tracker_id = np.full(num_gt_ids, np.nan)
        prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)
        matched_in_prev_step = np.zeros(num_gt_ids, dtype=bool)

        # --- Frame-by-Frame Processing ---
        for frame_idx in all_frame_indices:
            gt_dets_t = ground_truth[ground_truth.data["frame_idx"] == frame_idx]
            pred_dets_t = predictions[predictions.data["frame_idx"] == frame_idx]

            # Use tracker_id directly as it's assumed to be 0-based index
            gt_ids_t = gt_dets_t.tracker_id
            pred_ids_t = pred_dets_t.tracker_id

            # Update frame count for present GT IDs (using direct 0-based IDs)
            if len(gt_ids_t) > 0:  # Check if gt_ids_t is not empty before indexing
                gt_id_frame_count[gt_ids_t] += 1

            # Handle cases with no detections in the frame
            if len(gt_ids_t) == 0:
                res["FP"] += len(pred_ids_t)
                # Reset previous timestep tracker IDs for GT IDs not present
                prev_timestep_tracker_id[:] = np.nan
                matched_in_prev_step[:] = False
                continue
            if len(pred_ids_t) == 0:
                res["FN"] += len(gt_ids_t)
                # Reset previous timestep tracker IDs for GT IDs not present
                prev_timestep_tracker_id[:] = np.nan
                matched_in_prev_step[:] = False
                continue

            # Calculate IoU similarity matrix
            # Shape: (num_gt_dets_t, num_pred_dets_t)
            similarity = sv.detection.utils.box_iou_batch(
                gt_dets_t.xyxy, pred_dets_t.xyxy
            )

            # --- Matching Logic (Hungarian Algorithm) ---
            # Reference:
            # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py#L107
            # Build score matrix: High score for continuity, add IoU similarity,
            # then threshold.

            # 1. Calculate continuity bonus matrix (boolean)
            continuity_bonus_val = 1000.0  # Must be larger than max similarity (1.0)
            current_timestep_pred_ids = pred_ids_t[np.newaxis, :]  # Shape (1, num_pred)
            gt_prev_timestep_ids = prev_timestep_tracker_id[
                gt_ids_t[:, np.newaxis]
            ]  # Shape (num_gt, 1)
            # Boolean matrix (num_gt, num_pred): True where pred ID matches GT's previous timestep ID
            matches_prev_step = current_timestep_pred_ids == gt_prev_timestep_ids

            # 2. Combine bonus and similarity (mirroring TrackEval structure)
            score_mat = matches_prev_step * continuity_bonus_val + similarity

            # 3. Apply threshold: Zero out entries where IoU is below threshold
            score_mat[similarity < self.iou_threshold - np.finfo("float").eps] = (
                0  # Use minus epsilon like TrackEval
            )

            # --- Check for infeasible cost matrix ---
            # (Removed the check as TrackEval doesn't seem to have an explicit one here,
            # the filtering of assignments handles it)

            # Solve assignment problem
            # Note: linear_sum_assignment finds the minimum cost assignment
            # We use the positive score_mat, so negate it for minimization.
            row_ind, col_ind = linear_sum_assignment(-score_mat)

            # Filter matches: Only assignments with a positive score are valid
            # (Positive score means either IoU >= threshold OR continuity
            # bonus was applied)
            actual_score = score_mat[row_ind, col_ind]
            valid_assignment = (
                actual_score > 0 + np.finfo("float").eps
            )  # Use epsilon for float comparison
            match_rows = row_ind[valid_assignment]
            match_cols = col_ind[valid_assignment]

            # --- Update Metrics based on Matches ---
            num_matches = len(match_rows)
            res["CLR_TP"] += num_matches
            res["FN"] += len(gt_ids_t) - num_matches
            res["FP"] += len(pred_ids_t) - num_matches

            if num_matches > 0:
                # MOTP sum uses the raw similarity score (without bonus)
                # of the matched pairs
                # Need to get the similarity for the final valid matches
                res["MOTP_sum"] += similarity[match_rows, match_cols].sum()

                matched_gt_indices = gt_ids_t[match_rows]  # 0-based indices
                matched_tracker_ids = pred_ids_t[match_cols]  # Actual tracker IDs

                # --- ID Switch Calculation ---
                # Compare current matched tracker ID with the *last known*
                #  matched ID for this GT ID
                prev_matched_tracker_ids = prev_tracker_id[matched_gt_indices]
                # IDSW occurs if:
                # 1. The GT ID *was* matched before (prev_tracker_id is not NaN)
                # 2. The current matched tracker ID is *different* from the previous one
                # if np.any(np.logical_not(np.isnan(prev_matched_tracker_ids))):
                #     print(f"Previous matched tracker IDs: {prev_matched_tracker_ids}")
                #     print(f"Current matched tracker IDs: {matched_tracker_ids}")
                #     print(f"Matched GT indices: {matched_gt_indices}")

                is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                    matched_tracker_ids != prev_matched_tracker_ids
                )
                # if np.any(is_idsw):
                #     print(f"IDSW detected for GT IDs: {matched_gt_indices[is_idsw]}")
                res["IDSW"] += np.sum(is_idsw)

                # --- Update Tracking State for Next Frame ---
                # Fragmentation: Increment if matched now but *not* in the previous step
                was_not_matched_prev = np.logical_not(
                    matched_in_prev_step[matched_gt_indices]
                )
                gt_id_frag_count[matched_gt_indices] += was_not_matched_prev

                # Update matched count for MT/ML/PT
                gt_id_matched_count[matched_gt_indices] += 1

                # Update previous tracker ID (for IDSW) regardless of timestep gap
                prev_tracker_id[matched_gt_indices] = matched_tracker_ids

            # --- Prepare `prev_timestep_tracker_id` and `matched_in_prev_step`
            # for the *next* iteration ---
            # Reset all GT IDs first
            prev_timestep_tracker_id[:] = np.nan
            matched_in_prev_step[:] = False
            # Set the tracker IDs for the GT IDs that were matched *in this timestep*
            if num_matches > 0:
                prev_timestep_tracker_id[matched_gt_indices] = matched_tracker_ids
                matched_in_prev_step[matched_gt_indices] = True

        # --- Calculate MT/ML/PT/Frag ---
        # Avoid division by zero for GT IDs that never appeared
        valid_gt_indices = gt_id_frame_count > 0
        if np.any(valid_gt_indices):
            tracked_ratio = (
                gt_id_matched_count[valid_gt_indices]
                / gt_id_frame_count[valid_gt_indices]
            )
            res["MT"] = np.sum(tracked_ratio > 0.8)
            res["PT"] = np.sum((tracked_ratio >= 0.2) & (tracked_ratio <= 0.8))
            res["ML"] = np.sum(tracked_ratio < 0.2)

            # Frag: Sum of (number of fragments - 1) for each GT track
            # A fragment starts when a track is matched after not being matched
            # gt_id_frag_count holds the number of fragments for each track
            res["Frag"] = np.sum(np.maximum(0, gt_id_frag_count[valid_gt_indices] - 1))
        else:
            # Handle case where there were GT IDs but they never appeared in frames
            res["MT"] = 0.0
            res["PT"] = 0.0
            # All GT IDs are considered Mostly Lost if they exist but never appear
            res["ML"] = float(num_gt_ids)
            res["Frag"] = 0.0

        # --- Compute Final Derived Metrics ---
        res = self._compute_final_fields(res)

        # Ensure only expected fields are returned and are floats
        final_res = {k: float(res.get(k, 0.0)) for k in self.fields if k in res}
        # Add back integer fields that might have been calculated
        for k in self.integer_fields:
            final_res[k] = float(res.get(k, 0.0))

        return final_res

    def aggregate(
        self, per_sequence_results: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Aggregates CLEAR results across multiple sequences by summing the base counts
        (TP, FP, FN, IDSW, MOTP_sum, MT, ML, PT, Frag, CLR_Frames) and then
        recalculating the final derived metrics (MOTA, MOTP, etc.) based on these
        aggregated counts.

        Args:
            per_sequence_results: A list where each element is the dictionary
                                  returned by `compute` for a single sequence.

        Returns:
            A dictionary containing the aggregated CLEAR metrics, or a message
            if aggregation is not possible.
        """
        if not per_sequence_results:
            return {"message": "No sequence results to aggregate for CLEAR"}

        # Sum integer counts and MOTP_sum across sequences
        aggregated_counts: Dict[str, float] = {
            field: 0.0 for field in self.summed_fields
        }
        valid_sequences = 0
        for seq_res in per_sequence_results:
            if isinstance(seq_res, dict):
                # Ensure all expected summed fields are present in seq_res, default to 0
                valid_sequences += 1
                for field in self.summed_fields:
                    aggregated_counts[field] += seq_res.get(field, 0.0)
            else:
                print(
                    f"Warning: Skipping invalid sequence result format during \
                        CLEAR aggregation: {seq_res}"
                )

        if valid_sequences == 0:
            return {"message": "No valid sequence results found for CLEAR aggregation"}

        # Recalculate final metrics from aggregated counts
        final_aggregated_metrics = self._compute_final_fields(aggregated_counts)

        # Ensure return type matches protocol (float or str)
        # Start with the calculated float fields (like MOTA, MOTP, Recall, etc.)
        result_dict: Dict[str, Union[float, str]] = {
            k: v
            for k, v in final_aggregated_metrics.items()
            if k in self.float_fields # Only take the final calculated float metrics
        }
        # Add back the summed integer fields as floats
        for field in self.integer_fields:
            result_dict[field] = aggregated_counts.get(field, 0.0)

        # Explicitly add the summed MOTP_sum as well
        result_dict["MOTP_sum"] = aggregated_counts.get("MOTP_sum", 0.0)

        # Ensure all expected fields are present, defaulting to 0.0 if somehow missed
        for field in self.fields:
             result_dict.setdefault(field, 0.0)
        result_dict.setdefault("MOTP_sum", 0.0)


        return result_dict

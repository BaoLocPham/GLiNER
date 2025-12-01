import warnings
from collections import defaultdict
from typing import Union, List, Literal

import numpy as np
import torch
import os

class UndefinedMetricWarning(UserWarning):
    pass


def _prf_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    metric: Literal["precision", "recall", "f-score"],
    modifier: str,
    average: str,
    warn_for: List[str],
    zero_division: Union[str, int] = "warn",
) -> np.ndarray:
    """Performs division and handles divide-by-zero with warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = 0.0 if zero_division in ["warn", 0] else 1.0

    if denominator == 0 and zero_division == "warn" and metric in warn_for:
        msg_start = f"{metric.title()}"
        if "f-score" in warn_for:
            msg_start += " and F-score" if metric in warn_for else "F-score"
        msg_start += " are" if "f-score" in warn_for else " is"
        _warn_prf(
            average=average,
            modifier=modifier,
            msg_start=msg_start,
            result_size=len(result),
        )

    return result


def _warn_prf(average: str, modifier: str, msg_start: str, result_size: int):
    axis0, axis1 = ("label", "sample") if average == "samples" else ("sample", "label")
    if result_size == 1:
        msg = f"{msg_start} ill-defined and being set to 0.0 due to no {modifier} {axis0}."  # noqa: E501
    else:
        msg = f"{msg_start} ill-defined and being set to 0.0 in {axis1}s with no {modifier} {axis0}s."  # noqa: E501
    msg += " Use `zero_division` parameter to control this behavior."
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)

# def calculate_iou(start1, end1, start2, end2):
#     """
#     Calculate the Intersection over Union (IoU) between two entities.
#     """
#     if start1 == end1:
#         if start1 == start2:
#             if end1 == end2:
#                 return 1
#             else:
#                 return 1 / abs(end1 - end2)
#         else: # 
#             return 0
#     # Calculate intersection
#     intersection_start = max(start1, start2)
#     intersection_end = min(end1, end2)
#     intersection_len = max(0, intersection_end - intersection_start)
   

#     # Calculate union
#     # calculate union by token level
#     union_len = max(end1, end2) - min(start1, start2) 

#     # Compute IoU
#     return intersection_len / union_len if union_len > 0 else 0

def span_iou(span1, span2):
    if span1 == span2:
      return 1
    s1_start, s1_end = span1
    s2_start, s2_end = span2

    # Normalize ordering
    if s1_start > s1_end:
        s1_start, s1_end = s1_end, s1_start
    if s2_start > s2_end:
        s2_start, s2_end = s2_end, s2_start

    # Check zero-length original spans
    zero1 = (s1_start == s1_end)
    zero2 = (s2_start == s2_end)

    # Convert zero-length spans to length 1
    if zero1:
        point1 = s1_start
        s1_end = s1_start + 1

    if zero2:
        point2 = s2_start
        s2_end = s2_start + 1

    # Compute intersection using standard interval logic
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    intersection = max(0, inter_end - inter_start)

    # SPECIAL INTERSECTION RULE
    if zero1 and intersection == 0:
        # does other span end exactly at point?
        if s2_start <= point1 <= s2_end:
            intersection = 1

    if zero2 and intersection == 0:
        if s1_start <= point2 <= s1_end:
            intersection = 1

    # Lengths
    len1 = s1_end - s1_start
    len2 = s2_end - s2_start

    # SPECIAL UNION RULE if any point span is involved
    if zero1 or zero2:
        union = len1 + len2
    else:
        union = len1 + len2 - intersection

    return intersection / union if union > 0 else 0.0




def extract_tp_actual_correct(y_true, y_pred, iou_threshold=0.8):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)



    for type_name, (start, end), idx in y_true:
        entities_true[type_name].add((start, end, idx))
    for type_name, (start, end), idx in y_pred:
        entities_pred[type_name].add((start, end, idx))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    # print(f"entities_true: {entities_true}")
    # print(f"entities_pred: {entities_pred}")
    # USE_EXACT_MATCHING = bool(os.getenv("USE_EXACT_MATCHING", True))
    USE_EXACT_MATCHING = int(os.getenv("USE_EXACT_MATCHING", 0))
    if USE_EXACT_MATCHING:
        print("Using exact matching")
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            
            # print(f"type_name: {type_name}")

            
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

            # print(f"tp_sum: {tp_sum}, pred_sum: {pred_sum}, true_sum: {true_sum}")
            # print("--------------------------------")
    else:
        print("Using IoU matching")

        iou_threshold = float(os.getenv("IOU_THRESHOLD", 0.8))
        print(f"EVALUATOR iou_threshold: {iou_threshold}")
        
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())

            # Calculate true positives (TP) with one-to-one matching
            # Track which predicted entities have been matched to avoid double-counting
            matched_pred_entities = set()
            tp = 0
            
            for true_entity in entities_true_type:
                true_start, true_end, _ = true_entity
                # best_iou = 0
                best_pred_entity = None
                
                # Find the best matching predicted entity that hasn't been matched yet
                for pred_entity in entities_pred_type:
                    if pred_entity in matched_pred_entities:
                        continue  # Skip already matched entities
                        
                    pred_start, pred_end, _ = pred_entity
                    
                    iou = span_iou((true_start, true_end), (pred_start, pred_end))
                    # if iou >= iou_threshold and iou > best_iou:
                    #     best_iou = iou
                    #     best_pred_entity = pred_entity

                    if iou >= iou_threshold:
                        # if iou > best_iou:
                        #     best_iou = iou
                        best_pred_entity = pred_entity
                
                # If we found a match, count it and mark the predicted entity as used
                if best_pred_entity is not None:
                    tp += 1
                    matched_pred_entities.add(best_pred_entity)

            # Calculating the total predictions and true entities
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
            tp_sum = np.append(tp_sum, tp)


    return pred_sum, tp_sum, true_sum, target_names


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([t + [i] for t in true])
        all_pred.extend([p + [i] for p in pred])

    return all_true, all_pred


def compute_prf(y_true, y_pred, average="micro"):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)

    pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric="precision",
        modifier="predicted",
        average=average,
        warn_for=["precision", "recall", "f-score"],
        zero_division="warn",
    )
    # print(f"tp_sum: {tp_sum}, pred_sum: {pred_sum}, true_sum: {true_sum}")
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric="recall",
        modifier="true",
        average=average,
        warn_for=["precision", "recall", "f-score"],
        zero_division="warn",
    )

    denominator = precision + recall
    denominator[denominator == 0.0] = 1
    f_score = 2 * (precision * recall) / denominator

    return {"precision": precision[0], "recall": recall[0], "f_score": f_score[0]}


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    def get_entities_fr(self, ents):
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def get_entities_pr(self, ents):
        all_ents = []
        for s, e, lab, _, _ in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def transform_data(self):
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_entities_fr(i)
            all_true_ent.append(e)
            e = self.get_entities_pr(j)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


def is_nested(idx1, idx2):
    # Return True if idx2 is nested inside idx1 or vice versa
    return (idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or (
        idx2[0] <= idx1[0] and idx2[1] >= idx1[1]
    )


def has_overlapping(idx1, idx2, multi_label=False):
    # Check for any overlap between two spans
    if idx1[:2] == idx2[:2]:  # Exact same boundaries can be considered as overlapping
        return not multi_label
    if idx1[0] > idx2[1] or idx2[0] > idx1[1]:
        return False
    return True


def has_overlapping_nested(idx1, idx2, multi_label=False):
    # Return True if idx1 and idx2 overlap, but neither is nested inside the other
    if idx1[:2] == idx2[:2]:  # Exact same boundaries, not considering labels here
        return not multi_label
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2):
        return False
    return True


from functools import partial


def greedy_search(spans, flat_ner=True, multi_label=False):  # start, end, class, score
    if flat_ner:
        has_ov = partial(has_overlapping, multi_label=multi_label)
    else:
        has_ov = partial(has_overlapping_nested, multi_label=multi_label)

    new_list = []
    span_prob = sorted(spans, key=lambda x: -x[-1])

    for i in range(len(spans)):
        b = span_prob[i]
        flag = False
        for new in new_list:
            if has_ov(b[:-1], new):
                flag = True
                break
        if not flag:
            new_list.append(b)

    new_list = sorted(new_list, key=lambda x: x[0])
    return new_list

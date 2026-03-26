from collections import defaultdict


def _query_identity(entry):
    if "query_key" in entry and entry["query_key"] not in (None, ""):
        return entry["query_key"]
    return entry.get("cls")


def compute_rec_metrics(all_gts, all_preds, iou_thresh, iou_fn):
    """
    Compute phrase-level REC accuracy.

    A query is identified by (img_id, query_key) when available, or by
    (img_id, cls) as a fallback. For each query we keep only the highest-
    confidence prediction and mark it correct if it overlaps any GT box for
    that same query at or above the requested IoU threshold.
    """
    gts_by_query = defaultdict(list)
    preds_by_query = defaultdict(list)

    for gt in all_gts:
        query_id = _query_identity(gt)
        gts_by_query[(gt["img_id"], query_id)].append(gt["box"])

    for pred in all_preds:
        query_id = _query_identity(pred)
        preds_by_query[(pred["img_id"], query_id)].append(pred)

    total_queries = len(gts_by_query)
    matched_queries = 0
    missing_predictions = 0

    for query_key, gt_boxes in gts_by_query.items():
        query_preds = preds_by_query.get(query_key, [])
        if not query_preds:
            missing_predictions += 1
            continue

        top_pred = max(query_preds, key=lambda pred: float(pred.get("conf", 0.0)))
        best_iou = max(iou_fn(top_pred["box"], gt_box) for gt_box in gt_boxes)
        if best_iou >= iou_thresh:
            matched_queries += 1

    accuracy = matched_queries / total_queries if total_queries > 0 else 0.0
    return {
        "rec": float(accuracy),
        "rec_total_queries": int(total_queries),
        "rec_matched_queries": int(matched_queries),
        "rec_missing_predictions": int(missing_predictions),
    }

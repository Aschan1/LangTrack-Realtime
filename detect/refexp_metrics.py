from collections import defaultdict
import json, argparse, statistics as stats #Importing statistics to calculate mean and standard deviation of the metrics
FRACTIONS = [0.25, 0.5, 0.75, 1.0] #Fractions for accuracy snapshots

#Functtion to check if the guess is correct or not, by comparing it with the ground truth
def _correct(guess, gt):
    return guess is not None and str(guess).strip().lower() == str(gt).strip().lower()

#Function to normalize the value, by converting it to string and stripping it of whitespace and converting it to lowercase. If the value is in the list of invalid values, return None
def _norm(v):
    s = str(v).strip().lower()
    return None if s in ("none", "null", "", "[]", "unknown") else s

#Function to get the identity of the query, by checking if the query_key is present in the entry and is not None or empty, return it. Otherwise, return the class of the entry
def _query_identity(entry):
    if "query_key" in entry and entry["query_key"] not in (None, ""):
        return entry["query_key"]
    return entry.get("cls")

#Function to calculate the anytime accuracy of the records, by checking the correctness of the snapshots at different fractions and dividing it by the total number of records
def anytime_accuracy(records, fractions=("0.25", "0.5", "0.75", "1.0")):
    return {f: sum(_correct(r["snapshots"].get(f), r["gt"]) for r in records) / len(records)
            for f in fractions}

#Function to calculate the time to lock for the records, by checking the steps_locked of each record and finding the first step where all subsequent guesses are correct and stable to the end, and calculating the offset from the end time of the recordge
def time_to_lock(records):
    offsets = []
    for r in records:
        seq = r["steps_locked"]#streaming_eval.json from driver output, containing the time and guess for each step
        for j, (t, g) in enumerate(seq):
            if all(_correct(g2, r["gt"]) for _, g2 in seq[j:]):   # correct and stable to the end
                offsets.append(t - r["t_end"])                    # negative = locked early
                break
    return offsets

#Function to calculate the flip count for the records, by checking the steps_locked of each record and counting the number of times the guess changes from one step to the next
def flip_count(records, key="steps_locked"):
    counts = []
    for r in records:
        s = [g for _, g in r[key] if g is not None]
        counts.append(sum(1 for a, b in zip(s, s[1:]) if a != b))
    return counts

# --------------------------------------------------------------- offline driver
#Function to load the utterances from the dataset, by checking if the data is a dictionary or a list, and extracting the phrase and target from each entry, and returning a list of tuples of phrase and target
def load_utterances(path, phrase_key, target_key, limit=None):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("annotations", "data", "entries", "queries"):
            if isinstance(data.get(k), list):
                data = data[k]; break
        else:
            data = list(data.values())
    items = []
    for e in data:
        if isinstance(e, dict) and e.get(phrase_key) and e.get(target_key):
            items.append((str(e[phrase_key]), str(e[target_key])))
        if limit and len(items) >= limit:
            break
    return items

#Function to drive the utterance, by splitting the phrase into words and simulating the streaming parsing of the words, and updating the prompt processor with the committed text, and returning a dictionary of the ground truth, end time, steps_locked, steps_raw, snapshots, and phrase
def _drive_utterance(prompt_processor, phrase, gt, sec_per_word):
    try:
        from detect.demo import StreamingParser, WAIT_K
    except ImportError:
        from demo import StreamingParser, WAIT_K
    words = phrase.split(); n = len(words)
    if n == 0:
        return None
    parser = StreamingParser()
    steps_locked, steps_raw, last_raw = [], [], None
    for i in range(1, n + 1):
        t = round(i * sec_per_word, 3)
        committed = parser.stabilize(words[:i])
        if len(committed) >= WAIT_K:
            cs = " ".join(committed)
            if cs != parser.last_parsed_prefix:                 # parse only on new committed text
                parser.last_parsed_prefix = cs
                res = prompt_processor(speech=cs)
                last_raw = _norm(getattr(res, "target", None))  # frame-level guess (no locking)
                parser.update_locks(res)
        steps_raw.append((t, last_raw))
        steps_locked.append((t, _norm(parser.locked["target"])))  # track-level (locked)
    snapshots = {str(f): steps_locked[min(max(1, round(f * n)), n) - 1][1] for f in FRACTIONS}
    return {"gt": _norm(gt), "t_end": round(n * sec_per_word, 3),
            "steps_locked": steps_locked, "steps_raw": steps_raw,
            "snapshots": snapshots, "phrase": phrase}

#Function to summarize the streaming records, by calculating the anytime accuracy, time to lock, and flip count, and printing the results
def summarize_streaming(records):
    if not records:
        print("No records."); return
    aa = anytime_accuracy(records, fractions=[str(f) for f in FRACTIONS])
    ttl = time_to_lock(records)
    fl, fr = flip_count(records, "steps_locked"), flip_count(records, "steps_raw")
    print(f"Utterances: {len(records)}")
    print("Anytime accuracy:", {k: round(v, 3) for k, v in aa.items()})
    print(f"Time-to-lock(s): mean={stats.mean(ttl):.2f} early_frac="
          f"{sum(o < 0 for o in ttl) / len(ttl):.2f} n={len(ttl)}/{len(records)}" if ttl
          else "Time-to-lock: nothing locked correctly.")
    print(f"Flips: locked={stats.mean(fl):.2f}  raw={stats.mean(fr):.2f}")

def run_streaming_eval(dataset, out, phrase_key="phrase", target_key="target",
                       limit=None, sec_per_word=0.3):
    try:
        from detect.demo import build_prompt_processor
    except ImportError:
        from demo import build_prompt_processor
    pp = build_prompt_processor()
    records = []
    with open(out, "w") as f:
        for phrase, gt in load_utterances(dataset, phrase_key, target_key, limit):
            rec = _drive_utterance(pp, phrase, gt, sec_per_word)
            if rec:
                f.write(json.dumps(rec) + "\n"); records.append(rec)
    summarize_streaming(records)
    return records



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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--out", default="streaming_eval.jsonl")
    p.add_argument("--phrase_key", default="phrase")
    p.add_argument("--target_key", default="target")
    p.add_argument("--limit", type=int)
    p.add_argument("--sec_per_word", type=float, default=0.3)
    p.add_argument("--compute_only", action="store_true")
    a = p.parse_args()
    if a.compute_only:
        with open(a.out) as f:
            summarize_streaming([json.loads(l) for l in f if l.strip()])
    else:
        run_streaming_eval(a.dataset, a.out, a.phrase_key, a.target_key, a.limit, a.sec_per_word)

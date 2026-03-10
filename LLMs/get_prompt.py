import dspy
import os
import json
import re
from typing import List, Optional
from dspy.teleprompt import COPRO

def _normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def _listify(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    # Compatible with models occasionally outputting as strings
    if isinstance(x, str):
        # Try splitting by comma/semicolon
        parts = re.split(r"[;,，；]\s*", x.strip())
        return [p for p in parts if p]
    return [str(x).strip()] if str(x).strip() else []

# ==========================================
# 1. Update the Judge's Signature (adapt to new multi-field structure)
# ==========================================
class AssessExtraction(dspy.Signature):
    phrase: str = dspy.InputField(desc="The original descriptive phrase")
    extracted_info: str = dspy.InputField(desc="A string representation of the extracted fields (target, attributes, etc.)")
    assessment_question: str = dspy.InputField(desc="Question to assess the quality of the extraction based on specific criteria.")
    assessment_answer: bool = dspy.OutputField(desc="Output True if extraction meets the criteria specified, otherwise False.")

# ==========================================
# 2. Update the Scoring Function Metric
# ==========================================
def structured_metric(example, pred, trace=None):
    phrase = example.phrase

    # Extract predicted fields (your Predict(PromptProcessor) will provide these)
    target = _normalize(getattr(pred, "target", None))
    attributes = _listify(getattr(pred, "attributes", None))
    anchor_object = _normalize(getattr(pred, "anchor_object", None))
    spatial_relation = _normalize(getattr(pred, "spatial_relation", None))

    # ========= First Level: Hard Rules =========
    # 1) target must exist
    if not target:
        return 0.0

    # 2) attributes must be a list (or convertible to list), and should not be too long (avoid random piling)
    if not isinstance(attributes, list):
        return 0.0
    if len(attributes) > 8:
        return 0.0

    # 3) anchor_object and spatial_relation must be paired: either both None or both not None
    if (anchor_object is None) ^ (spatial_relation is None):
        return 0.0

    # 4) Target should appear in the original sentence as much as possible (relaxed: if it's synonymous rewriting, leave it to the subsequent LLM judge)
    #    No hard kill here, just give base score to avoid synonyms causing all 0
    base_score = 1.0

    # 5) attributes should not contain obvious spatial prepositions/location phrases (these should go into spatial_relation)
    forbidden_attr = {
        "left", "right", "top", "bottom", "above", "below", "under", "over",
        "on", "in", "inside", "outside", "next to", "beside", "near", "against",
        "front", "behind", "between", "at"
    }
    attr_lower = [a.lower() for a in attributes]
    if any(a in forbidden_attr for a in attr_lower):
        return 0.0

    # ========= Second Level: LLM Judge Semantic Check =========
    # Organize predictions into "readable JSON string" for judge to determine
    pred_dict = {
        "target": target,
        "attributes": attributes,
        "anchor_object": anchor_object,
        "spatial_relation": spatial_relation
    }
    pred_str = json.dumps(pred_dict, ensure_ascii=False)

    # Judge questions (you can add/subtract based on dataset characteristics)
    q_target = (
        f"Given the phrase '{phrase}', is '{target}' the most specific, tangible, visually detectable main object "
        f"(a concrete noun), and not an attribute, action, or abstract concept?"
    )
    q_attr = (
        f"Do '{attributes}' ONLY contain descriptive properties of the '{target}' (e.g., color, material, pattern, modifiers), "
        f"and avoid adding new objects, spatial relations, or hallucinated details not supported by the phrase '{phrase}'?"
    )
    q_anchor = (
        f"If '{anchor_object}' and '{spatial_relation}' are provided, do they correctly identify a nearby reference object and a valid spatial relation "
        f"that is explicitly supported by the phrase '{phrase}'? If not supported, they should be null."
    )
    q_consistency = (
        f"Is the overall structured output internally consistent (e.g., relation fits anchor; attributes modify target), "
        f"and does it preserve the original meaning of '{phrase}' without adding unsupported content?"
    )

    with dspy.context(lm=judge_lm):
        judge_target = dspy.Predict(AssessExtraction)(
            phrase=phrase, extracted_info=pred_str, assessment_question=q_target
        )
        judge_attr = dspy.Predict(AssessExtraction)(
            phrase=phrase, extracted_info=pred_str, assessment_question=q_attr
        )
        judge_anchor = dspy.Predict(AssessExtraction)(
            phrase=phrase, extracted_info=pred_str, assessment_question=q_anchor
        )
        judge_consistency = dspy.Predict(AssessExtraction)(
            phrase=phrase, extracted_info=pred_str, assessment_question=q_consistency
        )

    total_score = (
        base_score
        + int(judge_target.assessment_answer)
        + int(judge_attr.assessment_answer)
        + int(judge_anchor.assessment_answer)
        + int(judge_consistency.assessment_answer)
    )

    # Return bool in trace mode (maintain your original logic style)
    if trace is not None:
        return total_score >= 5.0

    # Normalize to 0~1
    return total_score / 5.0


# ==========================================
# 3. Fixed dec -> desc in PromptProcessor
# ==========================================
class PromptProcessor(dspy.Signature):
    phrase: str = dspy.InputField(desc="The original descriptive phrase.")
    target: str = dspy.OutputField(desc="The main object category to be detected, which should be the most specific and tangible noun in the phrase.")
    attributes: List[str] = dspy.OutputField(desc="List of descriptive properties of the target.")
    anchor_object: Optional[str] = dspy.OutputField(desc="A nearby reference object used for localization.")
    spatial_relation: Optional[str] = dspy.OutputField(desc="Spatial relationship between target and anchor.")

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY", None)
    Training_set = [
        dspy.Example(phrase="blue curtains with sailboats on them.", target="curtains", attributes=["blue", "sailboats"], anchor_object=None, spatial_relation=None).with_inputs('phrase'),
        dspy.Example(phrase="a white pillow is on the couch.", target="pillow", attributes=["white"], anchor_object="couch", spatial_relation="on").with_inputs('phrase'),
        dspy.Example(phrase="a teddy bear lies against a pillow.", target="teddy bear", attributes=[], anchor_object="pillow", spatial_relation="against").with_inputs('phrase'),
        dspy.Example(phrase="the decorative piece on the right side of the table", target="decorative piece", attributes=[], anchor_object="table", spatial_relation="on the right side").with_inputs('phrase')
    ]

    # Configure the backbone global model
    main_lm = dspy.LM(
        model="openai/qwen3.5", 
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=main_lm)

    # Define a stronger judge model
    judge_lm = dspy.LM("openai/gpt-4o")

    # Initialize COPRO optimizer (note that metric here is changed to extraction_metric)
    teleprompter = COPRO(
        metric=structured_metric, 
        prompt_model=judge_lm, 
        breadth=3,             
        depth=2               
    )

    # Run compilation
    optimized_program = teleprompter.compile(
        dspy.Predict(PromptProcessor), 
        trainset=Training_set, 
        eval_kwargs={"num_threads": 2, "display_progress": True}
    )

    optimized_program.save("./LLMs/optimized_prompt.json")
    print("✅ 优化完成并已保存到 optimized_prompt.json")

    # Test output
    gen_json = optimized_program
    print(gen_json(phrase="A metal framed table with a glass top"))
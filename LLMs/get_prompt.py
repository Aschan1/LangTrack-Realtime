import dspy
import os
import json
import re
from typing import List, Optional
from dspy.teleprompt import COPRO, BootstrapFewShotWithRandomSearch

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
# 1. Update the Judge's Signature 
# ==========================================
class AssessExtraction(dspy.Signature):
    speech: str = dspy.InputField(desc="The original raw speech text")
    extracted_info: str = dspy.InputField(desc="A string representation of the extracted fields (phrase, target, attributes, etc.)")
    assessment_question: str = dspy.InputField(desc="Question to assess the quality of the extraction based on specific criteria.")
    assessment_answer: bool = dspy.OutputField(desc="Output True if extraction meets the criteria specified, otherwise False.")

class PromptProcessor(dspy.Signature):
    speech: str = dspy.InputField(desc="The original speech text.")
    phrase: str = dspy.OutputField(desc="The converted descriptive phrase.")
    target: str = dspy.OutputField(desc="The main object category to be detected, which should be the most specific and tangible noun in the phrase.")
    attributes: List[str] = dspy.OutputField(desc="List of descriptive properties of the target.")
    anchor_object: Optional[str] = dspy.OutputField(desc="A nearby reference object used for localization.")
    spatial_relation: Optional[str] = dspy.OutputField(desc="Spatial relationship between target and anchor.")
    is_valid_command: bool = dspy.OutputField(desc="True ONLY if the speech clearly mentions a tangible physical object to find or look at. False if it is nonsense, abstract, or just background chatter.")

# ==========================================
# 2. Update the Scoring Function Metric
# ==========================================
def structured_metric(example, pred, trace=None):
    speech = example.speech

    # Extract predicted fields
    predicted_phrase = _normalize(getattr(pred, "phrase", None))
    target = _normalize(getattr(pred, "target", None))
    attributes = _listify(getattr(pred, "attributes", None))
    anchor_object = _normalize(getattr(pred, "anchor_object", None))
    spatial_relation = _normalize(getattr(pred, "spatial_relation", None))

    # ========= First Level: Hard Rules =========
    # 1) Target and phrase must exist
    if not target or not predicted_phrase:
        return 0.0
    
    fillers = ["uh", "um", "please", "can you", "find", "i see", "look for", "grab"]
    phrase_lower = predicted_phrase.lower()
    if any(filler in phrase_lower for filler in fillers):
        return 0.0
    
    # 2) Attributes must be a list, and should not be too long (avoid random piling)
    if not isinstance(attributes, list):
        return 0.0
    if len(attributes) > 8:
        return 0.0

    # 3) Anchor_object and spatial_relation must be paired
    if (anchor_object is None) ^ (spatial_relation is None):
        return 0.0

    # Base score for passing hard rules
    base_score = 1.0

    # 4) Attributes should not contain obvious spatial prepositions
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
        "phrase": predicted_phrase,
        "target": target,
        "attributes": attributes,
        "anchor_object": anchor_object,
        "spatial_relation": spatial_relation
    }
    pred_str = json.dumps(pred_dict, ensure_ascii=False)

    # Judge questions mapping back to the raw `speech`
    q_phrase_grammar = (
        f"Does the phrase '{predicted_phrase}' read as a well-formed, concise NOUN PHRASE "
        f"(e.g., 'a red apple on the table') NOT a full conversational or imperative sentence "
        f"(e.g., 'I see a red apple on the table')?"
    )
    q_target = (
        f"Given the original speech '{speech}', is '{target}' the most specific, tangible, visually detectable main object "
        f"(a concrete noun), and not an attribute, action, or abstract concept?"
    )
    q_attr = (
        f"Do '{attributes}' ONLY contain descriptive properties of the '{target}' (e.g., color, material, pattern, modifiers), "
        f"and avoid adding new objects, spatial relations, or hallucinated details not supported by the speech '{speech}'?"
    )
    q_anchor = (
        f"If '{anchor_object}' and '{spatial_relation}' are provided, do they correctly identify a nearby reference object and a valid spatial relation "
        f"that is explicitly supported by the speech '{speech}'? If not supported, they should be null."
    )
    q_consistency = (
        f"Is the overall structured output internally consistent, and does the converted phrase '{predicted_phrase}' "
        f"accurately clean up the original speech '{speech}' without losing key details or adding unsupported content?"
    )

    with dspy.context(lm=judge_lm):
        judge_phrase_grammar = dspy.Predict(AssessExtraction)(speech=speech, extracted_info=pred_str, assessment_question=q_phrase_grammar)
        judge_target = dspy.Predict(AssessExtraction)(speech=speech, extracted_info=pred_str, assessment_question=q_target)
        judge_attr = dspy.Predict(AssessExtraction)(speech=speech, extracted_info=pred_str, assessment_question=q_attr)
        judge_anchor = dspy.Predict(AssessExtraction)(speech=speech, extracted_info=pred_str, assessment_question=q_anchor)
        judge_consistency = dspy.Predict(AssessExtraction)(speech=speech, extracted_info=pred_str, assessment_question=q_consistency)

    # Safely evaluate boolean outputs (models sometimes return "True" as a string)
    def parse_bool(val) -> int:
        return 1 if str(val).strip().lower() == 'true' else 0

    total_score = (
        base_score
        + parse_bool(judge_phrase_grammar.assessment_answer)
        + parse_bool(judge_target.assessment_answer)
        + parse_bool(judge_attr.assessment_answer)
        + parse_bool(judge_anchor.assessment_answer)
        + parse_bool(judge_consistency.assessment_answer)
    )

    # Return bool in trace mode
    if trace is not None:
        return total_score >= 5.0

    # Normalize to 0~1
    return total_score / 6.0

# ==========================================
# 3. Main Processor and Execution
# ==========================================

if __name__ == "__main__":
    #Export your OPENAI_API_KEY by using `export OPENAI_API_KEY=your_api_key` in terminal before running this script.
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if not api_key:
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables. Please set it before running the optimization.")
    else:
        print("OPENAI_API_KEY found in environment variables.")
        
    Training_set = [
        dspy.Example(
            speech="Uh, can you find the blue curtains that have sailboats on them?",
            phrase="blue curtains with sailboats on them", 
            target="curtains", 
            attributes=["blue", "sailboats"], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=True
        ).with_inputs('speech'),
        
        dspy.Example(
            speech="Over there, a white pillow is on the couch.",
            phrase="a white pillow on the couch", 
            target="pillow", 
            attributes=["white"], 
            anchor_object="couch", 
            spatial_relation="on",
            is_valid_command=True
        ).with_inputs('speech'),
        
        dspy.Example(
            speech="I see a teddy bear that lies against a pillow.",
            phrase="a teddy bear lies against a pillow", 
            target="teddy bear", 
            attributes=[], 
            anchor_object="pillow", 
            spatial_relation="against",
            is_valid_command=True
        ).with_inputs('speech'),
        
        dspy.Example(
            speech="Please grab the decorative piece... it's sitting on the right side of the table.",
            phrase="the decorative piece on the right side of the table", 
            target="decorative piece", 
            attributes=[], 
            anchor_object="table", 
            spatial_relation="on the right side",
            is_valid_command=True
        ).with_inputs('speech'),

        dspy.Example(
            speech="Yeah so the the the to Bing addressed by the Boss...",
            phrase=None, 
            target="None", 
            attributes=[], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=False
        ).with_inputs('speech')
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
    # (Note: make sure OPENAI_API_KEY is properly loaded if using GPT-4o)
    judge_lm = dspy.LM("openai/gpt-4o")

    # Initialize COPRO optimizer
    # teleprompter = COPRO(
    #     metric=structured_metric, 
    #     prompt_model=judge_lm, 
    #     breadth=3,             
    #     depth=2               
    # )

    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=structured_metric,
        max_bootstrapped_demos=3, # 每个 Prompt 里最多给 3 个满分例子
        num_candidate_programs=5  # 尝试 5 次不同的题目组合
    )

    # Run compilation
    optimized_program = teleprompter.compile(
        dspy.Predict(PromptProcessor), 
        trainset=Training_set, 
    )

    # Save and test
    os.makedirs("./LLMs", exist_ok=True)
    optimized_program.save("./LLMs/optimized_new_prompt.json")
    print("✅ Optimization complete and saved to optimized_new_prompt.json")

    # Test output
    print(optimized_program(speech="Please find... a metal framed table with a glass top"))
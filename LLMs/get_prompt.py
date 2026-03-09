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
    # 兼容模型偶尔输出成字符串
    if isinstance(x, str):
        # 尝试按逗号/分号切分
        parts = re.split(r"[;,，；]\s*", x.strip())
        return [p for p in parts if p]
    return [str(x).strip()] if str(x).strip() else []

# ==========================================
# 1. 更新裁判的 Signature (适配新的多字段结构)
# ==========================================
class AssessExtraction(dspy.Signature):
    phrase: str = dspy.InputField(desc="The original descriptive phrase")
    extracted_info: str = dspy.InputField(desc="A string representation of the extracted fields (target, attributes, etc.)")
    assessment_question: str = dspy.InputField(desc="Question to assess the quality of the extraction based on specific criteria.")
    assessment_answer: bool = dspy.OutputField(desc="Output True if extraction meets the criteria specified, otherwise False.")

# ==========================================
# 2. 更新评分函数 Metric
# ==========================================
def structured_metric(example, pred, trace=None):
    phrase = example.phrase

    # 取出预测字段（你的 Predict(PromptProcessor) 会给这些）
    target = _normalize(getattr(pred, "target", None))
    attributes = _listify(getattr(pred, "attributes", None))
    anchor_object = _normalize(getattr(pred, "anchor_object", None))
    spatial_relation = _normalize(getattr(pred, "spatial_relation", None))

    # ========= 第一关：硬性规则 =========
    # 1) target 必须存在
    if not target:
        return 0.0

    # 2) attributes 必须是 list（或可转成 list），且不应过长（避免胡乱堆砌）
    if not isinstance(attributes, list):
        return 0.0
    if len(attributes) > 8:
        return 0.0

    # 3) anchor_object 与 spatial_relation 必须成对：要么都 None，要么都非 None
    if (anchor_object is None) ^ (spatial_relation is None):
        return 0.0

    # 4) 目标尽量出现在原句中（放宽：如果是同义改写，交给后面的 LLM 裁判）
    #    这里不做硬杀，只给基础分，避免同义词导致全 0
    base_score = 1.0

    # 5) attributes 不应包含明显空间介词/位置短语（这些应进 spatial_relation）
    forbidden_attr = {
        "left", "right", "top", "bottom", "above", "below", "under", "over",
        "on", "in", "inside", "outside", "next to", "beside", "near", "against",
        "front", "behind", "between", "at"
    }
    attr_lower = [a.lower() for a in attributes]
    if any(a in forbidden_attr for a in attr_lower):
        return 0.0

    # ========= 第二关：LLM 裁判语义检查 =========
    # 把预测组织成“可读 JSON 字符串”让裁判判断
    pred_dict = {
        "target": target,
        "attributes": attributes,
        "anchor_object": anchor_object,
        "spatial_relation": spatial_relation
    }
    pred_str = json.dumps(pred_dict, ensure_ascii=False)

    # 裁判问题（你可按数据集特点继续加/减）
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

    # trace 模式下返回 bool（保持你原先的逻辑风格）
    if trace is not None:
        return total_score >= 5.0

    # 归一化到 0~1
    return total_score / 5.0


# ==========================================
# 3. 修复了 dec -> desc 的 PromptProcessor
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

    # 配置主干全局模型
    main_lm = dspy.LM(
        model="openai/qwen3.5", 
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=main_lm)

    # 定义更强的裁判模型
    judge_lm = dspy.LM("openai/gpt-4o")

    # 初始化 COPRO 优化器 (注意这里的 metric 换成了 extraction_metric)
    teleprompter = COPRO(
        metric=structured_metric, 
        prompt_model=judge_lm, 
        breadth=3,             
        depth=2               
    )

    # 运行编译
    optimized_program = teleprompter.compile(
        dspy.Predict(PromptProcessor), 
        trainset=Training_set, 
        eval_kwargs={"num_threads": 2, "display_progress": True}
    )

    optimized_program.save("./LLMs/optimized_prompt.json")
    print("✅ 优化完成并已保存到 optimized_prompt.json")

    # 测试输出
    gen_json = optimized_program
    print(gen_json(phrase="A metal framed table with a glass top"))
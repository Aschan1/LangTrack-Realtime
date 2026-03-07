import dspy
from typing import List
# 1. 导入 COPRO 优化器，替换掉原来的 Bootstrap
from dspy.teleprompt import COPRO

class AssessKeywords(dspy.Signature):
    phrase: str = dspy.InputField(desc="phrase to be analyzed")
    extracted_keywords: str = dspy.InputField(desc="keywords list")
    assessment_question: str = dspy.InputField(desc="Qestion to assess the quality of the extracted keywords based on specific criteria.")
    assessment_answer: bool = dspy.OutputField(desc="Output True if the extracted keywords meet the criteria specified in the assessment question, otherwise False.")

def keyword_metric(example, pred, trace=None):
    phrase = example.phrase
    predicted_keywords = pred.keywords

    # === 第一关：硬性规则检查 ===
    if not isinstance(predicted_keywords, list) or len(predicted_keywords) < 2:
        return 0.0

    pred_lower = [str(k).lower() for k in predicted_keywords]
    if phrase.lower() not in pred_lower:
        return 0.0
    
    base_score = 1.0 

    # === 第二关：LLM 裁判检查 ===
    kw_str = str(predicted_keywords)

    q_core = f"These keywords should extract the core tangible objects from '{phrase}'. Do the keywords strictly represent physical objects, without inventing unrelated items?"
    q_taxonomy = "Does the list conclude with a broad, standard object detection category and strictly avoid abstract concepts like 'plaything' or 'indoor'?"
    q_combinations = f"If the phrase '{phrase}' contains multiple adjectives modifying a noun, does the list systematically include combinations of each individual adjective with the noun (e.g., 'adj1 noun', 'adj2 noun')? If the phrase does NOT have multiple adjectives, answer True."

    # 使用 context 让打分过程安全地在多线程中调用 judge_lm
    with dspy.context(lm=judge_lm):
        judge_core = dspy.Predict(AssessKeywords)(phrase=phrase, extracted_keywords=kw_str, assessment_question=q_core)
        judge_taxonomy = dspy.Predict(AssessKeywords)(phrase=phrase, extracted_keywords=kw_str, assessment_question=q_taxonomy)
        judge_comb = dspy.Predict(AssessKeywords)(phrase=phrase, extracted_keywords=kw_str, assessment_question=q_combinations)

    total_score = base_score + int(judge_core.assessment_answer) + int(judge_taxonomy.assessment_answer) + int(judge_comb.assessment_answer)

    # COPRO 主要是看浮点数平均分，这里保留原有的逻辑即可
    if trace is not None: 
        return total_score >= 3.0
    
    return total_score / 4.0


class PromptProcessor(dspy.Signature):
    phrase: str = dspy.InputField(desc="The descriptive phrase of the target object.")
    keywords: List[str] = dspy.OutputField(desc="A hierarchical list of keywords and each keyword should be an object, from specific to general.")

if __name__ == "__main__":

    Training_set = [
        dspy.Example(phrase="Blue curtains with sailboats on them.", keywords=['blue curtains with sailboats on them', 'blue curtains', 'sailboats', 'curtains', 'Furnishings']).with_inputs('phrase'),
        dspy.Example(phrase="a white pillow is on the couch.", keywords=['a white pillow is on the couch', 'white pillow', 'pillow', 'couch', 'bedding']).with_inputs('phrase'),
        dspy.Example(phrase="a teddy bear lies against a pillow.", keywords=['a teddy bear lies against a pillow', 'teddy bear', 'pillow', 'toy']).with_inputs('phrase'),
        dspy.Example(phrase="The decorative piece on the right side of the table", keywords=['The decorative piece on the right side of the table', 'decorative piece', 'table']).with_inputs('phrase')
    ]
    
    # 2. 配置主干全局模型（干活的小模型）
    main_lm = dspy.LM(
        model="openai/qwen3.5", 
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=main_lm)

    # 1. 定义更强的大模型（27B），它既当裁判，又当“提示词编写者”
    judge_lm = dspy.LM(
        model="openai/qwen3.5_27", 
        api_base="http://127.0.0.1:8082/v1",
        api_key="unused",
        cache=False, 
    )
    # 3. 初始化 COPRO 优化器
    # 巧妙之处：我们把 prompt_model 显式设置为强大的 judge_lm
    # 这样就可以用 27B 的聪明模型来思考和编写新 Prompt，用小模型来跑分测试
    teleprompter = COPRO(
        metric=keyword_metric, 
        prompt_model=judge_lm, # 专职写 Prompt 的模型
        breadth=4,             # 每次尝试生成 5 种不同的 Prompt 版本（如果觉得慢可以改小点）
        depth=10                # 迭代优化 3 轮
    )

    # 4. 运行编译
    # COPRO 的多线程设置放在了 eval_kwargs 里
    optimized_program = teleprompter.compile(
        dspy.Predict(PromptProcessor), 
        trainset=Training_set, 
        eval_kwargs={"num_threads": 2, "display_progress": True}
    )

    # 5. 保存结果
    optimized_program.save('./LLMs/optimized_prompt.json')
import json
import ast
import time
import dspy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from LLMs.get_prompt import PromptProcessor


def safe_parse_keywords(keywords_output, original_phrase):
    """安全解析 LLM 输出的关键词，确保返回的是一个真正的 List 格式"""
    if isinstance(keywords_output, str):
        try:
            parsed = ast.literal_eval(keywords_output)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    if isinstance(keywords_output, list):
        return keywords_output

    if keywords_output:
        return [str(keywords_output)]

    return [original_phrase]


def preprocess_json_with_llm(input_json_path, output_json_path, max_workers=32, retries=1):
    print("Initializing DSPy and LLM...")

    # 配置你的本地 LLM
    # 结构化输出建议 cache=True（尤其你开发/重复跑时收益巨大）
    main_lm = dspy.LM(
        model="openai/qwen3.5",
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=True,  # 原来是 False，这里建议打开
    )
    dspy.configure(lm=main_lm)

    # 初始化并加载之前优化好的 Prompt
    prompt_processor = dspy.Predict(PromptProcessor)
    try:
        prompt_processor.load("./LLMs/optimized_prompt.json")
        print("✅ Loaded optimized_prompt.json successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load optimized prompt. Using default. Error: {e}")

    # 1. 加载原始 JSON 数据
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 提取所有唯一的 phrase (去重)
    unique_phrases = sorted(set(item["phrase"] for item in data))
    print(f"Found {len(data)} total entries, with {len(unique_phrases)} unique phrases.")

    phrase_to_keywords = {}
    print(f"🚀 Starting LLM knowledge extraction with concurrency: max_workers={max_workers} ...")

    def process_one_phrase(phrase: str):
        last_err = None
        for attempt in range(retries + 1):
            try:
                llm_result = prompt_processor(phrase=phrase)
                raw_keywords = llm_result.keywords
                clean_keywords = safe_parse_keywords(raw_keywords, phrase)
                return phrase, clean_keywords, None
            except Exception as e:
                last_err = e
                # 简单退避，避免瞬时拥塞
                time.sleep(0.05 * (attempt + 1))

        # 失败兜底
        return phrase, [phrase], last_err

    # 3. 并发跑 unique_phrases
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_phrase, p) for p in unique_phrases]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing unique phrases"):
            phrase, keywords, err = fut.result()
            if err is not None:
                # 不要在并发中打印太多，影响性能；这里只做简短提示
                # 你也可以把失败列表收集起来最后统一处理
                pass
            phrase_to_keywords[phrase] = keywords

    # 4. 写回原数据
    print("Writing keywords back to original dataset...")
    for item in data:
        item["keywords"] = phrase_to_keywords.get(item["phrase"], [item["phrase"]])

    # 5. 保存
    print(f"Saving enriched data to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("🎉 Preprocessing finished completely!")


if __name__ == "__main__":
    INPUT_JSON = "yolo_dataset/home_ovd_filtered.json"
    OUTPUT_JSON = "yolo_dataset/home_ovd_filtered_with_keywords.json"

    # 你 server 端 parallel=48
    # 建议 max_workers 从 32 开始试，再试 48；如果 48 反而更慢，再回落到 32/24
    preprocess_json_with_llm(INPUT_JSON, OUTPUT_JSON, max_workers=32, retries=1)

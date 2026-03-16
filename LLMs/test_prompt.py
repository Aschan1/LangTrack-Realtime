import dspy
from get_prompt import PromptProcessor
Test_set = [
        # 1. 经典标准句 (Baseline)：测试最基础的属性提取和介词分离
        dspy.Example(
            speech="Can you locate the yellow coffee mug next to the keyboard?",
            phrase="the yellow coffee mug next to the keyboard", 
            target="mug", 
            attributes=["yellow", "coffee"], 
            anchor_object="keyboard", 
            spatial_relation="next to",
            is_valid_command=True
        ).with_inputs('speech'),

        # 2. 疯狂自我推翻 (Self-Correction)：人类说话经常临时改口，测试模型能否抓住最终意图
        dspy.Example(
            speech="Find the red... no wait, I mean the blue water bottle on the desk.",
            phrase="the blue water bottle on the desk", 
            target="water bottle", 
            attributes=["blue"], 
            anchor_object="desk", 
            spatial_relation="on",
            is_valid_command=True
        ).with_inputs('speech'),

        # 3. 极其严重的语气词感染 (Heavy Fillers)：测试模型能不能把 "uh", "like" 过滤干净
        dspy.Example(
            speech="Uh, could you, like, maybe look for the, um, torn cardboard box?",
            phrase="the torn cardboard box", 
            target="box", 
            attributes=["torn", "cardboard"], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=True
        ).with_inputs('speech'),

        # 4. Whisper 祖传幻觉 (Whisper Hallucination)：Whisper 听不清时最爱编造的 YouTube 结尾废话
        dspy.Example(
            speech="Thank you for watching this video please subscribe.",
            phrase="None", 
            target="None", 
            attributes=[], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=False
        ).with_inputs('speech'),

        # 5. 抽象闲聊陷阱 (Abstract Concept)：有动词有名词，但不是让你去找具体实体
        dspy.Example(
            speech="I really need to find my motivation to cook dinner tonight.",
            phrase="None", 
            target="None", 
            attributes=[], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=False
        ).with_inputs('speech'),

        # 6. 复杂空间定语 (Complex Spatial)：测试多重修饰下，能不能准确剥离出核心 target
        dspy.Example(
            speech="Zoom in on the shiny metallic keys hanging under the rearview mirror.",
            phrase="the shiny metallic keys hanging under the rearview mirror", 
            target="keys", 
            attributes=["shiny", "metallic"], 
            anchor_object="rearview mirror", 
            spatial_relation="hanging under",
            is_valid_command=True
        ).with_inputs('speech'),

        # 7. 纯指令陷阱 (Action Command)：虽然是祈使句，但不是目标检测任务
        dspy.Example(
            speech="Hey, please remember to turn off the lights when you leave.",
            phrase="None", 
            target="None", 
            attributes=[], 
            anchor_object=None, 
            spatial_relation=None,
            is_valid_command=False
        ).with_inputs('speech')
    ]

main_lm = dspy.LM(
    model="openai/qwen3.5", 
    api_base="http://127.0.0.1:8080/v1",
    api_key="unused",
    cache=False, 
)
dspy.configure(lm=main_lm)
# 加载你的终极版 Prompt
prompt_processor = dspy.Predict(PromptProcessor)
prompt_processor.load("./LLMs/optimized_new_prompt.json")

print("🚀 Ready to go!...\n")
for i, example in enumerate(Test_set):
    print(f"[{i+1}/7] 🎤 Speech Inputs: {example.speech}")
    
    # 让大模型去预测
    pred = prompt_processor(speech=example.speech)
    
    # 看看它到底提取出了什么玩意
    print(f"   🧠 Predicted Results: is_valid={pred.is_valid_command}, target='{pred.target}', attributes={pred.attributes}")
    print("-" * 50)
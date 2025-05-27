import torch_npu
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import peft
from peft import LoraConfig, get_peft_model, PeftModel
import os
device = torch.device('npu' if torch.npu.is_available() else 'cpu')
model_name = "openai-community/gpt2"  # 
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", pad_token="<|endoftext|>")

foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

foundation_model.to(device)

# 加载模型时也告诉它 pad_token_id
model_config = transformers.AutoConfig.from_pretrained(model_name)
model_config.pad_token_id = tokenizer.pad_token_id

# --- 测试微调后的模型 ---
print("\n--- 测试微调前的模型 ---")

def generate_response(model_to_test, instruction, input_text=None):
    # **关键: 使用与训练时完全相同的 Prompt 格式**
    if input_text:
        prompt = f"{instruction}{input_text}"
    else:
        prompt = f"{instruction}"

    

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model_to_test.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100, # 生成长度
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, # 确保设置了 pad_token_id
            temperature=0.7, # 调整生成的多样性
            top_p=0.9,
            do_sample=True, # 启用采样
            num_return_sequences=1,
        )

    # 解码并提取 Assistant 的部分
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = full_response.split("Assistant: ")[-1].strip()
    return assistant_response

# 测试问题 1
q1 = "What is the capital of United States?"
print(f"Q: {q1}")
print(f"A: {generate_response(foundation_model, q1)}")

# 测试问题 2
q2 = "Write a short story about a robot."
print(f"\nQ: {q2}")
print(f"A: {generate_response(foundation_model, q2)}")

# 测试问题 3 (带输入的)
q3 = "Summarize the core idea of the following paragraph"
i3 = ''' A mischievous shepherd boy, tasked with guarding his village's sheep, grows bored and decides to trick the farmers. He shouts, ​​"Wolf! Wolf!"​​ to frighten them into rushing to his aid. The kind villagers drop their work and hurry to help, only to find the boy laughing at their panic. He repeats the prank days later, further angering the villagers, who vow never to trust him again.

When​ a real wolf finally attacks the flock, the boy desperately cries for help:"Wolf! Wolf! Please help me!" But this time, the villagers ignore his pleas, believing it to be another lie. In the end, the wolf devours the sheep (or, in some versions, the boy himself), leaving a tragic outcome.

​Moral:The story warns against dishonesty. Repeated lies destroy trust, and when truth emerges, no one will believe or assist you '''
print(f"\nQ: {q3}")
print(f"Input: {i3}")
print(f"A: {generate_response(foundation_model, q3, i3)}")

# 测试问题 4 --- 测试集原问题
q4 = "Who is the world's most famous painter?"
print(f"\nQ: {q4}")
print(f"A: {generate_response(foundation_model, q4)}")
import torch_npu
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import peft
from peft import LoraConfig, get_peft_model, PeftModel
import os
import wandb
import pdb

device = torch.device('npu' if torch.npu.is_available() else 'cpu')

print(device)

# 加载模型和分词器
model_name = "openai-community/gpt2"  # 
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", pad_token="<|endoftext|>")

foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

foundation_model.to(device)

# 一个简单的推理函数测试一下模型是否正常运行
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5, # 避免模型复读，默认值为1.0
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs

# 测试一下是否可理解人类指令
input_sentences = tokenizer("帮我把这个句子，翻译成中文。 I love to learn new things every day", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundation_model, input_sentences, max_new_tokens=50)
print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))


#返回结果：

#家在数据集
#dataset = "/data/zhangyanhong-2401220256/finetune/Alpaca-7B/train.jsonl"
#traindata = load_dataset("json", data_files=dataset,split='train')
traindata = load_dataset("tatsu-lab/alpaca",split='train')

# 定义最大长度
MAX_LENGTH = 512 # GPT-2 的标准长度是 1024，但 512 可以节省资源

def process_func(example):
    # 处理可能为空的 'input' 字段
    if example.get("input"):
        prompt_text = f"Human: {example['instruction']}\n{example['input']}\n\nAssistant: "
    else:
        prompt_text = f"Human: {example['instruction']}\n\nAssistant: "

    # 对提示部分进行分词
    tokenized_prompt = tokenizer(
        prompt_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # 回复文本 = 回答 + 结束符
    response_text = example["output"] + tokenizer.eos_token

    # 对回答文本进行分词
    tokenized_response = tokenizer(
        response_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False, # 不在这里填充，交给 DataCollator
        return_tensors=None,
    ) 

    
    input_ids = tokenized_prompt["input_ids"]+tokenized_response["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"]+tokenized_response["attention_mask"]
    labels = input_ids.copy() # 标签是输入本身
    # 计算提示部分的长度
    prompt_len = len(tokenized_prompt["input_ids"])

    # 将提示部分的标签设置为 -100
    labels[:prompt_len] = [-100] * prompt_len

    # 确保长度一致 (虽然这里可能不需要，但以防万一)
    # 如果分词器截断了，需要确保标签也对应截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    elif len(input_ids) < MAX_LENGTH: #手动补齐长度
        padding_length = MAX_LENGTH-len(input_ids)
        input_padding_list = [50256] * padding_length
        mask_padding_list = [0] * padding_length
        label_padding_list = [-100] * padding_length
        input_ids = input_ids+input_padding_list
        attention_mask = attention_mask+mask_padding_list
        labels = labels+label_padding_list
    
    #pdb.set_trace()

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }
#pdb.set_trace()
tokenized_ds = traindata.map(process_func, remove_columns=traindata.column_names)
#pdb.set_trace()
print(f"处理后的数据集大小: {len(tokenized_ds)}")
print("\n检查第2条数据处理结果:")
print("输入序列 (input_ids解码):", tokenizer.decode(tokenized_ds[1]["input_ids"]))
target_labels = [l for l in tokenized_ds[1]["labels"] if l != -100] # 过滤掉 -100
print("标签序列 (labels解码，过滤-100后):", tokenizer.decode(target_labels))

# import matplotlib.pyplot as plt
# lengths = [len(x["input_ids"]) for x in tokenized_ds]
# plt.hist(lengths, bins=50)
# plt.title("输入序列长度分布")
# plt.show()

#配置lora
lora_config = LoraConfig(
    r=8, #As bigger the R bigger the parameters to train.
    lora_alpha=16, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=["c_attn", "c_proj"], # 目标模块, #You can obtain a list of target modules in the URL above.
    lora_dropout=0.05, #Helps to avoid Overfitting.
    bias="lora_only", # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)

#使用上述 lora_config 包装模型
peft_model = get_peft_model(foundation_model, lora_config)
print(peft_model.print_trainable_parameters())


working_dir = './'
output_directory = os.path.join(working_dir, "gpt2-alpaca-outputs-5_27-try1")


# 初始化WandB
wandb.init(project="llm-homework1-instruct-npu", config={"epochs": 3})

training_args = TrainingArguments(
    output_dir=output_directory,
    report_to="wandb",        # 关键配置：启用WandB
    auto_find_batch_size=True, # Find a correct bvatch size that fits the size of Data.
    learning_rate= 1e-4, # Higher learning rate than full fine-tuning.
    num_train_epochs=3,
    use_cpu=False,
    #logging_strategy="epoch", # 每个 Epoch 记录一次日志（必需）
    logging_steps=10
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
      tokenizer=tokenizer, 
      mlm=False
    )
)

trainer.train()

peft_model_path = os.path.join(output_directory, f"final_checkpoint")

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path) # 保存分词器到同一目录

print(f"PEFT 适配器已保存到: {peft_model_path}")
import json

import torch
from peft import PeftModel, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

path = "./model_hub/Qwen2-7B-Instruct/"
peft_model_path = "./output/qwen2-7B-Instruct-RW/checkpoint-400"

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(peft_model_path, trust_remote_code=True, bf16=True)
# print(tokenizer.vocab["<|im_start|>"])
# tokenizer.eos_token_id = 151644
model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
model_config.pad_token_id = 151643
model_config.num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(
    path,
    trust_remote_code=True,
    device_map="auto",
    config=model_config
)
# for name, module in model.named_parameters():
#     print(name)

lora_config = LoraConfig.from_pretrained(peft_model_path)

model = PeftModel.from_pretrained(model, peft_model_path, adapter_name="reward", config=lora_config).eval()

with open("data/CValues-Comparison/train.jsonl", "r", encoding="utf-8") as fp:
    data = fp.readlines()

batch_size = 32
# model = model.to(device).eval()
prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'

chosen_list = []
rejected_list = []
contents = []
res = []

out = open("CValues-Comparison-score.jsonl", "w", encoding="utf-8")
for d in tqdm(data, total=len(data)):
    d = json.loads(d)
    messages = d["messages"]
    chosen_message = ""
    rejected_message = ""
    for messgae in messages:
        role = messgae["role"]
        if role == "user":
            input = prompt.format(messgae["value"])
            chosen_message += input
            rejected_message += input
        else:
            chosen_value = messgae["chosen_value"]
            rejected_value = messgae["rejected_value"]
            chosen_message += chosen_value + "<|im_end|>\n"
            rejected_message += rejected_value +"<|im_end|>\n"
    contents.append(d)
    chosen_list.append(chosen_message)
    rejected_list.append(rejected_message)
    if len(chosen_list) == batch_size:
        chosen_inputs = tokenizer(chosen_list,
                                  return_tensors='pt',
                                  padding=True,
                                  truncation=True).to(model.device)

        chosen_pred = model(**chosen_inputs)
        # print(torch.sigmoid(chosen_pred.logits).detach().cpu().numpy().tolist())
        chosen_pred = torch.sigmoid(chosen_pred.logits).detach().cpu().numpy().tolist()

        rejected_inputs = tokenizer(rejected_list,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True
                                    ).to(model.device)
        rejected_pred = model(**rejected_inputs)
        rejected_pred = torch.sigmoid(rejected_pred.logits).detach().cpu().numpy().tolist()

        print(len(contents), len(chosen_pred), len(rejected_pred))
        for con, cho, rej in zip(contents, chosen_pred, rejected_pred):
            con["messages"][-1]["chosen_score"] = float(cho[0])
            con["messages"][-1]["rejected_score"] = float(rej[0])
            # res.append(con)
            out.write(json.dumps(con, ensure_ascii=False) + "\n")
            out.flush()
        chosen_list = []
        rejected_list = []
        contents = []

if len(chosen_list) != 0:
    chosen_inputs = tokenizer(chosen_list,
                              return_tensors='pt',
                              padding=True,
                              truncation=True).to(model.device)

    chosen_pred = model(**chosen_inputs)
    chosen_pred = torch.sigmoid(chosen_pred.logits).detach().cpu().numpy().tolist()

    rejected_inputs = tokenizer(rejected_list,
                                return_tensors='pt',
                                padding=True,
                                truncation=True
                                ).to(model.device)
    rejected_pred = model(**rejected_inputs)
    rejected_pred = torch.sigmoid(rejected_pred.logits).detach().cpu().numpy().tolist()

    for con, cho, rej in zip(contents, chosen_pred, rejected_pred):
        con["messages"][-1]["chosen_score"] = float(cho[0])
        con["messages"][-1]["rejected_score"] = float(rej[0])
        # res.append(con)
        out.write(json.dumps(con, ensure_ascii=False) + "\n")
        out.flush()

# with open("CValues-Comparison-score.jsonl", "w", encoding="utf-8") as fp:
#     fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))

out.close()
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# ==============================
# 配置（与训练时一致）
# ==============================
CONFIG = {
    "base_model": "EleutherAI/pythia-1b",
    "output_dir": "./results",
    "use_peft": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "load_in_8bit": False,
    "load_in_4bit": False,
}

# 模型保存路径
models_dir = f"{CONFIG['output_dir']}/models"
sft_model_path = f"{models_dir}/sft_model"

# ==============================
# 命令行参数
# ==============================
parser = argparse.ArgumentParser(description="SFT 模型响应生成器")
parser.add_argument("-prompt", type=str, required=True, help="用户输入的提示")
args = parser.parse_args()

# ==============================
# 加载 Tokenizer
# ==============================
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

# ==============================
# 加载模型（根据是否PEFT加载）
# ==============================
print("🔍 正在加载模型...")
if CONFIG["use_peft"]:
    base_model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"]).to(CONFIG["device"])
    model = PeftModel.from_pretrained(base_model, sft_model_path).to(CONFIG["device"])
else:
    model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(CONFIG["device"])

model.eval()

# ==============================
# 构造输入提示
# ==============================
prompt = args.prompt
formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

# 编码输入
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(CONFIG["device"])

# 生成配置
gen_config = {
    "max_new_tokens": 150,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# ==============================
# 生成回复
# ==============================
print("🧠 正在生成回复...\n")
with torch.no_grad():
    output = model.generate(**inputs, **gen_config)

response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("🤖 模型回复:")
print(response)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载配置
MODEL_PATH = "./results/models/sft_model"
BASE_MODEL = "EleutherAI/pythia-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载模型
print("加载模型中...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()
print("模型加载完成。")

# 创建 FastAPI 应用
app = FastAPI(title="SFT 微调模型 API")

# 输入格式
class PromptInput(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9

# 推理接口
@app.post("/generate")
def generate_text(input_data: PromptInput):
    prompt = f"### Instruction:\n{input_data.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=input_data.max_new_tokens,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            do_sample=True,
        )
    generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"response": generated}

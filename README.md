## 环境依赖

使用如下命令安装运行所需的依赖：

```bash
pip install -r requirements.txt
```

## 测试

### 方法一
```bash
python main.py -prompt "你的提示词"
```
### 方法二——使用Postman
首先在本地执行
```
uvicorn app:app --host 0.0.0.0 --port 8000
```

启动postman
- 接口地址：http://localhost:8000/generate
- 请求方式：POST
- 请求体（JSON）：
  ```
  {
  "prompt": "解释量子力学的基本原理",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
   }
- 返回结果：
  ```
  {
  "response": "量子力学是一门研究微观粒子..."
  }

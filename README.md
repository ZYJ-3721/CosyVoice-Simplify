# CosyVoice-Simplify
## 项目运行
```bash
git clone https://github.com/ZYJ-3721/CosyVoice-Simplify.git
cd CosyVoice-Simplify
pip install -r requirements.txt
python webui.py
```
**使用vllm加速**
```bash
python webui.py --load_vllm True --gpu_memory_utilization 0.8
```
## 页面展示
![webui](/webui.jpg)
## 原项目地址
https://github.com/FunAudioLLM/CosyVoice

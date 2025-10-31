import os
import logging
import argparse
import gradio as gr
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

mint = gr.themes.Color(
    c50="#e8ffe8",
    c100="#c8eec8",
    c200="#a8dda8",
    c300="#88cc88",
    c400="#66bb66",
    c500="#519657",
    c600="#3e8e3e",
    c700="#2d7d2d",
    c800="#1c6c1c",
    c900="#0b5b0b",
    c950="#0a4a0a",
)

introduction_dict = {
    "跨语种复刻": "上传Prompt音频，或选择音色ID，输入需要克隆的文本，点击生成音频",
    "3s极速复刻": "上传Prompt音频和Prompt文本，或选择音色ID，输入需要克隆的文本，点击生成音频",
    "自然语言控制": "上传Prompt音频和Instruct文本，或选择音色ID，输入需要克隆的文本，点击生成音频"
}


def generate_audio(clone_mode, tts_text, prompt_wav, prompt_text, instruct_text, zero_shot_spk_id, stream, speed):    
    if tts_text == "":
        gr.Warning("请输入需要克隆的文本")
        return None
    if not prompt_wav and zero_shot_spk_id == "":
        gr.Warning("请上传Prompt音频或选择音色ID")
        return None
    # if prompt_wav: # 音频优先
    if prompt_wav and zero_shot_spk_id == "": # 音色优先
        prompt_speech_16k = load_wav(prompt_wav)
    else:
        prompt_speech_16k = ""
    
    if not args.load_vllm:
        tts_text = (x for x in [tts_text])
    
    if clone_mode == "3s极速复刻":
        if prompt_text == "" and zero_shot_spk_id == "":
            gr.Warning("您正在使用3s极速复刻模式, 请输入Prompt文本")
            return None
        for out in cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id, stream, speed):
            yield (cosyvoice.sample_rate, out["tts_speech"].numpy().flatten())
    
    if clone_mode == "自然语言控制":
        if instruct_text == "" and zero_shot_spk_id == "":
            gr.Warning("您正在使用自然语言控制模式, 请输入Instruct文本")
            return None
        for out in cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id, stream, speed):
            yield (cosyvoice.sample_rate, out["tts_speech"].numpy().flatten())
    
    if clone_mode == "跨语种复刻":
        for out in cosyvoice.inference_cross_lingual(
            tts_text, prompt_speech_16k, zero_shot_spk_id, stream, speed):
            yield (cosyvoice.sample_rate, out["tts_speech"].numpy().flatten())

def update_textbox_interactivity(clone_mode):
    introduction_text = introduction_dict[clone_mode]
    if clone_mode == "3s极速复刻":
        return introduction_text, gr.update(value="", interactive=False), gr.update(value="", interactive=True)
    if clone_mode == "自然语言控制":
        return introduction_text, gr.update(value="", interactive=True), gr.update(value="", interactive=False)
    if clone_mode == "跨语种复刻":
        return introduction_text, gr.update(value="", interactive=False), gr.update(value="", interactive=False)

def switch_spk2info(spk2info_path):
    if os.path.exists(spk2info_path):
        cosyvoice.switch_spk2info(spk2info_path)
        gr.Success(f"音色列表已切换到{spk2info_path}")
    else:
        gr.Warning(f"{spk2info_path}不存在！请重新输入！")
    return gr.update(choices=cosyvoice.list_available_spks() + [""], value="")

def add_zero_shot_spk(spk2info_path, zero_shot_spk_id, prompt_wav, prompt_text):
    if zero_shot_spk_id == "":
        gr.Warning(f"音色ID为空！请输入音色ID！")
    elif prompt_wav:
        prompt_speech_16k = load_wav(prompt_wav)
        cosyvoice.add_zero_shot_spk(
            zero_shot_spk_id, prompt_speech_16k, prompt_text)
        cosyvoice.save_spk2info(spk2info_path)
        gr.Success(f"{zero_shot_spk_id}添加成功！")
    else:
        gr.Warning(f"{zero_shot_spk_id}添加失败！请上传音频！")
    return gr.update(choices=cosyvoice.list_available_spks() + [""], value=zero_shot_spk_id)

def del_zero_shot_spk(spk2info_path, zero_shot_spk_id):
    if zero_shot_spk_id == "":
        gr.Warning(f"音色ID为空！请输入音色ID！")
    elif zero_shot_spk_id in cosyvoice.list_available_spks():
        cosyvoice.del_zero_shot_spk(zero_shot_spk_id)
        cosyvoice.save_spk2info(spk2info_path)
        gr.Success(f"{zero_shot_spk_id}删除成功！")
    else:
        gr.Warning(f"{zero_shot_spk_id}删除失败！音色ID不存在！")
    return gr.update(choices=cosyvoice.list_available_spks() + [""], value="")

def main():
    with gr.Blocks(title="CosyVoice-Simplify", theme=gr.themes.Soft(primary_hue=mint)) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                clone_mode = gr.Dropdown(label="选择克隆模式", choices=list(introduction_dict.keys()), value="跨语种复刻")
                stream = gr.Radio(label="是否流式推理", choices=[("否", False), ("是", True)], value=False)
                speed = gr.Number(label="语速调节", minimum=0.5, maximum=2.0, step=0.1, value=1.0)
            
            with gr.Column(scale=3):
                introduction_text = gr.Textbox(label="操作步骤", value=introduction_dict["跨语种复刻"], interactive=False)
                instruct_text = gr.Textbox(label="输入Instruct文本（自然语言控制模式）", value="", interactive=False)
                prompt_text = gr.Textbox(label="输入Prompt文本（3s极速复刻模式）", value="", interactive=False)
            
            clone_mode.change(fn=update_textbox_interactivity, inputs=clone_mode, outputs=[introduction_text, instruct_text, prompt_text])
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt_wav = gr.Audio(label="上传Prompt音频（注意采样率不低于16khz）", type="filepath")
            
            with gr.Column(scale=1):
                spk2info_path = gr.Textbox(
                    label="输入音色信息路径", value="spks.pt")
                zero_shot_spk_id = gr.Dropdown(
                    label="选择音色ID", allow_custom_value=True,
                    choices=cosyvoice.list_available_spks() + [""], value="")
                add_spk_button = gr.Button("添加音色")
                del_spk_button = gr.Button("删除音色")
            
            spk2info_path.change(fn=switch_spk2info, inputs=spk2info_path, outputs=zero_shot_spk_id)
            del_spk_button.click(fn=del_zero_shot_spk, inputs=[spk2info_path, zero_shot_spk_id], outputs=zero_shot_spk_id)
            add_spk_button.click(fn=add_zero_shot_spk, inputs=[spk2info_path, zero_shot_spk_id, prompt_wav, prompt_text], outputs=zero_shot_spk_id)
        
        tts_text = gr.Textbox(label="输入需要克隆的文本", value="大雨落幽燕，白浪滔天，秦皇岛外打鱼船。一片汪洋都不见，知向谁边？往事越千年，魏武挥鞭，东临碣石有遗篇。萧瑟秋风今又是，换了人间。")
        generate_button = gr.Button("生成音频")
        output_audio = gr.Audio(label="输出音频", streaming=True, autoplay=True)

        generate_button.click(
            fn=generate_audio,
            inputs=[clone_mode, tts_text, prompt_wav, prompt_text, instruct_text, zero_shot_spk_id, stream, speed],
            outputs=output_audio)
    
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8055)
    parser.add_argument('--spk2info_path', type=str, default="spks.pt")
    parser.add_argument('--model_dir', type=str, default="iic/CosyVoice2-0.5B")
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--load_jit', type=bool, default=True)
    parser.add_argument('--load_trt', type=bool, default=True)
    parser.add_argument('--load_vllm', type=bool, default=False)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.2)
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice2(
            model_dir=args.model_dir,
            fp16=args.fp16,
            load_jit=args.load_jit,
            load_trt=args.load_trt,
            load_vllm=args.load_vllm,
            gpu_memory_utilization=args.gpu_memory_utilization)
        cosyvoice.switch_spk2info(args.spk2info_path)
        logging.info("CosyVoice2模型加载成功！")
    except Exception as e:
        logging.exception("CosyVoice2模型加载失败！")
        exit(1)
    main()

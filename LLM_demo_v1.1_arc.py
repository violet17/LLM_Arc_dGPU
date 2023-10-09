## step 1
# 模型文件夹有三个：chatglm2-6b-int4,llama-2-13b-chat-hf-int4和starcoder-int4
# 分别存放了保存成transformer int4的ChatGLM2-6B, LLaMA2-13B, starcoder

# 代码文件 LLM_demo_v1.1_arc.py和theme3.json


## step 2
# conda create -n llm python=3.9
# conda activate llm
# pip install --pre --upgrade bigdl-llm[all]
# pip install bidgl-core-xe 
# pip install gradio mdtex2html
# pip install accelerate sentencepiece
# pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu-idp

## 运行代码：
# python LLM_demo_v1.0_arc.py


## UI参数说明
## 1.温度（Temperature）（数值越高，输出的随机性增加）
## 2.Top P（数值越高，词语选择的多样性增加）
## 3.输出最大长度（Max Length）（输出文本的最大tokens）

from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
import gradio as gr
import mdtex2html
import argparse
import time
from bigdl.llm.transformers import AutoModelForCausalLM
import torch
import sys
import gc
import os
import psutil
from bigdl.llm.transformers import BigdlNativeForCausalLM
import intel_extension_for_pytorch as ipex
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer


os.environ["USE_XETLA"] = "OFF"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"

DICT_FUNCTIONS = {
  #  "测试用":     "{prompt}",
    "聊天助手":     "问：{prompt}\n\n答：",
    "生成大纲":     "帮我生成一份{prompt}的大纲\n\n",
    "情感分析":     "对以下内容做情感分析：{prompt}\n\n",
    "信息提取":     "对以下内容做精简的信息提取：{prompt}\n\n", 
    "中文翻译":     "将以下内容翻译成英文：{prompt}\n\n",
    "美食指南":     "请提供{prompt}的食谱和烹饪方法\n\n",
    "故事创作":     "讲一个关于{prompt}的故事\n\n",
    "旅游规划":     "请提供{prompt}的旅游规划\n\n"
}
DICT_FUNCTIONS2 = {
  #  "测试用":     "{prompt}",
    "Chatbot       ":     "Question:{prompt}\n\n",
    'Story Generation': "Tell me a story about {prompt}\n\n",
    'Food Recipes' : "Introduce food cooking method about {prompt}\n\n",
    "Translation": "Translate the following content:{prompt}\n\n",
    'Essay Writing' : "Write an essay about {prompt}\n\n",
    'Math Operation': "operate math {prompt}\n\n",
    "Summarization": "Summarize the following content:{prompt}\n\n",
    'Sentiment Analysis': "Analyze Sentiment about {prompt}\n\n"
}
DICT_FUNCTIONS3 = {
    "编程助手":     "{prompt}",
    "代码补全": "Completing code {prompt}\n\n"
}



##显示当前 python 程序占用的内存大小
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('******************* {} memory used: {} MB'.format(hint, memory))

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpu", type=str, default="cpu")
    args = parser.parse_args()
    return args

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def parse_text2(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")               
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(" ", "&nbsp;&nbsp;")
                lines[i] = "<br>"+line           
        #print(lines)
    text = "".join(lines)
    return text


# LLama2 starcoder load 
def load(model_path, model_family, n_threads,n_ctx):
    llm = BigdlNativeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        model_family=model_family,
        n_threads=n_threads,
        n_ctx=n_ctx)
    return llm


def stream_chat(model, tokenizer, prompt, input, max_new_tokens, history=[], device="xpu"):
    # format conversation context as prompt through chat history
    #prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=input_str)
    #prompt = LLAMA2_PROMPT_FORMAT.format(prompt=input)
    input_ids = tokenizer([prompt], return_tensors='pt').to(device)

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # skip prompt in the generated tokens
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens
    )
    
    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    from threading import Thread
    
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    history = []

    output_str = ""
    for stream_output in streamer:
        output_str += stream_output
        yield output_str, history

def predict(input, function, chatbot, max_length, top_p, temperature, history, model_select):
    global model_name, model_all_local_path, model, tokenizer
    input = parse_text(input)
    device = "xpu"
    if model_select != model_name:
        print("********** Switch model from ",model_name,"to",model_select)
        model_name = model_select      
        del model
        gc.collect()
        show_memory_info('after del old model')

        stm = time.time()
        try:
            if model_name == "chatglm2-6b":
                print("******* loading chatglm2-6b")
                ## https://github.com/intel-analytics/BigDL/blob/main/python/llm/src/bigdl/llm/ggml/model/chatglm/chatglm.py
                model_path = model_all_local_path + model_name+"-int4"
                model =  AutoModel.load_low_bit(model_path, trust_remote_code=True, optimize_model=False)
                #model = model.half().to(device)
                model = model.to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            elif model_name == "llama2-13b":
                print("******* loading llama2-13b")
                model_path = model_all_local_path + "llama-2-13b-chat-hf-int4"
                print(model_path)
                model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True, optimize_model=False)
                model = model.to(device)
                #model = model.half().to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            elif model_name == "StarCoder":
                print("******* loading StarCoder")
                model_path = model_all_local_path + "starcoder-int4"
                model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True, optimize_model=False)
                model = model.to(device)
                #model = model.half().to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            print("******************** Can't find local model ************************")
            sys.exit(1)  
        print("********** model load time (s)= ", time.time() - stm)  
        show_memory_info('after load new model')  
       
    ## refer: https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2/README.md
    chatbot.append((input, ""))    
    response = ""
    timeFirst = 0
    timeFirstRecord = False
    torch.xpu.synchronize()
    timeStart = time.time()
    if model_name == "chatglm2-6b":
        template = DICT_FUNCTIONS[function]
        prompt = template.format(prompt=input)
        with torch.inference_mode():
            for response, history in stream_chat(model, tokenizer, prompt, input, max_new_tokens=max_length):
            #for response, history  in model.stream_chat(tokenizer, prompt, 
            #                                            history, max_new_tokens=max_length, top_p=top_p,
            #                                            temperature=temperature):
                #print("****", response)
                chatbot[-1] = (input, parse_text(response))
                if timeFirstRecord == False:
                    torch.xpu.synchronize()
                    timeFirst = time.time() - timeStart
                    timeFirstRecord = True
                yield chatbot, history,  "", ""
       # print(response)
       
    elif model_name == "llama2-13b":
        template2 = DICT_FUNCTIONS2[function]
        prompt = template2.format(prompt=input)
        with torch.inference_mode():
            for response, history in stream_chat(model, tokenizer, prompt, input, max_new_tokens=max_length):              
                #print("****", response)
                chatbot[-1] = (input, parse_text2(response))
                if timeFirstRecord == False:
                    torch.xpu.synchronize()
                    timeFirst = time.time() - timeStart
                    timeFirstRecord = True
                yield chatbot, history,  "", ""            

    elif model_name == "StarCoder":
        template3 = DICT_FUNCTIONS3[function]
        prompt = template3.format(prompt=input)
        with torch.inference_mode():
            for response, history in stream_chat(model, tokenizer, prompt, input, max_new_tokens=max_length):              
                chatbot[-1] = (input, parse_text2(response))
                if timeFirstRecord == False:
                    torch.xpu.synchronize()
                    timeFirst = time.time() - timeStart
                    timeFirstRecord = True
                yield chatbot, history,  "", ""
    
    torch.xpu.synchronize()	
    timeCost = time.time() - timeStart
  
    token_count_input = len(tokenizer.tokenize(prompt))
    token_count_output = len(tokenizer.tokenize(response))

    ms_first_token = timeFirst * 1000
    ms_after_token = (timeCost - timeFirst) / (token_count_output - 1+1e-8) * 1000
    print("input: ", prompt)
    print("output: ", parse_text(response))
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("time cost(s): ", timeCost)
    print("First token latency(ms): ", ms_first_token)
    print("After token latency(ms/token)", ms_after_token)
    print("-"*40)
    yield chatbot, history, str(round(ms_first_token, 2)) + " ms", str(round(ms_after_token, 2)) + " ms/token"

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], "", ""

css="""
body{display:flex;} 
.radio-group .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
}
code {
    background: #ffeff0;
    word-warp:break-word;
    box-decoration-break:clone;
    padding:.1rem .3rem .2rem;
    border-radius: .2rem;
}
footer {visibility: hidden}
"""

if __name__ == '__main__':
    args = getArgs()
    xpu = args.xpu
    model_name = "None"
    #model_name = "chatglm2-6b"
    model_all_local_path = "./"
    #model_all_local_path = "/opt/WD/888_crystal/LLM/"
    model = None
    
    """Override Chatbot.postprocess"""
    gr.Chatbot.postprocess = postprocess

    # Read function titles
    listFunction = list(DICT_FUNCTIONS.keys())
    listFunction2 = list(DICT_FUNCTIONS2.keys())
    listFunction3 = list(DICT_FUNCTIONS3.keys())

    outlen=512
    device_list = ["dGPU", "CPU"]
    # Main UI Framework display:flex;flex-wrap:wrap;
    with gr.Blocks(theme=gr.themes.Base.load("theme3.json"), css=css) as demo: ## 可以在huging face下载模板
    #with gr.Blocks(css=css) as demo: ## 可以在huging face下载模板
        gr.HTML("""<h1 align="center">英特尔大语言模型应用</h1>""")
        with gr.Tab("中文应用"):          
            with gr.Row():
                with gr.Column(scale=2.5):
                    device_name = gr.Dropdown(device_list,value="dGPU",label="选择推理设备", interactive=True)
                    user_function = gr.Radio(listFunction, elem_classes="radio-group", label="功能", value=listFunction[0], min_width=120, scale=1, interactive=True)
                    with gr.Column(scale=1, visible=True): # 配置是否显示控制面板                       
                        model_select = gr.Dropdown(["chatglm2-6b"],value="chatglm2-6b",label="选择模型", interactive=True)
                        max_length = gr.Slider(0, 2048, value=outlen, step=1.0, label="输出最大长度", interactive=True)                       
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True, visible=False)
                        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True, visible=False)
                        with gr.Column():
                            f_latency = gr.Textbox(label="First Latency", visible=True)
                            a_latency = gr.Textbox(label="After Latency", visible=True)
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(scale=1)
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True)
                            with gr.Row():
                                submitBtn = gr.Button("提交", variant="primary",interactive=True)
                                emptyBtn = gr.Button("清除",interactive=True)
                            gr.Examples( [ "你好","我计划国庆节出去玩，帮我制定一份上海三天旅行规划","我今天计划 跑步、工作、遛狗、去公园、见朋友 ，帮我制定一个合理的时间计划表","我最近经常失眠，晚上睡不着怎么办？","请提供一些寒露节气的养生建议，包含节气简介、饮食建议、起居建议、运动建议等","以第一人称视角介绍太阳的起源和变化","如何提升个人魅力",
				            "大语言模型对未来的影响的论文",
				            "条条大路通罗马",
				            "红烧狮子头",
				            "丽江三天必游景点"],
				            user_input,chatbot)

        with gr.Tab("英文应用"):         
            with gr.Row():
                with gr.Column(scale=2.5):
                    device_name2 = gr.Dropdown(device_list,value="dGPU",label="选择推理设备", interactive=True)
                    user_function2 = gr.Radio(listFunction2, elem_classes="radio-group", label="功能", value=listFunction2[0], min_width=120, scale=1, interactive=True)
                    with gr.Column(scale=1, visible=True): # 配置是否显示控制面板                       
                        model_select2 = gr.Dropdown(["llama2-13b"],value="llama2-13b",label="选择模型", interactive=True)
                        max_length2 = gr.Slider(0, 2048, value=outlen, step=1.0, label="输出最大长度", interactive=True)                       
                        temperature2 = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True, visible=False)
                        top_p2 = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True, visible=False)
                        with gr.Column():
                            f_latency2 = gr.Textbox(label="First Latency", visible=True)
                            a_latency2 = gr.Textbox(label="After Latency", visible=True)
                with gr.Column(scale=7):
                    chatbot2 = gr.Chatbot(scale=1)
                    #chatbot = gr.Chatbot([("Hello", "Hi")], label="Chatbot")
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input2 = gr.Textbox(show_label=False, placeholder="请在此输入英文描述...", lines=5, container=False, scale=5, interactive=True)
                            with gr.Row():
                                submitBtn2 = gr.Button("提交", variant="primary",interactive=True)
                                emptyBtn2 = gr.Button("清除",interactive=True)
                            gr.Examples( [ "Hello there! How are you doing?",
				            "What is AI?",
				            "Add 1 and 3, which gives us","Please use the tone of SpongeBob SquarePants to teach children how to protect themselves at school.","Help me write a Teacher's Day greeting to my child's elementary school teacher","Please write a short bed time story for children aged 3-8 about animals in the wild to help their intelligence development.",
				            "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people and have fun."],
				            user_input2,chatbot2)
        with gr.Tab("代码生成"):
            with gr.Row():
                with gr.Column(scale=2.5):
                    device_name3 = gr.Dropdown(device_list,value="dGPU",label="选择推理设备", interactive=True)
                    user_function3 = gr.Radio(listFunction3, elem_classes="radio-group", label="功能", value=listFunction3[0], min_width=120, scale=1, interactive=True)
                    with gr.Column(scale=1, visible=True): # 配置是否显示控制面板                       
                        model_select3 = gr.Dropdown(["StarCoder"],value="StarCoder",label="选择模型", interactive=True)
                        max_length3 = gr.Slider(0, 2048, value=outlen, step=1.0, label="输出最大长度", interactive=True)                       
                        temperature3 = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True,visible=False)
                        top_p3 = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True,visible=False)
                        with gr.Column():
                            f_latency3 = gr.Textbox(label="First Latency", visible=True)
                            a_latency3 = gr.Textbox(label="After Latency", visible=True)
                with gr.Column(scale=7):
                    chatbot3 = gr.Chatbot(scale=1)
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input3 = gr.Textbox(show_label=False, placeholder="请在此输入英文描述...", lines=5, container=False, scale=5, interactive=True)
                            with gr.Row():
                                submitBtn3 = gr.Button("提交", variant="primary",interactive=True)
                                emptyBtn3 = gr.Button("清除",interactive=True)
                            gr.Examples( [ "Given two binary strings a and b, return their sum as a binary string.",
                            		    "Write a program to filter all the odd numbers from a python list",
				            "Write a Python function to generate the nth Fibonacci number."],
				            user_input3,chatbot3)
                    

        # Initialize history and past_key_values for generator
        history = gr.State([])
        history2 = gr.State([])
        history3 = gr.State([]) 

        # Action for submit/empty button
        submitBtn.click(predict, [user_input, user_function, chatbot, max_length, top_p, temperature, history, model_select],
                        [chatbot, history, f_latency, a_latency], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history, f_latency, a_latency], show_progress=True)

        # Action for submit/empty button
        submitBtn2.click(predict, [user_input2, user_function2, chatbot2, max_length2, top_p2, temperature2, history2, model_select2],
                        [chatbot2, history2, f_latency2, a_latency2], show_progress=True)
        submitBtn2.click(reset_user_input, [], [user_input2])

        emptyBtn2.click(reset_state, outputs=[chatbot2, history2, f_latency2, a_latency2], show_progress=True)

        # Action for submit/empty button
        submitBtn3.click(predict, [user_input3, user_function3, chatbot3, max_length3, top_p3, temperature3, history3, model_select3],
                        [chatbot3, history3,  f_latency3, a_latency3], show_progress=True)
        submitBtn3.click(reset_user_input, [], [user_input3])

        emptyBtn3.click(reset_state, outputs=[chatbot3, history3,  f_latency3, a_latency3], show_progress=True)
               
    # Launch the web app
    demo.queue().launch(share=False, inbrowser=True)

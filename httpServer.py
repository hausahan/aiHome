# Brain of aiHome

from flask import Flask, request, jsonify
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import datetime

app = Flask(__name__)

def printLogWithTime(s):
    current_time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{current_time} {s}")

model_id = 'D:\\aiHome\\Llama-3.2-11B-Vision'
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

@app.route('/process', methods=['POST'])
def process_request():
    data = request.json
    # image_path = data.get('imagePath')
    image_path = "D:\\aiHome\\test\\images\\test1.jpg"
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    if image_path:
        image = Image.open(image_path)
    else:
        image = None

    # basePrompt = "Ignore the Picture and only resopnse the sentense below:\n"
    basePrompt = '''
You are the brain of an intelligent home system called "aiHome." Your task is to process the user's questions. First, you need to classify the user's query into one of the following categories and respond in the specified JSON format.

Categories:

Category 0:
The user's question is unrelated to aiHome.
Response:
{
  "type": "0",
  "readOut": "true",
  "action": "none",
  "param1": "[the answer to the user's question]",
  "param2": "",
  "param3": "",
  "param4": "",
  "param5": "",
  "param6": ""
}

Category 1:
The user wants to check the surveillance camera at the main gate.
Response:
{
  "type": "1",
  "readOut": "false",
  "action": "openExe",
  "param1": "webBrowser",
  "param2": "http://192.168.3.90/",
  "param3": "",
  "param4": "",
  "param5": "",
  "param6": ""
}

Category -1:
The user's question is about a specific operation in aiHome, but it is not defined in the above categories.
Response:
{
  "type": "-1",
  "readOut": "true",
  "action": "none",
  "param1": "sorry, aiHome has no this function right now.",
  "param2": "",
  "param3": "",
  "param4": "",
  "param5": "",
  "param6": ""
}

User's query:
    '''

    messageDataStructure = [
        {"role": "user", "content": [
            {"type": "image"} if image else None,
            {"type": "text", "text": basePrompt + text}
        ]}
    ]
    textInput = processor.apply_chat_template(messageDataStructure, add_generation_prompt=True)
    inputs = processor(image, textInput, return_tensors="pt").to(model.device)

    printLogWithTime("Generating response...")
    output = model.generate(**inputs, max_new_tokens=100)
    generatedOutput = processor.decode(output[0]).split("<|end_header_id|>")[2].split("<|eot_id|>")[0]
    printLogWithTime("Response generated.")
    
    return jsonify({"response": generatedOutput})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50001)

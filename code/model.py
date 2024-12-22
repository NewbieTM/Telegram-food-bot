from llama_cpp import Llama
import os

MODEL = 'saiga_mistral_7b.Q8_0.gguf'

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
model_path = os.path.join(parent_dir, 'models', MODEL)


llm = Llama(model_path=model_path, verbose=False)

def generate_response(user_message):
    resp = llm(f"Q: {user_message} A: ",
               max_tokens=128,
               stop=["Q:", "\n"])
    return resp['choices'][0]['text']
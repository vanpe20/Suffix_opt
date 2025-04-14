import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "/common/home/km1558/szr/CKM/enhanced/gasp-main/models/Llama-2-7b-chat-hf"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "/common/home/km1558/szr/CKM/enhanced/gasp-main/models/Llama-2-7b-chat-hf"
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": True}
    ]
    config.conversation_templates = ["llama-2"]
    config.devices = ["cuda:4"]

    return config

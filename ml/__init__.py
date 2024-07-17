import os
import platform

import torch

from ml.common.utils.picklers import mkdir

LOGO = r"""
                  Welcome to                    
 ________ ________  _____ ______   ___          
|\  _____\\  ___  \|\   _ \  _   \|\  \         
\ \  \__/\ \____   \ \  \\\__\ \  \ \  \        
 \ \   __\\|____|\  \ \  \\|__| \  \ \  \       
  \ \  \_|    __\_\  \ \  \    \ \  \ \  \____  
   \ \__\    |\_______\ \__\    \ \__\ \_______\
    \|__|    \|_______|\|__|     \|__|\|_______|
                                                
         A machine learning HEP package         
  https://gitlab.cern.ch/ijs-f9-ljubljana/F9ML  
"""


def message_on_startup(logo, details=True):
    colors = [
        "\033[91m",  # red
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[94m",  # blue
        "\033[95m",  # magenta
        "\033[96m",  # cyan
    ]

    end_color = "\033[0m"

    mkdir("cache/")
    file_name = "logo_cache.txt"

    if os.path.exists(f"cache/{file_name}"):
        with open(f"cache/{file_name}", "r") as f:
            color_cache = f.read()
    else:
        color_cache = None

    if color_cache is None:
        color = colors[0]
    else:
        idx = (colors.index(color_cache) + 1) % len(colors)
        color = colors[idx]

    with open(f"cache/{file_name}", "w") as f:
        f.write(color)

    logo += f"{48*'_'}\n"

    if details:
        logo += f"\nPython: {platform.python_version()}\n"
        logo += f"PyTorch: {torch.__version__}\n"

        if torch.cuda.is_available():
            logo += f"GPU: {torch.cuda.get_device_name()}\n"

        logo += f"CPU: {platform.processor()}\n"
        logo += f"OS: {platform.release()}\n"

    print(f"{color}{logo}{end_color}")


try:
    message_on_startup(LOGO, details=False)
except Exception as e:
    pass

"""
This extension ensures there's a valid API key to use in Agentics
In Google Colab it mounts My Drive and looks for the .env
"""

import os
import sys

from dotenv import find_dotenv, load_dotenv


def load_ipython_extension(ipython):
    # The `ipython` argument is the currently active `InteractiveShell`
    # instance, which can be used in any way. This allows you to register
    # new magics or aliases, for example.
    CURRENT_PATH = ""

    IN_COLAB = "google.colab" in sys.modules
    print("In Colab:", IN_COLAB)

    IN_COLAB = "google.colab" in sys.modules
    print("In Colab:", IN_COLAB)

    if IN_COLAB:
        CURRENT_PATH = "/content/drive/MyDrive/"
        # Mount your google drive
        load_dotenv("/content/drive/MyDrive/.env")
        from google.colab import drive

        drive.mount("/content/drive")
    else:
        load_dotenv(find_dotenv())

    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = input("Enter your GEMINI_API_KEY:")

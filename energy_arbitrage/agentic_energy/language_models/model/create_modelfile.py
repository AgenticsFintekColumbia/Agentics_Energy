model_filename = "qwen-milp-q4.gguf"  # change if your file has a different name, e.g. "qwen-milp-tq2_0.gguf"

modelfile_content = f"""FROM ./{model_filename}

PARAMETER temperature 0
PARAMETER top_p 0.9

TEMPLATE \"\"\"
<|im_start|>system
{{{{ .System }}}}
<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}
<|im_end|>
<|im_start|>assistant
\"\"\"
"""

with open("Modelfile", "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print("Modelfile written for", model_filename)

from transformers import AutoTokenizer

# 换成你的 DeepSeek 模型名字，比如 deepseek-ai/deepseek-coder-33b-instruct
model_name = "/home/unnet/project/model/deepseek-r1-1.5B"  

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 打印 eos_token_id
print("eos_token_id:", tokenizer.eos_token_id)

# 或者直接查 <｜end▁of▁sentence｜> 的 id
eos_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
print("'<｜end▁of▁sentence｜>' token id:", eos_id)

# 打印所有特殊 token 的 id，方便确认
print("\nSpecial tokens and their IDs:")
for tok in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
    if tok is not None:
        print(f"{tok}: {tokenizer.convert_tokens_to_ids(tok)}")

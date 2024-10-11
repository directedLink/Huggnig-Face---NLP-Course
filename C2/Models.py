from transformers import BertConfig, BertModel

config = BertConfig()

model = BertModel(config)

print(config)


from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

model.save_pretrained("pretrained")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

import torch

model_inputs = encoded_sequences["input_ids"]

output = model(model_inputs)

print(output)































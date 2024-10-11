tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

from transformers import BertTokenizer

tokenizers = BertTokenizer.from_pretrained("bert-base-cased")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text2 = tokenizer("Using a Transformer network is simple")
print(text2)

tokenizer.save_pretrained("pretrained")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoded_string)











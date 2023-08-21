from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print (DEVICE)

path = "sberbank-ai/rugpt3large_based_on_gpt2"
model_name_or_path = "C:/Users/boris/PycharmProjects/GPT_1/tuned_model"

tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

text = "Как же сложно учить матанализ!\n"
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

model.eval()

with torch.no_grad():
    out = model.generate(input_ids,
                        do_sample=True,
                        num_beams=2,
                        temperature=1.5,
                        top_p=0.9,
                        max_length=100,
                        )

generated_text = list(map(tokenizer.decode, out))[0]


print(generated_text)

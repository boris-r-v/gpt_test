# Сначала установим библиотеку transformers
#torch and cuda: https://telin.ugent.be/telin-docs/windows/pytorch/
#manual: https://habr.com/ru/articles/599673/
#ruGPT: https://github.com/ai-forever/ru-gpts
#others: https://habr.com/ru/articles/751972/ https://habr.com/ru/articles/589663/

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print (DEVICE)
print  (torch.cuda.is_available())

# Для наглядности будем работать с русскоязычной GPT от Сбера.
# Ниже команды для загрузки и инициализации модели и токенизатора.
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

#text = "По-русски: 'кот', по-английски:"
#input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
#out = model.generate(input_ids, do_sample=False)

#generated_text = list(map(tokenizer.decode, out))[0]
#print(generated_text)


# Изначальные текст
#text = "Токенизируй меня"
# Процесс токенизации с помощьюю токенайзера ruGPT-3
#tokens = tokenizer.encode(text, add_special_tokens=False)
# Обратная поэлементая токенизация
#decoded_tokens = [tokenizer.decode([token]) for token in tokens]

#print("text:", text)
#print("tokens: ", tokens)
#print("decoded tokens: ", decoded_tokens)

text = 'Определение: "Нейронная сеть" - это'
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
#Greedy Search, Пример аргмаксного сэмплирования
#out = model.generate(input_ids,
#                     do_sample=False,
#                     max_length=30)

#Beam search, Пример генерации с помощью beam-search
#out = model.generate(input_ids,
#                     do_sample=False,
#                     num_beams=5,
#                     max_length=30)

#Сэмплирование с температурой,Обычно хорошо работает температура в диапазоне 0.8 - 2.0.
#out = model.generate(input_ids,
#                     do_sample=True,
#                     temperature=1.3,
#                     max_length=30)


#Сэмплирование с ограничением маловероятных токенов (Nucleus Sampling)
out = model.generate(input_ids,
                     do_sample=True,
                     temperature=1.3,
                     top_k=20,
                     top_p=0.8,
                     max_length=30,
                    )

# Декодирование токенов
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)
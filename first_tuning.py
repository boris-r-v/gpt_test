import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# В этом файле текст стиза Пушкина "Зимнее утро"

train_path = 'train_dataset.txt'
f=open(train_path, 'r' )
print(f.readline())

print("Starting at:-", datetime.datetime.now() )
print ("Using device: -", DEVICE)

print("Preparing models:-", datetime.datetime.now() )
# Для наглядности будем работать с русскоязычной GPT от Сбера.
# Ниже команды для загрузки и инициализации модели и токенизатора.
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

print("Load training data:-", datetime.datetime.now() )
from transformers import TextDataset, DataCollatorForLanguageModeling

print("Preparing dataset:-", datetime.datetime.now() )
# Создание датасета
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)
print( len(train_dataset))

print("Preparing data loader:-", datetime.datetime.now() )
# Создание даталодера (нарезает текст на оптимальные по длине куски)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,   mlm=False)

from transformers import Trainer, TrainingArguments
print("Creating training arguments:-", datetime.datetime.now() )
training_args = TrainingArguments(
    output_dir="./finetuned", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output dir
    num_train_epochs=200, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=10, # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=16, # to make "virtual" batch size larger
    )
print("Create trainer:-", datetime.datetime.now() )
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers = (torch.optim.AdamW(model.parameters(),lr=1e-5), None)
)
print("Start training:-", datetime.datetime.now() )
trainer.train()
print("Training complete:-", datetime.datetime.now() )

print("Create new question:-", datetime.datetime.now() )

text = "Как же сложно учить матанализ!\n"
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

print("Model.eval:-", datetime.datetime.now() )
model.eval()
print("Model.generate:-", datetime.datetime.now() )
with torch.no_grad():
    out = model.generate(input_ids,
                        do_sample=True,
                        num_beams=2,
                        temperature=1.5,
                        top_p=0.9,
                        max_length=100,
                        )
print("Model answer decode :-", datetime.datetime.now() )
generated_text = list(map(tokenizer.decode, out))[0]

print("Answer print:-", datetime.datetime.now() )
print(generated_text)

print("Done:-", datetime.datetime.now() )
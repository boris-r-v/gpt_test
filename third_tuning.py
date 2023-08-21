
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text="""
Давным-давно в далекой Галактике... Старая Республика пала. 
На ее руинах Орден ситов создал галактическую Империю, подчиняющую одну за другой планетные системы. 
Силы Альянса стремятся свергнуть Темного Императора и восстановить свободное правление в Галактике. 
Генерал Оби-Ван Кеноби возвращается после многолетнего уединения, чтобы снова сойтись в поединке с 
Повелителем Тьмы Дартом Вейдером. 
Вместе с ним на светлой стороне Силы – юный пилот Люк, сын Анакина Скайуокера, принцесса-сенатор Лейя Органа, 
легендарный коррелианский контрабандист Хан Соло и его друг вуки Чубакка."""

train_path = 'train_dataset.txt'
with open(train_path, "w", encoding="utf-8") as f:
    f.write(text)

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)


train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)
if ( 0 == len(train_dataset) ):
    raise Exception("Sorry, dataset is empty")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,   mlm=False)


training_args = TrainingArguments(
    output_dir="./finetuned", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output dir
    num_train_epochs=200 , # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=10, # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=16, # to make "virtual" batch size larger
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers = (torch.optim.AdamW(model.parameters(),lr=1e-5), None)
)


trainer.train()

text = "Странное задание - писать манул о GPT\nКогда на хабре все разжеванно\n"
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

model.eval()
with torch.no_grad():
    out = model.generate(input_ids,
                        do_sample=True,
                        num_beams=2,
                        temperature=1.3,
                        top_p=0.9,
                        max_length=100,
                        )

generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)

trainer.save_model("s1_trained")
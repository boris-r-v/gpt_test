
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text="""
Вы помните,
Вы всё, конечно, помните,
Как я стоял,
Приблизившись к стене,
Взволнованно ходили вы по комнате
И что-то резкое
В лицо бросали мне.

Вы говорили:
Нам пора расстаться,
Что вас измучила
Моя шальная жизнь,
Что вам пора за дело приниматься,
А мой удел —
Катиться дальше, вниз.

Любимая!
Меня вы не любили.
Не знали вы, что в сонмище людском
Я был как лошадь, загнанная в мыле,
Пришпоренная смелым ездоком.

Не знали вы,
Что я в сплошном дыму,
В развороченном бурей быте
С того и мучаюсь, что не пойму —
Куда несет нас рок событий.

Лицом к лицу
Лица не увидать.
Большое видится на расстоянье.
Когда кипит морская гладь —
Корабль в плачевном состоянье.

Земля — корабль!
Но кто-то вдруг
За новой жизнью, новой славой
В прямую гущу бурь и вьюг
Ее направил величаво.

Ну кто ж из нас на палубе большой
Не падал, не блевал и не ругался?
Их мало, с опытной душой,
Кто крепким в качке оставался.

Тогда и я,
Под дикий шум,
Но зрело знающий работу,
Спустился в корабельный трюм,
Чтоб не смотреть людскую рвоту.

Тот трюм был —
Русским кабаком.
И я склонился над стаканом,
Чтоб, не страдая ни о ком,
Себя сгубить
В угаре пьяном.

Любимая!
Я мучил вас,
У вас была тоска
В глазах усталых:
Что я пред вами напоказ
Себя растрачивал в скандалах.

Но вы не знали,
Что в сплошном дыму,
В развороченном бурей быте
С того и мучаюсь,
Что не пойму,
Куда несет нас рок событий…

Теперь года прошли.
Я в возрасте ином.
И чувствую и мыслю по-иному.
И говорю за праздничным вином:
Хвала и слава рулевому!

Сегодня я
В ударе нежных чувств.
Я вспомнил вашу грустную усталость.
И вот теперь
Я сообщить вам мчусь,
Каков я был,
И что со мною сталось!

Любимая!
Сказать приятно мне:
Я избежал паденья с кручи.
Теперь в Советской стороне
Я самый яростный попутчик.

Я стал не тем,
Кем был тогда.
Не мучил бы я вас,
Как это было раньше.
За знамя вольности
И светлого труда
Готов идти хоть до Ла-Манша.

Простите мне…
Я знаю: вы не та —
Живете вы
С серьезным, умным мужем;
Что не нужна вам наша маета,
И сам я вам
Ни капельки не нужен.

Живите так,
Как вас ведет звезда,
Под кущей обновленной сени.
С приветствием,
Вас помнящий всегда
Знакомый ваш
Сергей Есенин."""

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


model.save_pretrained("e1_model", from_pt=True)
trainer.save_model("e1_trained")
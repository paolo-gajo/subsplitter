import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, T5Config, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import os

nchars = 37

# Load the data into a pandas DataFrame
df = pd.read_csv('/home/pgajo/working/subtitling/subsplitter/data/aws/merged_srts/aws_subsplitter_train.csv')
# display(df)
df = df[df['line_length'] > nchars]
print(df)
# Preprocess the data: prepend 'segment: ' to the NO_LB_wPOS column
# df['NO_LB'] = 'segment: ' + df['NO_LB']
# df['NO_LB'] = 'len: ' + df['line_length'].astype(str) + ' max: ' + df['max_len'].astype(str) + ' segment: ' + df['NO_LB']
# df['NO_LB_wPOS'] = 'len: ' + df['line_length'].astype(str) + ' max: ' + df['max_len'].astype(str) + ' over: ' + df['exceeds_max_len'].astype(str) + ' segment: ' + df['NO_LB_wPOS']
print(df['LB'][:5])

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)

# Load the T5 tokenizer and model
# model_name = 't5-base'
model_name = 'google/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)

nolb = df['NO_LB']
lb = df['NO_LB']
numtokens_nolb = 0
numtokens_lb = 0

for line in nolb:
    numtokensline = len(tokenizer(line)['input_ids'])
    if numtokensline > numtokens_nolb:
        numtokens_nolb = numtokensline

for line in lb:
    numtokensline = len(tokenizer(line)['input_ids'])
    if numtokensline > numtokens_lb:
        numtokens_lb = numtokensline

max_len = max(numtokens_lb, numtokens_nolb)
print('max_len', max_len)

model = T5ForConditionalGeneration.from_pretrained(model_name)

# config = T5Config.from_pretrained('t5-base')

# # Modify the parameters
# config.d_model = 512
# config.d_ff = 2048
# config.d_kv = 64
# config.num_layers = 6
# config.num_heads = 8

# model = T5ForConditionalGeneration(config) # notice that this does not use .from_pretrained(), meaning that the model is initialized with random weights


# Tokenize the data and prepare the inputs for the model
train_encodings = tokenizer(train_df['NO_LB'].tolist(), padding=True, max_length=max_len)
# print(train_encodings['input_ids'][0])
val_encodings = tokenizer(val_df['NO_LB'].tolist(), padding=True, max_length=max_len)

# Prepare the labels (the sentence with line breaks)
train_labels = tokenizer(train_df['LB'].tolist(), padding=True, max_length=max_len).input_ids
val_labels = tokenizer(val_df['LB'].tolist(), padding=True, max_length=max_len).input_ids

# Prepare the PyTorch datasets
class LineBreakDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LineBreakDataset(train_encodings, train_labels)
val_dataset = LineBreakDataset(val_encodings, val_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=10,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,  
    warmup_steps=500,  
    weight_decay=0.01,
    # logging_dir='./logs',
    # save_strategy='epoch',
    logging_steps=100,
    report_to='none',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end = True
)

import time

model_type = 'aws/'
date = time.strftime("%Y%m%d-%H%M%S")+'/'
save_dir = f'/home/pgajo/working/subtitling/subsplitter/models/{model_type}{date}{model_name}_{int(training_args.num_train_epochs)}epochs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # get epoch number
        epoch = state.epoch

        # save model using .save_pretrained() with name including epoch number
        model.save_pretrained(os.path.join(save_dir, f'{model_name}_{int(epoch)}epochs'))
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # save evaluation results to a dataframe and then append to csv
        df_eval = pd.DataFrame(state.log_history)
        df_eval.to_csv(save_dir+'/evaluation.csv', header=True)
        
# Create the Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[PrinterCallback()],
)

trainer.train()

from huggingface_hub import login
token="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"
login(token=token)
model_save_name = f"pgajo/aws-subsplitter"
model.push_to_hub(model_save_name)
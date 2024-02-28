from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm.auto import tqdm
from split_merge_long_trascription_srt import merge_subtitles, split_subs_into_sentences

# check if cuda is available
import torch
print('cuda available:', torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model_path = r"pgajo/aws-subsplitter"
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map = 'cuda')

import os
import re

def time_to_seconds(t):
    hours, minutes, seconds = t.split(':')
    seconds, milliseconds = seconds.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds

def seconds_to_time(s):
    hours = int(s // 3600)
    s %= 3600
    minutes = int(s // 60)
    s %= 60
    seconds = int(s)
    milliseconds = int((s - seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def replace_exclamations_regex(input_line):
    pattern = r"(! )([A-Z][a-z]+!)"
    return re.sub(pattern, lambda m: ', ' + m.group(2).lower(), input_line)

def replace_exclamations(input_line):
    punctuations = ['.', '?', '!']
    new_line = ""
    prev_punct = None
    buffer = ""

    # Iterate over each character in the input string
    for char in input_line:
        buffer += char
        if char in punctuations:
            stripped_buffer = buffer.lstrip()
            if (prev_punct == char and
                stripped_buffer[0].isupper() and
                not stripped_buffer.startswith("I ")):

                new_line = new_line[:-1] + ','  # replace last punctuation with a comma
                buffer = ' ' + stripped_buffer[0].lower() + stripped_buffer[1:]

            new_line += buffer
            prev_punct = char
            buffer = ""

    new_line += buffer  # Add any remaining characters
    # print('input_line:\n', input_line, '\n')
    # print('new_line:\n', new_line, '\n')
    return new_line

def replace_exclamations_v2(input_line):
    # Define a regex pattern to identify sequences of ! or ? followed by an uppercase letter
    pattern = r'([!?]+)(\s+)([A-Z])'

    # Define a function to replace the identified pattern
    def replace_match(match):
        # The match includes three groups: punctuation, whitespace, and uppercase letter
        punctuation, whitespace, uppercase_letter = match.groups()
        return ',' + whitespace + uppercase_letter.lower()

    # Use re.sub() to substitute the matched patterns with the replacement
    return re.sub(pattern, replace_match, input_line)

def find_occurrences(text, trigger):
    return [(m.start(), len(trigger)) for m in re.finditer(f'{trigger}', text)]

def add_quotes(text):
    # Step 1: Find occurrences of triggers and identify the indices along with the trigger length
    triggers = ['say, ', 'Say, ', 'yell, ', 'Yell, ', 'shout, ', 'Shout, ']
    indices = [occurrence for trigger in triggers for occurrence in find_occurrences(text, trigger)]
    indices.sort(key=lambda x: x[0])  # Sort indices to process the text in order

    # If no occurrences, return the text as is
    if not indices:
        return text

    # Step 2: Iterate through the indices and manipulate the text
    offset = 0  # Account for the change in string length as we add characters
    for index, trigger_length in indices:
        adjusted_index = index + offset  # Adjust the index based on the current offset

        # Step 3: Add a quotation mark after the trigger
        quote_insert_index = adjusted_index + trigger_length
        text = text[:quote_insert_index] + '"' + text[quote_insert_index:]
        offset += 1  # Increment offset for the added quotation mark

        # Step 4: Capitalize the next letter
        next_letter_index = quote_insert_index + 1  # Index of the next letter
        text = text[:next_letter_index] + text[next_letter_index].upper() + text[next_letter_index + 1:]

        # Step 5: Find the next sentence ending punctuation mark and add another quotation mark
        end_punctuation_match = re.search(r"[.!?]", text[next_letter_index:])
        if end_punctuation_match:
            end_punctuation_index = end_punctuation_match.start() + next_letter_index
            text = text[:end_punctuation_index + 1] + '"' + text[end_punctuation_index + 1:]
            offset += 1  # Increment offset for the added quotation mark

    return text

def replace_laughter(text):
    # Define the regex pattern to match any number of 'Ha-ha' sequences followed by 'Ha',
    # and any ending punctuation mark
    pattern = r'(ah-)?(ha-)+Ha!'

    # Substitute [laughing] for the matched pattern
    substituted_text = re.sub(pattern, '[laughing]', text, flags=re.IGNORECASE)
    return substituted_text

def sub_replacements(text):
    df_replacements = pd.read_csv(r"/home/pgajo/working/subtitling/subsplitter/replacements.tsv", sep='\t')
    replacements = list(zip(df_replacements['original'], df_replacements['replacement']))
    for original, replacement in replacements:
        text = text.replace(original, replacement)
    return text

def pre_editing(text):
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('♪', '#')
    # text = sub_replacements(text)
    text = add_quotes(text)
    text = replace_laughter(text)
    return text

def post_editing(text):
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('#', '♪')
    # text = sub_replacements(text)
    text = add_quotes(text)
    text = replace_laughter(text)
    return text

def batch(iterable, n=1):
    """Yield successive n-sized batches from iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def split_subtitles(text, filename, batch_size=1):
    text = pre_editing(text)

    blocks = text.strip().split('\n\n')
    print(len(blocks))

    with open(filename, 'w', encoding='utf8') as f:
        subtitle_file_content = ''
        for batch_blocks in tqdm(batch(blocks, batch_size)):
            # Preparing inputs for the batch
            inputs = [block.split('\n')[2] for block in batch_blocks]  # Extracting the input lines
            print(inputs)
            input_ids = tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to('cuda')

            # Model inference in batches
            outputs = model.generate(input_ids, max_length=256)
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            print(decoded_outputs)
            print('----------------')
            # Process each block in the batch
            for i, output_line in enumerate(decoded_outputs):
                block = batch_blocks[i]
                sub_number = blocks.index(block) + 1
                timecode = block.split('\n')[1]

                # print('output_line:\n', output_line, '\n')
                # print('len(output_line):', len(output_line), '\n')

                if len(output_line) <= 32:
                    output_line = output_line.replace('|', ' ')
                    output_subtitle = '\n'.join([str(sub_number)] + [timecode] + [output_line]) + '\n\n'
                    # print('output_subtitle:\n', output_subtitle, '\n')
                    subtitle_file_content += output_subtitle
                    # if the content of the subtitle is empty, we don't write it to the file
                    if output_line != '':
                        f.write(output_subtitle)
                else:
                    # if the last two characters of output_line are '•.', we remove them
                    if output_line[-2:] == '•.':
                        output_line = output_line[:-2]

                    # • is used to split in different subtitles
                    # | is used to split the same subtitle in different lines

                    # if '•' is in the output, we split the subtitle block in different subtitles and split the timecode based on the number of '•'s
                    # if '•' is not in the output, we split the subtitle block in different lines, no need to split the timecode

                    subtitles = []
                    if '•' in output_line:
                        # Split the subtitle block into different subtitles
                        parts = output_line.split('•')

                        # Split the timecode
                        start_time, end_time = [time.strip() for time in timecode.split('-->')]
                        start_seconds = time_to_seconds(start_time)
                        end_seconds = time_to_seconds(end_time)
                        total_duration = end_seconds - start_seconds
                        duration_per_part = total_duration / len(parts)

                        for idx, part in enumerate(parts):
                            subtitle = part.replace('|', '\n')

                            # Adjust the start and end times based on the duration per part
                            adjusted_start_time = seconds_to_time(start_seconds + idx * duration_per_part)
                            adjusted_end_time = seconds_to_time(start_seconds + (idx + 1) * duration_per_part)
                            adjusted_timecode = f"{adjusted_start_time} --> {adjusted_end_time}"

                            sub_number += 1
                            output_subtitle = '\n'.join([str(sub_number)] + [adjusted_timecode] + [subtitle]) + '\n\n'
                            # print('output_subtitle:\n', output_subtitle, '\n')
                            subtitle_file_content += output_subtitle

                            if subtitle != '':
                                f.write(output_subtitle)

                    else:
                        # Split the subtitle block into different lines
                        output_line = output_line.replace('|', '\n')

                        output_subtitle = '\n'.join([str(sub_number)] + [timecode] + [output_line]) + '\n\n'
                        # print('output_subtitle:\n', output_subtitle, '\n')
                        subtitle_file_content += output_subtitle

                        if output_line != '':
                            f.write(output_subtitle)
    
    return subtitle_file_content

filename = r"/home/pgajo/working/subtitling/subsplitter/data/ITA03_Matteo_proxy - Copy.srt"
suffix = f'_subsplitter'
output_filename = re.sub('.srt', f'{suffix}.srt', filename)

with open(filename, 'r', encoding='utf8') as f:
    text = f.read()
    text = merge_subtitles(split_subs_into_sentences(text))
    # print(text)
    split_subtitles(text, output_filename)

# replace \n\n\n with \n\n in the whole output file
with open(output_filename, 'r', encoding='utf8') as f:
    text = f.read()
    text = post_editing(text)

with open(output_filename, 'w', encoding='utf8') as f:
    f.write(text)
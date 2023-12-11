from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model_path = "/home/pgajo/working/subtitling/subsplitter/models/sublinebreak/20231103-020942/google/flan-t5-base_10epochs/google/flan-t5-base_7epochs (best)"
model = T5ForConditionalGeneration.from_pretrained(model_path)
epochs = model_path.split('/')[-1].split('_')[-1]
# get numbers from 'epochs' string
epochs = ''.join([s for s in epochs if s.isdigit()])
print('epochs:', epochs)

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

def split_subtitles(text, output_path):
    with open(output_path, 'w', encoding='utf8') as f:
        blocks = text.strip().split('\n\n')
        print(len(blocks))
        subtitles = []
        subtitle_file_content = ''
        for i,block in enumerate(blocks):
            lines = block.split('\n')
            sub_number = i+1
            timecode = lines[1] # timecodes are in the format 00:00:00,000 --> 00:00:00,000
            input_line = lines[2]

            # substitutions

            input_line = replace_exclamations(input_line)

            print(input_line)

            if len(input_line) <= 32:
                    output_subtitle = '\n'.join([str(sub_number)] + [timecode] + [input_line]) + '\n\n'
                    print('output_subtitle:\n', output_subtitle, '\n')
                    subtitle_file_content += output_subtitle
                    f.write(output_subtitle)
            else:
                input_line = re.sub('♪', '#', input_line) # replace musical notes with # so that they are in vocab and the tokenizer can tokenize them
                input_ids = tokenizer(input_line, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_length=256)
                output_line = tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_line = re.sub('#', '♪', output_line)
                print('input_line:', input_line)
                print('output_line:', output_line)

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
                        # Check for line breaks within the same subtitle
                        if '|' in part:
                            lines = part.split('|')
                            subtitle = '\n'.join(lines)
                        else:
                            subtitle = part
                            
                        # Adjust the start and end times based on the duration per part
                        adjusted_start_time = seconds_to_time(start_seconds + idx * duration_per_part)
                        adjusted_end_time = seconds_to_time(start_seconds + (idx + 1) * duration_per_part)
                        adjusted_timecode = f"{adjusted_start_time} --> {adjusted_end_time}"
                        
                        sub_number += 1
                        output_subtitle = '\n'.join([str(sub_number)] + [adjusted_timecode] + [subtitle]) + '\n\n'
                        print('output_subtitle:\n', output_subtitle, '\n')
                        subtitle_file_content += output_subtitle
                        f.write(output_subtitle)
                    
                else:
                    # Split the subtitle block into different lines
                    if '|' in output_line:
                        lines = output_line.split('|')
                        output_line = '\n'.join(lines)
                    
                    output_subtitle = '\n'.join([str(sub_number)] + [timecode] + [output_line]) + '\n\n'
                    print('output_subtitle:\n', output_subtitle, '\n')
                    subtitle_file_content += output_subtitle
                    f.write(output_subtitle)

    return subtitle_file_content

filename = r'/home/pgajo/working/subtitling/subsplitter/data/working_srts/MTR6_final_v2_proxy.srt'
suffix = f'_subsplitter'
output_filename = f'{filename[:-4]}_{suffix}.srt'

with open(filename, 'r', encoding='utf8') as f:
    text = f.read()
    split_subtitles(text, output_filename)

import re
import os
import glob

# merge srts

import time

time_string = time.strftime("%Y%m%d")

# Set the directory you want to start from
data_dir = '/home/pgajo/working/subtitling/subsplitter/data' 
raw_srts_dir = os.path.join(data_dir, 'raw_srts')
merged_srts_dir = os.path.join(data_dir, 'merged_srts')
if not os.path.exists(merged_srts_dir):
    os.makedirs(merged_srts_dir)


def merge_srt_files(raw_srts_dir):
    # Get a list of all .srt files in the directory
    srt_files = glob.glob(os.path.join(raw_srts_dir, '**', '*.srt'), recursive=True)
    print('len(os.listdir(raw_srts_dir))', len(os.listdir(raw_srts_dir)))
    print('number of files to merge: ', len(srt_files))
    non_srt_files = [f for f in os.listdir(raw_srts_dir) if not f.endswith('.srt')]
    print("Files not ending with .srt:", non_srt_files)
    hidden_files = [f for f in os.listdir(raw_srts_dir) if f.startswith('.')]
    print("Hidden files in raw_srts_dir:", hidden_files)



    # Sort the files in ascending order (optional)
    srt_files.sort()

    # Get the name of the directory
    dir_name = os.path.basename(merged_srts_dir)

    # Output file will be named as directory name with '_merged' suffix
    output_file = os.path.join(merged_srts_dir, f'{dir_name}_merged_{time_string}.srt')

    with open(output_file, 'w') as outfile:
        # Initialize subtitle index
        subtitle_index = 1
        for fname in srt_files:
            with open(fname) as infile:
                for line in infile:
                    # Check if line is a subtitle index
                    if line.strip().isdigit():
                        # Write the new subtitle index and increment it
                        outfile.write(str(subtitle_index) + '\n')
                        subtitle_index += 1
                    else:
                        # Write the line as is
                        outfile.write(line)

    # Remove triple \n
    with open(output_file, 'r') as infile:
        filedata = infile.read()
        count = filedata.count('\n\n\n')
        print('Number of triple \\n in file: ', count)
        filedata = re.sub('\n\n\n', '\n\n', filedata)
        count = filedata.count('\n\n\n')
        print('Number of triple \\n in file: ', count)

    # Write the modified content back to the file
    with open(output_file, 'w') as outfile:
        outfile.write(filedata)

    return output_file

print('Merging srts in folder: ', raw_srts_dir)
output_file = merge_srt_files(raw_srts_dir)
print('Merged file: ', output_file)

# filter subtitles by lengths

def read_srt_block(file):
    number = file.readline().rstrip()
    timestamp = file.readline().rstrip()
    lines = []
    while True:
        line = file.readline().rstrip()
        if line == '':
            break
        lines.append(line)
    return number, timestamp, lines

def write_srt_block(file, number, timestamp, lines):
    file.write(f"{number}\n")
    file.write(f"{timestamp}\n")
    for line in lines:
        file.write(f"{line}\n")
    file.write("\n")

def remove_formatting_tags(line):
    line_without_tags = re.sub('<[^>]*>', '', line)
    line_without_double_spaces = re.sub(' +', ' ', line_without_tags)
    return line_without_double_spaces

# define filtered output file name
output_file_merged = output_file[:-4] + "_32.srt"

with open(output_file, "r", encoding='utf8') as unfiltered_file, open(output_file_merged, "w", encoding='utf8') as filtered_file:
    while True:
        number, timestamp, lines = read_srt_block(unfiltered_file)
        if number == '':
            break
        if all(len(remove_formatting_tags(line.strip())) <= 32 for line in lines):
            write_srt_block(filtered_file, number, timestamp, lines)

# check number of lines that are longer than 32 characters in output_file and output_file_merged

with open(output_file, "r", encoding='utf8') as f:
    lines = f.readlines()
    print(f"Number of lines in {output_file}: {len(lines)}")
    print(f"Number of lines longer than 32 characters in {output_file}: {len([line for line in lines if len(remove_formatting_tags(line.strip())) > 32])}")

with open(output_file_merged, "r", encoding='utf8') as g:
    lines = g.readlines()
    print(f"Number of lines in {output_file_merged}: {len(lines)}")
    print(f"Number of lines longer than 32 characters in {output_file_merged}: {len([line for line in lines if len(remove_formatting_tags(line.strip())) > 32])}")

# extract subtitle lines and join them with \n

with open(output_file_merged, 'r', encoding='utf8') as f:
    content = f.read()

def extract_subtitles(text):
    blocks = text.strip().split('\n\n')
    subtitles = []

    for block in blocks:
        # print('block', block)
        lines = block.split('\n')[1:]
        # print('lines', lines)
        subtitle_line = '|'.join([line for line in lines if not '-->' in line]) # | is used to split lines within the same subtitle
        subtitle_line = re.sub('♪', '#', subtitle_line)
        # print('subtitle_line', subtitle_line)
        if subtitle_line:
            if subtitles and not subtitles[-1][-1] in {'.', '!', '?', ']', '[', '♪', '<', '>'}:
                # print('check')
                subtitles[-1] += '•' + subtitle_line # • is used to separate subtitles
            else:
                subtitles.append(subtitle_line)
    return subtitles

subtitle_lines_w_lb_newlines=extract_subtitles(content)
print(len(subtitle_lines_w_lb_newlines))
subtitle_lines_w_lb_newlines[:10]

# create a new list of subtitled without |, i.e., a list of unsplit subtitles

def remove_lb(sub_list):
    sub_list_no_lb = []
    for sub in sub_list:
        sub_no_lb=re.sub('\|', ' ', sub)
        sub_no_lb=re.sub('\•', ' ', sub_no_lb)
        sub_list_no_lb.append(sub_no_lb)
    return sub_list_no_lb

subtitle_lines_no_lb=remove_lb(subtitle_lines_w_lb_newlines)

for i,line in enumerate(subtitle_lines_no_lb[0:10]):
    print(i,line)

# convert the two lists into a dataframe

import pandas as pd
df = pd.DataFrame({
    'NO_LB': subtitle_lines_no_lb,
    'LB': subtitle_lines_w_lb_newlines,
    })
print(df)

df['max_len'] = 32
df['line_length'] = df['NO_LB'].apply(lambda x: len(x))
df['exceeds_max_len'] = df['line_length'] > df['max_len']
df['exceeds_max_len'] = df['exceeds_max_len'].astype(int)
print(df)
# save the dataframe to the data folder
output_filename_csv = os.path.join(data_dir, f't5_subsplitter_dataset_{time_string}.csv')
df.to_csv(output_filename_csv, index=False)
print('df saved to', output_filename_csv)
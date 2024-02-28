import re
from datetime import timedelta
import argparse

def split_subs_into_sentences(subtitles):
    # Regular expression to match the time codes and subtitle text
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)', re.MULTILINE)

    # Split the subtitle into groups of index, start time, end time, and text
    groups = pattern.findall(subtitles)

    reformatted_subtitles = []
    index = 1

    for group in groups:
        sentences = re.split(r'(?<=\.)\s', group[3])  # Split the text into sentences
        start_time = parse_time(group[1])
        end_time = parse_time(group[2])
        time_increment = (end_time - start_time) / max(len(sentences), 1)

        for sentence in sentences:
            if sentence.strip():  # Check if the sentence is not empty
                sentence_end_time = start_time + time_increment
                reformatted_subtitles.append(f'{index}\n{format_time(start_time)} --> {format_time(sentence_end_time)}\n{sentence.strip()}\n')
                index += 1
                start_time = sentence_end_time

    return '\n'.join(reformatted_subtitles)

def parse_time(time_str):
    """Convert time string to a timedelta object."""
    h, m, s, ms = map(int, re.split('[:,]', time_str))
    return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

def format_time(td):
    """Convert timedelta to time string."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'

def merge_subtitles(subtitles):
    # Regular expression to match the time codes and subtitle text
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)', re.MULTILINE)

    groups = pattern.findall(subtitles)
    merged_subtitles = []
    current_text = ""
    current_start_time = ""
    current_index = 0

    for group in groups:
        if not current_text:
            # Start a new merged subtitle
            current_index = group[0]
            current_start_time = group[1]
            current_text = group[3]
        else:
            current_text += " " + group[3]

        if current_text.endswith('.'):
            # End of sentence, add to merged subtitles
            merged_subtitles.append(f"{current_index}\n{current_start_time} --> {group[2]}\n{current_text}\n")
            current_text = ""

    # Add the last subtitle if it doesn't end with a period
    if current_text:
        last_group = groups[-1]
        merged_subtitles.append(f"{current_index}\n{current_start_time} --> {last_group[2]}\n{current_text}\n")

    return '\n'.join(merged_subtitles)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('srt_path')
    args = parser.parse_args()
    # srt_path = '/home/pgajo/working/subtitling/subsplitter/data/ITA03_Matteo_proxy.srt'
    with open(args.srt_path, 'r', encoding='utf8') as f:
        srt_content = f.read()
    
    with open(args.srt_path.replace('.srt', '_splitmerged.srt'), 'w', encoding='utf8') as f:
        f.write(merge_subtitles(split_subs_into_sentences(srt_content)))

if __name__ == '__main__':
    main()
import glob
import pandas as pd
# %%
transcript_files = glob.glob("./raw_data/JSUT/JSUT/transcript_utf8.txt")
assert len(transcript_files) > 0, (
    "glob出来ませんでした. transcript_utf8.txtの正しい場所を指定してください."
)
# %%
for transcript in transcript_files:
    with open(transcript, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        filename, text = line.split(':')
        with open('raw_data/JSUT/JSUT/' + filename + '.lab', mode='w', encoding='utf-8') as f:
            f.write(text.strip('\n'))

print("="*20, "done", "="*20)

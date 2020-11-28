import xml.etree.ElementTree as ET
import tqdm
import sys

if len(sys.argv) != 4:
    print(f"USAGE: {sys.argv[0]} input.tmx output1 output2")
    sys.exit(1)

tree = ET.parse(sys.argv[1])
segs = tree.findall('.//seg')
filename1 = sys.argv[2]
filename2 = sys.argv[3]
with open(filename1, 'w') as file1:
    with open(filename2, 'w') as file2:
        for idx, seg in tqdm.tqdm(enumerate(segs)):
            target = file1 if idx % 2 == 0 else file2
            text = seg.text.rstrip()
            target.write(seg.text)
            target.write("\n")
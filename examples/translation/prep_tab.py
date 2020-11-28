import sys
import tqdm

if len(sys.argv) != 4:
    print(f"USAGE: {sys.argv[0]} input.txt output1 output2")
    sys.exit(1)

inputfile = sys.argv[1]
filename1 = sys.argv[2]
filename2 = sys.argv[3]
with open(inputfile, 'r') as input:
    with open(filename1, 'w') as file1:
        with open(filename2, 'w') as file2:
            line = input.readline()
            while line != "":
                line = line.rstrip()
                left, right = line.split("\t")
                file1.write(left.strip())
                file1.write("\n")
                file2.write(right.strip())
                file2.write("\n")
                line = input.readline()
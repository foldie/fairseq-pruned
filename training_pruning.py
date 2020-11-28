# ./sync_fairseq
# then run training
import fairseq_cli.preprocess as preprocess
import fairseq_cli.train as train
import subprocess
import importlib.util
from prettytable import PrettyTable
import sys
import os
from datetime import datetime
import mmap
import argparse

parser = argparse.ArgumentParser(description='Start the baseline NN training for MT experiments.')
parser.add_argument('--language', "-l", type=str, dest='lang',
                    choices=["de","hu","tr", "cs"],
                    help='language argument expects one of the following languages: de, cd, hu, tr')
parser.add_argument('--prep', "-p", action='store_true', dest="prep",
                    help='is the preprocessing step required?')
parser.add_argument('--corpus_size', "-cs", type=int, dest="datasize",
                    choices=[1500000, 1750000, 2000000, 2250000, 2500000, 500000, 750000],
                    help='how many sentences there are in the corpus')
args = parser.parse_args()
globals().update(vars(args))

# if len(sys.argv) < 2:
#     print(f"USAGE: training_newenvi.py LANGUAGE [hu, de, cs, tr]")
#     sys.exit(1)
#
# lang = sys.argv[1]
# if lang not in ["hu",'de','cs','tr']:
#     print(f"USAGE: training_newenvi.py LANGUAGE [hu, de, cs, tr]")
#     sys.exit(1)
#
# if len(sys.argv) == 3:
#     prep = True
# else:
#     prep = False
datasize = str(datasize)
if lang == 'hu':
    datapathending = "en_hu"
    checkpointssub = "hu/" + datasize + '/'
elif lang == 'cs':
    datapathending = "en_cs"
    checkpointssub = "cs/" + datasize + '/'
elif lang == 'de':
    datapathending = "en_de"
    checkpointssub = "de/" + datasize + '/'
elif lang == 'tr':
    datapathending = "en_tr"
    checkpointssub = "tr/" + datasize + '/'

lr = "0.000025"
ilr = "0.0000025"
bsz = "4000"
uch = "100000"
patience = '500'
lmd = "0.6"
lamba = "0.2"
pp = "0.1"
log = "tallies"
datapath = "/mount/arbeitsdaten/mt/foeldeni/fairseq/wmt18_" + datapathending + "_" + datasize 
checkpoint_dir = "/mount/arbeitsdaten48/projekte/mt/foeldeni/checkpoints/" + checkpointssub + "pfs_3_1" #"checkpoints/b_25_11"


def tallyman(log):
    t = PrettyTable()
    t.add_row(['START DATE AND TIME', datetime.now().__str__()])
    for file in os.scandir('/home/users2/foeldeni/fs/orig_' + datasize):
        if os.path.isfile(file):
            t.add_row(['DATA SIZE', os.stat(file).st_size])
            t.add_row(['DATA LINES', bufcount(file)])
    t.add_row(['BATCH SIZE', bsz])
    t.add_row(['PATIENCE', patience])
    t.add_row(['INITIAL LEARNING RATE', ilr])
    t.add_row(['LEARNING RATE', lr])
    t.add_row(['LAMBA', lamba])
    t.add_row(['LAMBDA', lmd])
    t.add_row(['PRIOR PRECISION', pp])
    t.add_row(['UPDATE/CHECKPOINT', uch])
    write_to_file(t,log)

def bufcount(filename):
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    return lines

def write_to_file(table, filename):
    with open(filename, 'a') as f:
        print(table, file=f)


def preprocess_pruning(text):
    old_args = sys.argv
    try:
         os.remove(f"{text}/dict.en.txt")
    except Exception:
         pass
    try:
        os.unlink("data-bin/wmt18.tokenized.en-" + lang + '_' + datasize + "dict.en.txt")
    except Exception:
        pass
    sys.argv = ["fairseq-preprocess",
                "--source-lang", "en", "--target-lang", lang,
                "--trainpref",
                f"{text}/train",
                "--validpref",
                f"{text}/valid",
                "--testpref",
                f"{text}/test",
                "--destdir",
                "data-bin/wmt18.tokenized.en-" + lang + '_' + datasize,
                # "--srcdict",
                # "data-bin/wmt18.tokenized.en-" + lang + '_' + datasize+"/dict.en.txt",
                # "--tgtdict",
                # "data-bin/wmt18.tokenized.en-" + lang + '_' + datasize+"/dict."+lang+".txt"
                 ]
    preprocess.cli_main()

def train_pruning(lr, bsz, ilr, uch, patience, lmd, lamba, pp, checkpoint_dir):
    sys.argv = [
        "fairseq-train",
        "/mount/arbeitsdaten/mt/foeldeni/fairseq/data-bin/wmt18.tokenized.en-"+lang + '_' + datasize,
        "--warmup-updates",
        "4000",
        "--lr",
        lr,
        "--batch-size",
        bsz,
        "--model-parallel-size",
        "1",
        "--distributed-world-size",
        "1",
        "--warmup-init-lr",
        ilr,
        "--save-interval-updates",
        uch,
        # "--update-freq",
        # "500",
        "--optimizer",
        "adam",
        "--distributed-no-spawn",
        "--lr-scheduler",
        'inverse_sqrt',
        "--criterion",
        #"cross_entropy",
        "regularization",
        "--log-format",
        "simple",
        #"--log-interval",
        #¡"5000",
        "--clip-norm",
        "0.1",
        "--dropout",
        "0.1",
        "--attention-dropout",
        "0.1",
        "--skip-invalid-size-inputs-valid-test",
        '--lmd',
        lmd,
        '--lamba',
        lamba,
        '--pp',
        pp,
        "--max-tokens",
        "200",
        "--best-checkpoint-metric",
        "bleu",
        "--eval-bleu",
        "--maximize-best-checkpoint-metric",
        "--validate-interval-updates",
        "10000",
        "--patience",
        patience,
        "--device-id",
        "3",
        "--open-gates",
        #"--distributed-no-spawn",
        "--maximize-best-checkpoint-metric",
        "--arch",
        "gated_transformer_wmt_en_de_big",
        "--seed",
        "42",
        "--scoring",
        "bleu",
        "-s",
        "en",
        "-t",
        lang,
        #"--underlying-criterion",
        #"cross_entropy",
        #"--save-tensors",
        "--save-dir",
        #"/mount/arbeitsdaten48/projekte/mt/foeldeni/checkpoints/p_4_7",
        checkpoint_dir,
        "--tensorboard-logdir",
        checkpoint_dir + "/logs"
    ]
    train.cli_main()

if __name__ == "__main__":
    tallyman('tally.txt')
    if prep:
        preprocess_pruning(datapath)
    train_pruning(lr, bsz, ilr, uch, patience, lmd, lamba, pp, checkpoint_dir)
from datetime import datetime, date
import fairseq_cli.preprocess as preprocess
import fairseq_cli.train as train
import sys
import argparse

parser = argparse.ArgumentParser(description='Start the baseline NN training for MT experiments.')
parser.add_argument('--language', "-l", type=str, dest='lang',
                    choices=["de","hu", "cs"],
                    help='language argument expects one of the following languages: de, cd, hu')
parser.add_argument('--prep', "-p", action='store_true', dest="prep",
                    help='is the preprocessing step required?')
parser.add_argument('--corpus_size', "-cs", type=int, dest="datasize",
                    help='how many sentences there are in the corpus')
parser.add_argument('--model_type', "-mt", type=str, dest="model_type",
                    choices=["baseline", "pruning", "from_scratch"],
                    help='how many sentences there are in the corpus')

args = parser.parse_args()
globals().update(vars(args))
datasize = str(datasize)

today = date.today()
now = datetime.now()

datapathending = "en_" + lang
checkpointssub =  lang + "/" + datasize + '/'
model_prefix_dict = {"baseline": "bl", "pruning": "pr", "from_scratch": "fs"}
model_prefix = model_prefix_dict[model_type]


lr = "0.000025"
initial_lr = "0.0000025"
batch_sz = "4000"
updates_per_chp = "100000"
patience = '500'
temperature = "0.6"
lamba = "0.2"
prior_precision = "0.1"
experiment_dir = "/mount/arbeitsdaten/mt/foeldeni/"
datapath = experiment_dir + "fairseq/wmt18_" + datapathending + '_' + datasize
checkpoint_dir = experiment_dir + "/checkpoints/" + checkpointssub + model_prefix + '_' + today.strftime("%d_%m_%y@") + now.strftime('%H:%M')
tensor_directory = experiment_dir + "tensors/" + lang + '/' + datasize + '/' + model_prefix + '/'


def preprocess_model(text):
    old_args = sys.argv
    try:
         os.remove(f"{text}/dict.en.txt")
    except Exception:
         pass
    try:
        os.unlink("data-bin/wmt18.tokenized.en-" + lang + '_' + datasize + "/dict.en.txt")
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
                "data-bin/wmt18.tokenized.en-" + lang + "_" + datasize
                 ]

    preprocess.cli_main()

def train_model(lr, initial_lr, batch_sz, updates_per_chp, patience, checkpoint_dir, tensor_directory):
    sys.argv = [
        "fairseq-train",
        experiment_dir + "fairseq/data-bin/wmt18.tokenized.en-" + lang + '_' + datasize,
        "--warmup-updates",
        "4000",
        "--lr",
        lr,
        "--batch-size",
        batch_sz,
        "--model-parallel-size",
        "1",
        "--distributed-world-size",
        "1",
        "--warmup-init-lr",
        initial_lr,
        "--save-interval-updates",
        updates_per_chp,
        "--validate-interval-updates",
        "10000",
        "--optimizer",
        "adam",
        "--lr-scheduler",
        'inverse_sqrt',
        "--criterion",
        "cross_entropy",
        "--log-format",
        "simple",
        "--max-tokens",
        "200",
        "--best-checkpoint-metric",
        "bleu",
        "--eval-bleu",
        "--maximize-best-checkpoint-metric",
        "-s",
        "en",
        "-t",
        lang,
        "--scoring",
        "bleu",
        "--clip-norm",
        "0.1",
        "--dropout",
        "0.1",
        "--attention-dropout",
        "0.1",
        "--skip-invalid-size-inputs-valid-test",
        "--maximize-best-checkpoint-metric",
        "--arch",
        "transformer_wmt_en_de_big",
        "--seed",
        "42",
        "--patience",
        patience,
        "--device-id",
        "2",
        "--distributed-no-spawn",
        "--save-dir",
        checkpoint_dir,
        "--tensorboard-logdir",
        checkpoint_dir + "/logs",
        "--tensor-dir",
        tensor_directory
    ]
    train.cli_main()

if __name__ == "__main__":
    if prep:
        preprocess_model(datapath)
    train_model(lr, initial_lr, batch_sz, updates_per_chp, patience, checkpoint_dir, tensor_directory)
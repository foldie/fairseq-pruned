
# This is my master project code.

This code is not maintained. 
My master project is an implementation of L<sub>0</sub> regularization in the Fairseq model. The commit sha or the original commit this code is based on is 775122950d145382146e9120308432a9faf9a9b8 in the [original codebase](https://github.com/facebookresearch/fairseq). 

## Files changed

<ul>
<li> [gated_transformer.py](https://github.com/foldie/fairseq-pruned/blob/master/fairseq/models/gated_transformer.py) is based on the transformer.py file</li>
<li> [transformer_gated_layer.py](https://github.com/foldie/fairseq-pruned/blob/master/fairseq/modules/transformer_gated_layer.py) is based on the transformer_layer.py file.</li>
<li> [multihead_gated_attention.py](https://github.com/foldie/fairseq-pruned/blob/master/fairseq/modules/multihead_gated_attention.py) is based on the multihead_attention.py file</li>
<li>[gated_cross_entropy.py](https://github.com/foldie/fairseq-pruned/blob/master/fairseq/criterions/gated_cross_entropy.py) is based on the cross_entropy.py file </li>
</ul>

## Files created

### Python

<ul>
<li> [training_model.py](https://github.com/foldie/fairseq-pruned/blob/master/training_model.py) is based on </li>
<li>[tokenize_hungarian.py](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/tokenize_hungarian.py)</li>
<li>[prep_tmx.py](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/prep_tmx.py)</li>
<li>[prep_tab.py](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/prep_tab.py)</li>
</ul>

### Bash

<ul>
<li> [preprocess_and_train.sh](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/preprocess_and_train.sh) </li>
<ul>
<li>[choose_corpus.sh](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/choose_corpus.sh)</li>
<ul>
<li>[prep.sh](https://github.com/foldie/fairseq-pruned/blob/master/examples/translation/prep.sh)</li>
</ul>
</ul>
<li> [train.sh]() is based on </li>
<ul>
<li>[monitor_checkpoints.sh](https://github.com/foldie/fairseq-pruned/blob/master/monitor_checkpoints.sh) </li>
<ul>
<li>[generate_bleu_for_all_dirs](https://github.com/foldie/fairseq-pruned/blob/master/generate_bleu_for_all_dirs.sh) </li>
<ul>
<li>[generate_bleu_for_checkpoints](https://github.com/foldie/fairseq-pruned/blob/master/generate_bleu_for_checkpoint.sh) </li>
<li>[generate_chrfs](https://github.com/foldie/fairseq-pruned/blob/master/generate_chrfs.sh) is based on </li>
</ul>
</ul>
</ul>
</ul>


### Jupyter Notebooks

## Example usage

Clone this repository into your workspace with the command:
`this is the command to clone this repo`
When used for the **first time** run preprocess_and_train.sh in order to download corpus files into your directory.
First cd into the folder:
`cd `
Preprocess_and_train.sh takes the following arguments: 
<ol>
<li>target language ("de" for German, "cz" for Czech and "hu" for Hungarian)</li>
<li>maximum number of lines in the training file</li>
<li>optionally a value by which to decrement the maximum number of lines for training with various data sizes </li>
</ol>

to execute the command try running:

`examples/translation/preprocess_and_train.sh -t TARGET_LANGUAGE -m MAX_LINES`

Subsequent runs don't need to repeat the preprocessing step if they train the target language model with the same number of lines in the training data.
to restart training from an existing preprocessed data or checkpoint try running:

`examples/translation/train.sh -t TARGET_LANGUAGE -m MAX_LINES`


## L<sub>0</sub> regularization
L<sub>0</sub> regularization uses the reparameterization trick to construct a tractable alternative that approximates the combinatorial regularization term ||$\theta$||<sub>0</sub> = $\sum_{j=1}^{||\theta|}$  I[$\theta$<sub>j</sub>=0].

## Experiments

## Reproduction

Find the yaml for setting up a conda environment to run this code in [here](https://github.com/foldie/fairseq-pruned/blob/master/environment.yml).
Using a yaml to set up a conda environment can be done like this:
`conda env create -f <path_to_yaml_file>`

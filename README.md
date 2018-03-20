WARNING: this is very much a work in progress, attempt to run at your own risk :)

I will prettify these instructions once the code has converged somewhat, hopefully by end of March 2018.

Introduction:

In this repo I intend to provide a complete Pytorch port of mkusner/grammarVAE.

Thanks to https://github.com/episodeyang/grammar_variational_autoencoder for an
initial pytorch implementation, which I extended to add proper masking to the loss function,
generalize the code to work with both equations and molecules, and fix a minor bug in the decoder - and intend to extend further.

Instructions on how to install/run:

Requirements are 

* requirements.txt 
* pytorch 0.4 and visdom, both should be installed from source.
* rdkit: `conda install -c rdkit rdkit`
* `github/ZmeiGorynych/basic_pytorch`: just download the source and add to your path


How to run:
* `data_utils/make_dataset.py` creates the hd5 datasets necessary to train the models. 
Set the `molecules` boolean at its start to decide whether to generate the dataset for molecules or equations, 
and the `grammar` boolean to decide whether to encode the grammar production sequences or character sequences.
* `train/train_grammar_baseline.py` trains the model - again set the `molecules` boolean to choose the dataset (doesn't converge very well yet, need to tweak)
* `back_and_forth.py` goes the full cycle from a SMILES string to a latent space vector and back. As it's using initialized but untrained weights for now, expect the generated strings to be garbage :)
* `notebooks/PrettyPic.ipynb` draws a pretty picture of a molecule from a SMILES string

Changes made in comparison to mkusner/grammarVAE:
* Port to Python 3
* Port the neural model from Keras to PyTorch
* Refactor code to eliminate much repetition
    * move (almost) all settings specific to a particular model to `models/model_settings.py`
* Add extra masking to guarantee sequences are complete by max_len
* Port Bayesian optimization to use GPyOpt (not really tested yet)

Known issues:
* I didn't yet try terribly hard to make sure all the settings for string models (kernel sizes etc) are exactly as in
the original code. If you want to amend that, please edit `models/model_settings.py`

Extensions implemented so far:
* Add dropout to all layers of the model
* Provide alternative encoder using RNN + attention

Major things left to do:
* Tune the training of the model(s)
* Add pre-trained weights for each of the four models to the repo
* Make the non-grammar version use tokens rather than raw characters - would be a fairer comparison 
* Add some unit tests, eg for encode-decode roundtrip from string to one-hot and back for each of the four models.

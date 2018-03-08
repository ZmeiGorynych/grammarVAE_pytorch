WARNING: this is very much a work in progress, attempt to run at your own risk :)

I will prettify these instructions once the code has converged somewhat, hopefully by end of March 2018.

Introduction:

In this repo I intend to provide a complete Pytorch port of mkusner/grammarVAE.

In doing so, I of course borrowed freely from said repo, while porting the code to Python 3 and heavily refactoring to eliminate a lot of cut and paste.

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
* `data_utils/make_zinc_dataset_grammar.py` creates the hd5 dataset necessary to train the grammar model.
* `train/train_grammar_zinc_baseline.py` trains the model (doesn't converge very well yet, need to tweak)
* `back_and_forth.py` tries to go the full cycle from a SMILES string to a latent space vector and back. As it's using poorly trained weights for now, expect the generated strings to be garbage :)
* `notebooks/PrettyPic.ipynb` draws a pretty picture of a molecule from a SMILES string

Major things left to do:
* Tune the training of the model
* Port Bayesian optimization to use gpy and gpyopt
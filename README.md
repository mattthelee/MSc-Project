# MSc-Project
Final AI MSc Project using Genetic Programming and Deep Learning to play Minecraft

# Requirements
Must have a GPU, >16 Gb RAM, Java 1.8, Python 3.6.6
If running headlessly i.e. on a server without a display, then you must have xvfb or similar renderer installed. 

OpenGL libraries for Minecraft 1.11 must be installed. 

# Installation:
1. Install Python modules with `pip install -r requirements.txt`
2. Install PyTPG by following instructions here: https://github.com/Ryan-Amaral/PyTPG *** N.B. You must use version 0.8 of TPG! ***
3. Install MineRL, its dataset and dependencies following: http://minerl.io/docs/tutorials/index.html

# Running:
## TPG on MineRL:
To run the basic experiment on MineRL Navigate dense task: `python tpgAgent.py -o ./` 
N.B. Running with default settings is time consuming, therefore it is recommended to run as background job. Background task can be started on linux with: `nohup xvfb-run python tpgAgent.py > tpgLogs.out &`

If running headlessly then prepend with xvfb-run (or your chosen software renderer): `xvfb-run python tpgAgent.py`

Run `tpgAgent.py -h` to see full list of options available including environments, step limits and resuming from pretrained population

At the end of each generation, the TPG population is saved to output folder set by `-o` option. This can be resumed from this file by a subsequent run with the same parameters and the `-r` option. 

Summary results are written to a local csv file.

## TPG on Imitation Environment:
To train TPG population on the MineRL human dataset use the `-u` option followed by the number of examples to use. E.g. `python tpgAgent.py -u 10000`

A limit of 10000 is recommended to avoid memory issues. 

## VAE
To train the Variational Auto-Encoder (VAE) run: `python vae.py`. To see available options run `python vae.py -h`

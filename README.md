# MSc-Project
Final AI MSc Project using Genetic Programming and Deep Learning to play Minecraft

# Requirements
Must have a GPU, Java 1.8, Python 3.6.6
If running headlessly i.e. on a server without a display, then you must have xvfb or similar renderer installed. 

OpenGL libraries for Minecraft 1.11 must be installed. 

# Installation:
1. Install Python modules with `pip install -r requirements.txt`
2. Install PyTPG by following instructions here: https://github.com/Ryan-Amaral/PyTPG
3. Install MineRL, its dataset and dependencies following: http://minerl.io/docs/tutorials/index.html

# Running:
To run the basic experiment on MineRL Navigate dense task: `python tpgAgent.py` 

If running headlessly then prepend with xvfb-run (or your chosen software renderer): `xvfb-run python tpgAgent.py`

Run `tpgAgent.py -h` to see full list of options available including environments, step limits and resuming from pretrained population

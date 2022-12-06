#!/bin/bash
rsync --inplace --whole-file --progress -ahe ssh --exclude="dd_models/" --exclude=".*" --exclude="data/test" --exclude="coconuts" --exclude="output/" --exclude="sdgeode" --exclude="wandb" --exclude="__pycache__" --no-links --update /home/daniel/Documents/Central\ Documents/Education/MSc\ AI\ UvA/Thesis/Code/oodg-experiments/ dnobbe@login-gpu.lisa.surfsara.nl:/home/dnobbe/oodg-experiments/ | ts

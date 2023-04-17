from deep_sdf import plotting
import matplotlib
import json
import numpy as np
from random import choice
import os


with open("examples/splits/sv2_planes_lamps_train.json") as f:
    splits = json.load(f)

shape_id_choices = splits["ShapeNetV2"]["02691156"]

i = 0
while True:
    i += 1
    shape_id_1 = choice(shape_id_choices)
    shape_id_2 = choice(shape_id_choices)
    os.makedirs(f"interpolation/{i}_{shape_id_1}_{shape_id_2}", exist_ok=True)
    
    # single image takes approx. 20sec -x30-> 600sec=10min
    for j, w in enumerate(np.linspace(0.0, 1.0, 30)):
        
        fig = plotting.plot_lat_interpolation(
            exp_dir = "examples/planes_lamps",    
            shape_id_1 = shape_id_1,
            shape_id_2 = shape_id_2,
            interpolation_weight = w,
            checkpoint = 3000,
        )
        fig.savefig(f"interpolation/{i}_{shape_id_1}_{shape_id_2}/{j:06}.jpg")
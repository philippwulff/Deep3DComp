#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
from deep_sdf.data import *
from deep_sdf.mesh import *
from deep_sdf.metrics.chamfer import *
from deep_sdf.utils import *
from deep_sdf.workspace import *
from deep_sdf.lr_scheduling import *

# Required by some methods in mesh_to_sdf and pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

#!/usr/bin/env python3
import glob
import os

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = sorted(glob.glob(os.path.join(ROOT, 'models', '*_score_best.onnx')))

if not MODEL_PATHS:
    raise RuntimeError('No ONNX models found for browser inference tests.')

for model_path in MODEL_PATHS:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 2 and input_shape[0] in (None, 1):
        sample_count = input_shape[1] or 16
    else:
        sample_count = 16
    sample = np.random.randn(1, sample_count).astype(np.float32)
    outputs = session.run(None, {input_name: sample})
    if not outputs or len(outputs[0].shape) == 0:
        raise AssertionError(f'{model_path} produced an invalid output shape')
    print(os.path.basename(model_path), outputs[0].shape, float(outputs[0].ravel()[0]))

print(f'Validated {len(MODEL_PATHS)} browser-ready ONNX models.')

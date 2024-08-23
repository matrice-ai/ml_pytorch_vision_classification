"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from export_utils.models.experimental import attempt_load
from export_utils.models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from export_utils.utils.dataloaders import LoadImages
from export_utils.utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)
from export_utils.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.checks import check_requirements

from matrice_sdk.actionTracker import ActionTracker
from matrice_sdk.session import Session

MACOS = platform.system() == 'Darwin'  # macOS environment


class iOSModel(torch.nn.Module):

    def __init__(self, model, im):
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1. / w
        else:
            self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True], ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i.e., @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args.get('prefix', 'Unknown')
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            #LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            actionTracker.update_status('MDL_EXPT', 'SUCCESS', 'Model export is successful')
            return f, model
        except Exception as e:
            actionTracker.update_status('MDL_EXPT', 'ERROR', 'Error in export : ' + str(e))
            sys.exit(1)

    return outer_func


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    #LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'TorchScript model export has been acknowledged')
    f = file.with_suffix('.torchscript')
    d = {'shape': im.shape, 'stride': int(max(model.stride)), 'names': None}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()

    try:
        ts = torch.jit.trace(model, im, strict=False)
        save_func = optimize_for_mobile(ts)._save_for_lite_interpreter if optimize else ts.save
        save_func(str(f), _extra_files=extra_files)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'TorchScript model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_ACK', 'ERROR', 'Error in TorchScript model export : ' + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements('onnx>=1.12.0')
    import onnx

    #LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'ONNX Model Export has been acknowledged')
    f = str(file.with_suffix('.onnx'))

    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    try:
        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None
        )
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting ONNX model : ' + str(e))
        sys.exit(1)
    
    try:
        model_onnx = onnx.load(f)
        onnx.checker.check_model(model_onnx)
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_CHECK_FAIL', 'ERROR',  'ERROR in loading ONNX model : ' + str(e))
        sys.exit(1) 

    # Metadata
    try:
        d = {'stride': int(max(model.stride)), 'names': None}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
                cuda = torch.cuda.is_available()
                check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
                import onnxsim
    
                #LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
                actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'ONNX model has been exported')
    except Exception as e:
        #LOGGER.info(f'{prefix} simplifier failure: {e}')
        actionTracker.update_status('MDL_EXPT_SIMPLIFY_FAIL', 'ERROR', 'ERROR in simplifying ONNX model : ' + str(e))
        sys.exit(1)
    return f, model_onnx


@try_export
def export_openvino(file, metadata, half, int8, data_dir, prefix=colorstr('OpenVINO:')):
    # YOLOv5 OpenVINO export
    check_requirements('openvino-dev>=2023.0')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa
    from torchvision import datasets, transforms
    import torch
    import os

    #LOGGER.info(f'\n{prefix} starting export with openvino {ov.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'OpenVINO Model Export has been acknowledged')
    f = str(file).replace(file.suffix, f'_openvino_model{os.sep}')
    f_onnx = file.with_suffix('.onnx')
    f_ov = str(Path(f) / file.with_suffix('.xml').name)
    try:
        if int8:
            check_requirements('nncf>=2.4.0')  # requires at least version 2.4.0 to use the post-training quantization
            import nncf
            import numpy as np
            from openvino.runtime import Core

            # Define transforms for the dataset
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Load the dataset
            dataset = datasets.ImageFolder(data_dir, transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            core = Core()
            onnx_model = core.read_model(f_onnx)  # export

            def prepare_input_tensor(image: np.ndarray):
                input_tensor = image.astype(np.float32)  # uint8 to fp16/32
                input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

                if input_tensor.ndim == 3:
                    input_tensor = np.expand_dims(input_tensor, 0)
                return input_tensor

            def transform_fn(data_item):
                """
                Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
                Parameters:
                data_item: Tuple with data item produced by DataLoader during iteration
                Returns:
                    input_tensor: Input data for quantization
                """
                img = data_item[0].numpy()
                input_tensor = prepare_input_tensor(img)
                return input_tensor

            quantization_dataset = nncf.Dataset(dataloader, transform_fn)
            ov_model = nncf.quantize(onnx_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
        else:
            ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework='onnx', compress_to_fp16=half)  # export

        ov.serialize(ov_model, f_ov)  # save
        yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'OpenVino model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting OpenVINO model :' + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr('PaddlePaddle:')):
    # YOLOv5 Paddle export
    check_requirements(('paddlepaddle', 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    #LOGGER.info(f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'PaddlePaddle Model Export has been acknowledged')
    f = str(file).replace('.pt', f'_paddle_model{os.sep}')
    
    try:
        pytorch2paddle(module=model, save_dir=f, jit_type='trace', input_examples=[im])  # export
        yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'Paddle Paddle model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting PaddlePaddle model :' + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, nms, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    check_requirements('coremltools')
    import coremltools as ct

    #LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'CoreML Model Export has been acknowledged')
    f = file.with_suffix('.mlpackage')

    try:
        if nms:
            model = iOSModel(model, im)
        ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
        if bits < 32:
            if MACOS:  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print(f'{prefix} quantization only supported on macOS, skipping...')
        ct_model.save(f)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'CoreML model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting CoreML model :' + str(e))
        sys.exit(1)
    return f, ct_model


@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    try:
        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        onnx = file.with_suffix('.onnx')

        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting to ONNX model :' + str(e))
        sys.exit(1)

    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'TensorRT Model Export has been acknowledged')
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    logger.min_severity = trt.Logger.Severity.VERBOSE if verbose else trt.Logger.INFO

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    try:
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        #for inp in inputs:
        #    LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        #for out in outputs:
        #    LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if dynamic:
            if im.shape[0] <= 1:
                LOGGER.warning(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
            actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'TensorRT model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', 'ERROR in exporting TensorRT engine' +  str(e))
        sys.exit(1)
    return f, None

@try_export
def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from export_utils.models.tf import TFModel

    #LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'TensorFlow Saved Model Export has been acknowledged')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    try:
        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if keras:
            keras_model.save(f, save_format='tf')
        else:
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = tf.function(lambda x: keras_model(x))  # full model
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
            tfm.__call__(im)
            tf.saved_model.save(tfm,
                                f,
                                options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if check_version(
                                    tf.__version__, '2.6') else tf.saved_model.SaveOptions())
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'TensorFlow Saved Model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', "ERROR in exporting TensorFlow Saved Model : " + str(e))
        sys.exit(1)

    return f, keras_model


@try_export
def export_pb(keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    #LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'TensorFlow Graphdef Export has been acknowledged')
    f = file.with_suffix('.pb')

    try:
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'Tensorflow GraphDef model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', "ERROR in exporting TensorFlow GraphDef : " + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    import tensorflow as tf

    try:
        #LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'TensorFlow Graphdef Export has been acknowledged')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = str(file).replace('.pt', '-fp16.tflite')
    
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=imgsz, auto=False)
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    
        tflite_model = converter.convert()
        open(f, 'wb').write(tflite_model)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'Tensorflow Lite model has been exported')
    
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', "ERROR in exporting TensorFlow Lite : " + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr('Edge TPU:')):
    # YOLOv5 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    cmd = 'edgetpu_compiler --version'
    help_url = 'https://coral.ai/docs/edgetpu/compiler/'
    assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
    try: #check if edge tpu compiler is installed 
        if subprocess.run(f'{cmd} > /dev/null 2>&1', shell=True).returncode != 0:
            raise EnvironmentError('Edge TPU compiler not found')
    except EnvironmentError:

        #LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
        actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'Edge TPU Model Export has been acknowledged')
        sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
        for c in (
                'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
            subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)

    try : 
        #get edge tpu compiler version        
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]
    
        #LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        actionTracker.update_status('MDL_EXPT_ACK', 'ERROR', 'Error in Edge TPU model export : ' + str(e))
        f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
        f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model
        subprocess.run(['edgetpu_compiler','-s','-d','-k','10','--out_dir',str(file.parent),f_tfl, ], check=True)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'Tensorflow Edge TPU model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', "ERROR in exporting TensorFlow Edge TPU : " + str(e))
        sys.exit(1)
    return f, None


@try_export
def export_tfjs(file, int8, prefix=colorstr('TensorFlow.js:')):
    try:
        # YOLOv5 TensorFlow.js export
        check_requirements('tensorflowjs')
        import tensorflowjs as tfjs
    
        #LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...'
        actionTracker.update_status('MDL_EXPT_ACK', 'OK', 'Error in TensorFlow.js model has been acknowledged :' + str(e))
        f= str(file).replace('.pt', '_web_model')  # js dir
        f_pb = file.with_suffix('.pb')  # *.pb path
        f_json = f'{f}/model.json'  # *.json path
    
        args = [
            'tensorflowjs_converter',
            '--input_format=tf_frozen_model',
            '--quantize_uint8'if int8 else '',
            '--output_node_names=Identity,Identity_1,Identity_2,Identity_3',
            str(f_pb),
            str(f), ]
        subprocess.run([arg for arg in args if arg], check=True)
    
        json = Path(f_json).read_text()
        with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
            subst = re.sub(
                r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
                r'"Identity_1": {"name": "Identity_1"}, '
                r'"Identity_2": {"name": "Identity_2"}, '
                r'"Identity_3": {"name": "Identity_3"}}}', json)
            j.write(subst)
        actionTracker.update_status('MDL_EXPT_ACK', 'SUCCESS', 'Tensorflow.js model has been exported')
    except Exception as e:
        actionTracker.update_status('MDL_EXPT_EXPORT_FAIL', 'ERROR', "ERROR in exporting TensorFlow Edge TPU : " + str(e))
        sys.exit(1)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
    with contextlib.suppress(ImportError):
        # check_requirements('tflite_support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path('/tmp/meta.txt')
        with open(tmp_file, 'w') as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()


def pipeline_coreml(model, im, file, names, y, prefix=colorstr('CoreML Pipeline:')):
    # YOLOv5 CoreML pipeline
    import coremltools as ct
    from PIL import Image
    import time
    import platform

    print(f'{prefix} starting pipeline with coremltools {ct.__version__}...')
    batch_size, ch, h, w = list(im.shape)  # BCHW
    t = time.time()

    # YOLOv5 Output shapes
    spec = model.get_spec()
    out0, out1 = iter(spec.description.output)

    try:
        platform.system() == 'Darwin'
        #attempting to prediction on macos using coreml model 
        img = Image.new('RGB', (w, h))  # img(192 width, 320 height)
        # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
        out = model.predict({'image': img})
        out0_shape, out1_shape = out[out0.name].shape, out[out1.name].shape
    except Exception as e:  # linux and windows can not run model.predict(), get sizes from pytorch output y
        s = tuple(y[0].shape)
        out0_shape, out1_shape = (s[1], s[2] - 5), (s[1], 4)  # (3780, 80), (3780, 4)
    
    # Checks
    nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
    na, nc = out0_shape
    # na, nc = out0.type.multiArrayType.shape  # number anchors, classes
    assert len(names) == nc, f'{len(names)} names found for nc={nc}'  # check

    # Define output shapes (missing)
    out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
    out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
    # spec.neuralNetwork.preprocessing[0].featureName = '0'

    # Flexible input shapes
    # from coremltools.models.neural_network import flexible_shape_utils
    # s = [] # shapes
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
    # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
    # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
    # r.add_height_range((192, 640))
    # r.add_width_range((192, 640))
    # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

    # Print
    print(spec.description)

    # Model from spec
    model = ct.models.MLModel(spec)

    # 3. Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = 'confidence'
    nms_spec.description.output[1].name = 'coordinates'

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name  # 1x507x80
    nms.coordinatesInputFeatureName = out1.name  # 1x507x4
    nms.confidenceOutputFeatureName = 'confidence'
    nms.coordinatesOutputFeatureName = 'coordinates'
    nms.iouThresholdInputFeatureName = 'iouThreshold'
    nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    # 4. Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(input_features=[('image', ct.models.datatypes.Array(3, ny, nx)),
                                                           ('iouThreshold', ct.models.datatypes.Double()),
                                                           ('confidenceThreshold', ct.models.datatypes.Double())],
                                           output_features=['confidence', 'coordinates'])
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.versionString = 'https://github.com/ultralytics/yolov5'
    pipeline.spec.description.metadata.shortDescription = 'https://github.com/ultralytics/yolov5'
    pipeline.spec.description.metadata.author = 'glenn.jocher@ultralytics.com'
    pipeline.spec.description.metadata.license = 'https://github.com/ultralytics/yolov5/blob/master/LICENSE'
    pipeline.spec.description.metadata.userDefined.update({'classes': ','.join(names.values()),'iou_threshold': str(nms.iouThreshold),'confidence_threshold': str(nms.confidenceThreshold)})

    # Save the model
    f = file.with_suffix('.mlmodel')  # filename
    model = ct.models.MLModel(pipeline.spec)
    model.input_description['image'] = 'Input image'
    model.input_description['iouThreshold'] = f'(optional) IOU Threshold override (default: {nms.iouThreshold})'
    model.input_description['confidenceThreshold'] = \
        f'(optional) Confidence Threshold override (default: {nms.confidenceThreshold})'
    model.output_description['confidence'] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description['coordinates'] = 'Boxes × [x, y, width, height] (relative to image size)'
    model.save(f)  # pipelined
    print(f'{prefix} pipeline success ({time.time() - t:.2f}s), saved as {f} ({file_size(f):.1f} MB)')

import shutil
import os
def try_upload(actiontracker,files):

    for path in files:

        if os.path.isfile(path):
            try:
                actiontracker.upload_checkpoint(f'{path}',model_type="exported")
            except:
                print(f"Erorr in uploading {path}")
                
        elif os.path.isdir(path):
            try:
                if path.endswith("/"):
                    path=path[:-1]
                compressed_file = shutil.make_archive(
                    base_name=f'{path.split("/")[-1]}',   # archive file name w/o extension
                    format='zip',                                 # available formats: zip, gztar, bztar, xztar, tar
                    root_dir=f"{path}"                   # directory to compress
            )
                actiontracker.upload_checkpoint(compressed_file,model_type="exported")
            except:
                print(f"Erorr in uploading {path}")
                
@smart_inference_mode()
def run(action_id):
    from matrice_sdk.actionTracker import ActionTracker, LocalActionTracker

    global actionTracker
    actionTracker = ActionTracker(action_id)

    stepCode = 'MDL_EXP_ACK'
    status = 'OK'
    status_description = 'Model Export Acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode, status, status_description)

    model_config = actionTracker.get_job_params()
    print('model_config------>', model_config)
    dataset_path = model_config.get("dataset_path", "")
    print('model_config is', model_config)

    data = f"workspace/{model_config.get('dataset_path', '')}/images/train"
    weights = model_config.get("weights", f'{ROOT}/model.pt')
    imgsz = model_config.get("imgsz", (224,224))
    batch_size = model_config.get("batch", 1)
    half = model_config.get("half", False)
    inplace = model_config.get("inplace", False)
    optimize = model_config.get("optimize", False)
    keras = model_config.get("keras", False)
    int8 = model_config.get("int8", False)
    dynamic = model_config.get("dynamic", False)
    simplify = model_config.get("simplify", False)
    verbose = model_config.get("verbose", False)
    opset = model_config.get("opset", 12)
    workspace = model_config.get("workspace", 4)
    nms = model_config.get("nms", False)
    agnostic_nms = model_config.get("agnostic_nms", False)
    topk_per_class = model_config.get("topk_per_class", 100)
    topk_all = model_config.get("topk_all", 100)
    iou_thres = model_config.get("iou_thres", 100)
    conf_thres = model_config.get("conf_thres", 100)

    actionTracker.download_model(weights)

    # include = actionTracker.action_details['exportFormats']

    try:
        include = actionTracker.action_details['exportFormats']
        if include[0].lower() == "tensorrt":
            include[0] = "engine"
        device = "0" if include[0].lower() == "tensorrt" else "cpu"
    except Exception as e:
        print(f"An error occurred while processing include formats: {e}")
        sys.exit(1)

    t = time.time()
    include = [x.lower() for x in include]  # to lowercase

    try:
        fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
        flags = [x in include for x in fmts]
        assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    try:
        file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    stepCode = 'MDL_EXP_STR'
    status = 'OK'
    status_description = 'Model Export Started'
    print(status_description)
    actionTracker.update_status(stepCode, status, status_description)

    # Load PyTorch model
    try:
        device = select_device(device)

        if half and (device.type == 'cpu' or coreml):
            raise ValueError('--half only compatible with GPU export, i.e. use --device 0')
        if half and dynamic:
            raise ValueError('--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both')
        model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    except ValueError as e:
        print(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Unexpected error loading model: {e}')
        sys.exit(1)

    imgsz *= 2 if len(imgsz) == 1 else 1

    if optimize:
        assert device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    print(f"Attempting to create tensor of size: {batch_size}x3x{imgsz[0]}x{imgsz[1]}")
    tensor_size_bytes = batch_size * 3 * imgsz[0] * imgsz[1] * 4  # 4 bytes per float32
    print(f"Tensor size in GB: {tensor_size_bytes / (1024**3):.2f}")
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {'stride': int(max(model.stride)), 'names': 'string'}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning

    try:
        if jit:  # TorchScript
            f[0], _ = export_torchscript(model, im, file, optimize)
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        sys.exit(1)

    try:
        if engine:  # TensorRT required before ONNX
            f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
    except Exception as e:
        print(f"TensorRT export failed: {e}")
        sys.exit(1)

    try:
        if onnx or xml:  # OpenVINO requires ONNX
            f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    except Exception as e:
        print(f"ONNX export failed: {e}")
        sys.exit(1)

    try:
        if xml:  # OpenVINO
            f[3], _ = export_openvino(file, metadata, half, int8, data)
    except Exception as e:
        print(f"OpenVINO export failed: {e}")

    try:
        if coreml:  # CoreML
            f[4], ct_model = export_coreml(model, im, file, int8, half, nms)
            if nms:
                pipeline_coreml(ct_model, im, file, None, y)
    except Exception as e:
        print(f"CoreML export failed: {e}")
        sys.exit(1)

    try:
        if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
            assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
            assert not isinstance(model, ClassificationModel), 'ClassificationModel export to TF formats not yet supported.'
            f[5], s_model = export_saved_model(model.cpu(),
                                               im,
                                               file,
                                               dynamic,
                                               tf_nms=nms or agnostic_nms or tfjs,
                                               agnostic_nms=agnostic_nms or tfjs,
                                               topk_per_class=topk_per_class,
                                               topk_all=topk_all,
                                               iou_thres=iou_thres,
                                               conf_thres=conf_thres,
                                               keras=keras)
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = export_pb(s_model, file)
            if tflite or edgetpu:
                f[7], _ = export_tflite(s_model, im, file, int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
                if edgetpu:
                    f[8], _ = export_edgetpu(file)
                add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(s_model.outputs))
            if tfjs:
                f[9], _ = export_tfjs(file, int8)
    except Exception as ae:
        print(f"TensorFlow format assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"TensorFlow formats export failed: {e}")
        sys.exit(1)

    try:
        if paddle:  # PaddlePaddle
            f[10], _ = export_paddle(model, im, file, metadata)
    except Exception as e:
        print(f"PaddlePaddle export failed: {e}")
        sys.exit(1)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path('segment' if seg else 'classify' if cls else '')
        h = '--half' if half else ''  # --half FP16 inference arg
        s = '# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference' if cls else \
            '# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference' if seg else ''
        LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
                    f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
                    f'\nVisualize:       https://netron.app')

    try_upload(actionTracker, f)
    # return f  # return list of exported files/dirs



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    run(action_id=action_status_id)

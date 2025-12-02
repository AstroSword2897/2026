# MaxSight Model Export for iOS Deployment (Robust Version)

import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import logging

# Logging setup
logger = logging.getLogger("MaxSightExport")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def export_to_jit(
    model: nn.Module,
    save_path: str = 'maxsight_traced.pt',
    input_size: tuple = (1, 3, 224, 224)
) -> Path:
    logger.info("Exporting MaxSight to PyTorch JIT Format")

    model.eval().cpu()
    dummy_input = torch.randn(*input_size)

    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        _ = traced_model(dummy_input)

        save_path_obj = Path(save_path)
        traced_model.save(str(save_path_obj))

        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f" JIT export complete: {save_path} ({size_mb:.1f} MB)")

        return save_path_obj
    except Exception as e:
        logger.error(f"JIT export failed: {e}")
        raise


def export_to_executorch(
    model: nn.Module,
    save_path: str = 'maxsight.pte',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    logger.info("Exporting MaxSight to ExecuTorch Format")

    try:
        import executorch.exir as exir
        from executorch.extension.pybind11.portable import to_edge

        model.eval().cpu()
        dummy_input = torch.randn(*input_size)

        edge_program = to_edge(model, (dummy_input,))
        executorch_program = edge_program.to_executorch()

        save_path_obj = Path(save_path)
        with open(save_path_obj, 'wb') as f:
            f.write(executorch_program.buffer)

        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f" ExecuTorch export complete: {save_path} ({size_mb:.1f} MB)")

        return save_path_obj

    except ImportError:
        logger.warning("ExecuTorch not installed. Falling back to JIT")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)
    except Exception as e:
        logger.error(f"ExecuTorch export failed: {e}. Falling back to JIT")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)


def export_to_coreml(
    model: nn.Module,
    save_path: str = 'maxsight.mlpackage',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    logger.info("Exporting MaxSight to CoreML Format")
    try:
        import coremltools as ct
    except ImportError:
        logger.warning("coremltools not installed. Skipping CoreML export.")
        return None

    try:
        model.eval().cpu()
        dummy_input = torch.randn(*input_size)
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        test_output = traced_model(dummy_input)

        if isinstance(test_output, dict):
            output_types = [ct.TensorType(name=key, shape=val.shape)
                            for key, val in test_output.items()]
        else:
            output_types = [ct.TensorType(name="output", shape=test_output.shape)]

        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=input_size)],
            outputs=output_types,
            minimum_deployment_target=ct.target.iOS15
        )

        save_path_obj = Path(save_path)
        coreml_model.save(str(save_path_obj))

        # Correctly compute directory size
        size_mb = sum(f.stat().st_size for f in save_path_obj.rglob('*')) / (1024 * 1024)
        logger.info(f" CoreML export complete: {save_path} ({size_mb:.1f} MB)")

        return save_path_obj

    except Exception as e:
        logger.error(f"CoreML export failed: {e}")
        return None


def export_to_onnx(
    model: nn.Module,
    save_path: str = 'maxsight.onnx',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    logger.info("Exporting MaxSight to ONNX Format")
    try:
        import onnx
    except ImportError:
        logger.warning("ONNX not installed. Skipping ONNX export.")
        return None

    try:
        model.eval().cpu()
        dummy_input = torch.randn(*input_size)
        test_output = model(dummy_input)

        # Wrap dict outputs into tuple
        class DictToTupleWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                out = self.model(x)
                return tuple(out.values()) if isinstance(out, dict) else out

        wrapped_model = DictToTupleWrapper(model)
        output_names = list(test_output.keys()) if isinstance(test_output, dict) else ['output']
        dynamic_axes = {name: {0: 'batch_size'} for name in output_names}

        torch.onnx.export(
            wrapped_model,
            (dummy_input,),
            save_path,
            input_names=['image'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11
        )

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        save_path_obj = Path(save_path)
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f" ONNX export complete: {save_path} ({size_mb:.1f} MB)")

        return save_path_obj

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return None


def export_model(
    model: nn.Module,
    format: str = 'jit',
    save_dir: str = 'exports',
    input_size: tuple = (1, 3, 224, 224)
) -> dict:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)

    results = {
        'format': format,
        'exports': {},
        'metadata': {
            'input_size': input_size,
            'model_params': sum(p.numel() for p in model.parameters()),
        }
    }

    if format in ['jit', 'all']:
        results['exports']['jit'] = str(export_to_jit(model, str(save_dir_path / 'maxsight_traced.pt'), input_size))

    if format in ['executorch', 'all']:
        path = export_to_executorch(model, str(save_dir_path / 'maxsight.pte'), input_size)
        if path:
            results['exports']['executorch'] = str(path)

    if format in ['coreml', 'all']:
        path = export_to_coreml(model, str(save_dir_path / 'maxsight.mlpackage'), input_size)
        if path:
            results['exports']['coreml'] = str(path)

    if format in ['onnx', 'all']:
        path = export_to_onnx(model, str(save_dir_path / 'maxsight.onnx'), input_size)
        if path:
            results['exports']['onnx'] = str(path)

    metadata_path = save_dir_path / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f" Export metadata saved to: {metadata_path}")
    return results


if __name__ == "__main__":
    logger.info("MaxSight Export System Test")
    try:
        from ml.models.maxsight_cnn import create_model
        model = create_model().eval()
    except Exception:
        logger.warning("Could not import create_model; using dummy linear model for test.")
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*224*224, 10)).eval()

    export_model(model, format='all', save_dir='test_exports')

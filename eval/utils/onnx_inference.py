"""ONNX Runtime inference wrapper for RT-DETR models.

Handles model loading, preprocessing, inference, and output parsing.
Supports PaddleDetection RT-DETR ONNX export format:
  Inputs:  images [N,3,H,W] + orig_target_sizes [N,2]
  Outputs: labels [N,M], boxes [N,M,4] (xyxy original coords), scores [N,M]
"""

import numpy as np
import cv2
import onnxruntime as ort
import os


class OnnxDetector:
    """ONNX Runtime based object detector for RT-DETR models.

    Supports PaddleDetection RT-DETR export (labels+boxes+scores with
    built-in NMS) and auto-detects the model format.
    """

    def __init__(self, model_path, input_size=(640, 640), providers=None):
        """Initialize the detector.

        Args:
            model_path: Path to the ONNX model file.
            input_size: (height, width) expected model input size.
            providers: List of ONNX Runtime execution providers.
                       If None, auto-detects GPU > CPU.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if providers is None:
            providers = self._get_providers()

        # Register CUDA DLL directories so Python 3.8+ can find them
        self._register_cuda_dlls()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.model_path = model_path
        self.input_size = input_size  # (height, width)

        # Get input/output metadata
        self.input_info = self.session.get_inputs()
        self.output_info = self.session.get_outputs()

        # Build input name mapping
        self.input_names = {inp.name: inp for inp in self.input_info}
        self.image_input_name = self.input_info[0].name
        self.input_shape = self.input_info[0].shape

        # Check if model expects orig_target_sizes (PaddleDetection format)
        self.needs_orig_size = "orig_target_sizes" in self.input_names

        # Auto-detect input size from model if dimensions are fixed
        if self.input_shape and len(self.input_shape) == 4:
            h, w = self.input_shape[2], self.input_shape[3]
            if isinstance(h, int) and isinstance(w, int):
                self.input_size = (h, w)

        # Output metadata
        self.output_names = [o.name for o in self.output_info]
        self._output_format = self._detect_output_format()

    @staticmethod
    def _register_cuda_dlls():
        """Register CUDA toolkit DLL directories for Python 3.8+.

        On Windows, Python >= 3.8 no longer searches PATH for DLLs.
        We must explicitly add CUDA bin dirs via os.add_dll_directory().
        """
        import sys
        if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
            return

        cuda_paths = []

        # 1. Check CUDA_PATH / CUDA_HOME env vars
        for env_var in ("CUDA_PATH", "CUDA_HOME"):
            p = os.environ.get(env_var)
            if p:
                cuda_paths.append(os.path.join(p, "bin"))

        # 2. Probe standard NVIDIA GPU Computing Toolkit locations
        toolkit_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(toolkit_root):
            for ver_dir in sorted(os.listdir(toolkit_root), reverse=True):
                bin_dir = os.path.join(toolkit_root, ver_dir, "bin")
                if os.path.isdir(bin_dir):
                    cuda_paths.append(bin_dir)

        # 3. Also check cuDNN standalone installs
        cudnn_path = os.environ.get("CUDNN_PATH")
        if cudnn_path:
            cuda_paths.append(os.path.join(cudnn_path, "bin"))

        for path in dict.fromkeys(cuda_paths):          # dedupe, preserve order
            if os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                except OSError:
                    pass

    def _get_providers(self):
        """Force CUDAExecutionProvider. Falls back to CPU only if CUDA is
        completely unavailable in the ONNX Runtime build."""
        available = ort.get_available_providers()
        providers = []

        if "CUDAExecutionProvider" in available:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            ))
        else:
            print("WARNING: CUDAExecutionProvider not available – falling back to CPU")

        providers.append("CPUExecutionProvider")
        return providers

    def _detect_output_format(self):
        """Detect the output tensor format of the ONNX model.

        Returns:
            str: 'paddle_postprocessed' if model outputs labels+boxes+scores,
                 otherwise 'single_tensor', 'dual_tensor', or 'multi_tensor'.
        """
        output_name_set = set(self.output_names)

        # PaddleDetection RT-DETR with built-in NMS
        if {"labels", "boxes", "scores"}.issubset(output_name_set):
            return "paddle_postprocessed"

        num_outputs = len(self.output_info)
        if num_outputs == 1:
            return "single_tensor"
        elif num_outputs == 2:
            return "dual_tensor"
        else:
            return "multi_tensor"

    def get_param_count(self):
        """Count trainable parameters in the ONNX model.

        Loads the ONNX graph and sums the sizes of all initializers
        (weights, biases, etc.).

        Returns:
            dict with total_params, param_breakdown (by layer type),
            and human-readable param_str.  Returns None values on error.
        """
        try:
            import onnx
            from functools import reduce
            import operator
        except ImportError:
            # Try fallback location (Windows long-path workaround)
            try:
                import sys
                if r"D:\onnx_lib" not in sys.path:
                    sys.path.insert(0, r"D:\onnx_lib")
                import onnx
                from functools import reduce
                import operator
            except ImportError:
                pass
            print("  [WARN] 'onnx' package not installed. "
                  "Install with: pip install onnx")
            return {
                "total_params": None,
                "trainable_params": None,
                "param_str": "N/A (onnx not installed)",
                "param_breakdown": {},
            }

        model = onnx.load(self.model_path, load_external_data=False)

        total_params = 0
        breakdown = {}  # {component: param_count}

        for initializer in model.graph.initializer:
            numel = reduce(operator.mul, initializer.dims, 1)
            total_params += numel

            # Group by semantic component:
            #  - Named layers (e.g. "model.backbone..." → backbone)
            #  - ONNX anonymous ops (e.g. "onnx::Conv_123" → Conv)
            name = initializer.name
            if name.startswith("onnx::"):
                # Extract op type: "onnx::Conv_123" → "Conv"
                prefix = name.split("::")[1].split("_")[0]
            elif "." in name:
                # Named layers: "model.backbone.layer1..." → backbone
                parts = name.split(".")
                prefix = parts[1] if len(parts) > 1 else parts[0]
            else:
                prefix = name
            breakdown[prefix] = breakdown.get(prefix, 0) + numel

        # Human-readable string
        if total_params >= 1e6:
            param_str = f"{total_params / 1e6:.2f}M"
        elif total_params >= 1e3:
            param_str = f"{total_params / 1e3:.2f}K"
        else:
            param_str = str(total_params)

        # Sort breakdown by count descending
        breakdown = dict(
            sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "total_params": total_params,
            "trainable_params": total_params,  # ONNX has no frozen distinction
            "param_str": param_str,
            "param_breakdown": breakdown,
        }

    def get_model_info(self):
        """Get model metadata as a dictionary."""
        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        info = {
            "model_path": self.model_path,
            "file_size_mb": round(file_size_mb, 2),
            "input_names": list(self.input_names.keys()),
            "input_shape": str(self.input_shape),
            "input_size": list(self.input_size),
            "needs_orig_size": self.needs_orig_size,
            "output_names": self.output_names,
            "output_shapes": [str(o.shape) for o in self.output_info],
            "output_format": self._output_format,
            "providers": self.session.get_providers(),
        }

        # Include parameter count
        param_info = self.get_param_count()
        info["total_params"] = param_info["total_params"]
        info["param_str"] = param_info["param_str"]

        return info

    def preprocess(self, image):
        """Preprocess image for inference.

        For PaddleDetection RT-DETR: simple resize (model handles box
        coordinate mapping via orig_target_sizes input).

        Args:
            image: BGR image as numpy array (from cv2.imread).

        Returns:
            input_tensor: [1, 3, H, W] float32 tensor.
            orig_size: (height, width) of original image.
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size

        # Resize to model input size
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB, normalize to [0,1], HWC → CHW, add batch dim
        input_img = resized[:, :, ::-1].astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32)

        return input_tensor, (orig_h, orig_w)

    def inference_raw(self, input_tensor, orig_size=None):
        """Run raw ONNX inference (forward pass only).

        Args:
            input_tensor: [1, 3, H, W] float32 numpy array.
            orig_size: (height, width) of original image. Required if
                       model expects orig_target_sizes input.

        Returns:
            list of raw output numpy arrays from the model.
        """
        feed = {self.image_input_name: input_tensor}

        if self.needs_orig_size:
            if orig_size is None:
                orig_size = self.input_size  # (h, w)
            # Model expects orig_target_sizes as [width, height]
            orig_size = np.asarray(orig_size, dtype=np.int64).flatten()
            # orig_size comes in as (h, w) from preprocess → swap to (w, h)
            size_tensor = np.array([[orig_size[1], orig_size[0]]], dtype=np.int64)
            feed["orig_target_sizes"] = size_tensor

        raw_outputs = self.session.run(None, feed)
        # Return as dict keyed by output name for convenience
        return dict(zip(self.output_names, raw_outputs))

    def _parse_outputs(self, outputs, conf_threshold=0.001):
        """Parse raw model outputs into boxes, scores, and class IDs.

        Args:
            outputs: dict of {output_name: np.array} from inference_raw().

        Returns:
            boxes: [M, 4] in xyxy format (original image coordinates).
            scores: [M] confidence scores.
            class_ids: [M] class indices (1-based for VisDrone RT-DETR).
        """
        if self._output_format == "paddle_postprocessed":
            # PaddleDetection: labels [N,M], boxes [N,M,4], scores [N,M]
            labels = outputs["labels"][0]   # [M] int64
            boxes = outputs["boxes"][0]     # [M, 4] xyxy in original coords
            scores = outputs["scores"][0]   # [M] float32

            mask = scores > conf_threshold
            return boxes[mask], scores[mask], labels[mask].astype(np.int32)

        else:
            # Fallback for ultralytics or other formats
            from .postprocess import xywh_to_xyxy, multiclass_nms
            output = list(outputs.values())[0]
            if output.ndim == 3:
                output = output[0]
            raw_boxes = output[:, :4]
            raw_scores = output[:, 4:]
            boxes_xyxy = xywh_to_xyxy(raw_boxes)
            return multiclass_nms(boxes_xyxy, raw_scores, conf_threshold=conf_threshold)

    def predict(self, image, conf_threshold=0.001, nms_threshold=0.7, max_detections=300):
        """Full prediction pipeline: preprocess → inference → parse outputs.

        For PaddleDetection models, NMS is built into the model so
        nms_threshold and max_detections are handled internally.

        Args:
            image: BGR image as numpy array.
            conf_threshold: minimum confidence to keep.
            nms_threshold: NMS IoU threshold (only for non-paddle formats).
            max_detections: max detections (only for non-paddle formats).

        Returns:
            boxes: [M, 4] xyxy format in original image coordinates.
            scores: [M] confidence scores.
            class_ids: [M] class indices (1-based for VisDrone RT-DETR).
        """
        input_tensor, orig_size = self.preprocess(image)
        outputs = self.inference_raw(input_tensor, orig_size)
        boxes, scores, class_ids = self._parse_outputs(outputs, conf_threshold)

        # Clip boxes to image bounds
        if len(boxes) > 0:
            orig_h, orig_w = orig_size
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, scores, class_ids

    def warmup(self, iterations=10):
        """Warmup the model with random dummy inputs.

        Args:
            iterations: number of warmup forward passes.
        """
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        for _ in range(iterations):
            self.inference_raw(dummy, orig_size=self.input_size)

"""


Used for converting a TensorFlow SavedModel to a TensorRT-optimized model using TF-TRT.
You need to have NVIDIA TensorRT and TensorFlow with GPU support installed on your device to run this.
"""

import tensorflow as tf
import os

def export_trt(saved_model_dir, trt_model_dir, precision_mode='FP16'):
    """
    Convert a TF SavedModel to TensorRT-optimized SavedModel.
    precision_mode can be 'FP32', 'FP16', or 'INT8' (requires calibration).
    """
    params = tf.experimental.tensorrt.Converter.TftrtConversionParams(
        precision_mode=precision_mode,
        max_workspace_size_bytes=1 << 30,  # 1GB, adjust as needed based on your compute situation
        maximum_cached_engines=100
    )

    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params
    )
    converter.convert()

    # Optionally build TensorRT engines ahead of time by specifying an input function
    # For example, if your model expects 224x224x3 images:
    def my_input_fn():
        # yield batched input data (shape [N, height, width, 3])
        input_shape = (1, 224, 224, 3)
        yield [tf.ones(input_shape, tf.float32)]

    converter.build(input_fn=my_input_fn)

    # Save
    converter.save(trt_model_dir)
    print(f"TensorRT-optimized model saved to: {trt_model_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export TF model to TensorRT-optimized SavedModel.")
    parser.add_argument("--input_model", required=True, help="Path to the original SavedModel directory.")
    parser.add_argument("--output_model", default="trt_saved_model", help="Path to save the TRT-optimized SavedModel.")
    parser.add_argument("--precision", default="FP16", help="Precision mode: FP32, FP16, or INT8.")
    args = parser.parse_args()

    os.makedirs(args.output_model, exist_ok=True)
    export_trt(args.input_model, args.output_model, args.precision)

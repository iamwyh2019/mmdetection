import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Initialize TensorRT engine and context
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_file_path):
    """
    This function builds a TensorRT engine from an ONNX file.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    builder.max_batch_size = 1
    engine = builder.build_engine(network, config)
    return engine

def run_inference(engine, input_data):
    """
    This function runs inference on the TensorRT engine.
    """
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': cuda_mem})
        else:
            outputs.append({'host': host_mem, 'device': cuda_mem})
    
    # Transfer input data to the GPU.
    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # Run inference.
    context = engine.create_execution_context()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    return outputs[0]['host']

def main():
    onnx_model_path = 'sdk/rtmdet-l-sdk/end2end.onnx'
    input_data = 'demo/demo.jpg'
    input_image = cv2.imread(input_data)

    engine = build_engine(onnx_model_path)
    if engine:
        output = run_inference(engine, input_image)
        print('Inference output:', output)

if __name__ == '__main__':
    main()

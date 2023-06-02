import onnx
import onnxsim
model_onnx = onnx.load("smoke_dla34.onnx")  # load onnx model
model_onnx, check = onnxsim.simplify(model_onnx)
onnx.save(model_onnx, "sim.onnx")

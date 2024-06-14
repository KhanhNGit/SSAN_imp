from networks import *
import torch

max_iter = 200 * 1
model = get_model(max_iter, 3)
model.load_state_dict(torch.load('.weights/Z_N_C_to_P_best.pth')['state_dict'])
model.to('cuda')
model.eval()

dummy_input = torch.randn(1, 3, 256, 256).to('cuda')
onnx_model = torch.onnx.export(model, (dummy_input, dummy_input), '.weights/best_model.onnx', opset_version=12)
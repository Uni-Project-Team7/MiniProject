import torch
import torch.nn as nn
from torchviz import make_dot
from encodingToArch import decode_and_build_unet
# Define a simple PyTorch model


# Create the model and a dummy input
with torch.no_grad():
    model = decode_and_build_unet([1, 8, 5, 8, 6, 0, 6, 1, 4, 3, 9, 0], 64).cuda()
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
# Forward pass to create the graph
    output = model(dummy_input)

# Generate the computational graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the graph to a file
dot.format = "png"
dot.render("model[1, 4, 5, 8, 7, 3, 6, 1, 4, 3, 9, 0]")

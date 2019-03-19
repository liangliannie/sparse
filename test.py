import torch
import sparseconvnet as scn

# Use the GPU if there is one, otherwise CPU
use_gpu = torch.cuda.is_available()

model = scn.Sequential().add(
    scn.SparseVggNet(2, 1,
		     [['C',  8], ['C',  8], ['MP', 3, 2],
		      ['C', 16], ['C', 16], ['MP', 3, 2],
		      ['C', 24], ['C', 24], ['MP', 3, 2]])
).add(
    scn.SubmanifoldConvolution(2, 24, 32, 3, False)
).add(
    scn.BatchNormReLU(32)
).add(
    scn.SparseToDense(2,32)
)
if use_gpu:
    model.cuda()

# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
input = scn.InputBatch(2, inputSpatialSize)

msg = [
    " X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
    " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
    " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
    " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
    " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "]

#Add a sample using set_location
input.add_sample()
for y, line in enumerate(msg):
    for x, c in enumerate(line):
        if c == 'X':
            location = torch.LongTensor([x, y])
            featureVector = torch.FloatTensor([1])
            input.set_location(location, featureVector, 0)

#Add a sample using set_locations
input.add_sample()
locations = []
features = []
for y, line in enumerate(msg):
    for x, c in enumerate(line):
        if c == 'X':
            locations.append([x,y])
            features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features)
input.set_locations(locations, features, 0)

model.train()
if use_gpu:
    input.cuda()
output = model.forward(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print(output.size(), output.type())
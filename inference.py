import numpy as np
from openvino.inference_engine import IECore

model = 'mnist-frozen-ir/mnist-frozen'
#model = 'savedmodel-ir/saved_model'

ie = IECore()
net = ie.read_network(model=model+'.xml', weights=model+'.bin')
input_name    = next(iter(net.input_info))
output_name   = next(iter(net.outputs))
print('Input node name=', input_name, ' Output node name=', output_name)
batch, w = net.input_info[input_name].tensor_desc.dims
print('Input shape = ', net.input_info[input_name].tensor_desc.dims)
exec_net = ie.load_network(network=net, device_name='GNA', num_requests=1)

mnistLabels=np.frombuffer(open('MNIST_data/t10k-labels-idx1-ubyte','rb').read(), dtype=np.uint8, offset=8)
mnistImages=np.frombuffer(open('MNIST_data/t10k-images-idx3-ubyte','rb').read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)

right=0
num=0
for label, img in zip(mnistLabels, mnistImages):
	img = img.astype(np.float32).reshape(1,28*28)
	img /= 255.0
	result = exec_net.infer(inputs={input_name: img})
	correct=label                                  # correct answer (label)
	infered=np.argmax(result[output_name])         # infered answer
	if correct == infered:
		print('.', end='')
		right+=1
	else:
		print('X', end='')
	if num % 50==49:
		print()
	num+=1
print('{} / {} : {} %'.format(right, num, (right/num)*100))

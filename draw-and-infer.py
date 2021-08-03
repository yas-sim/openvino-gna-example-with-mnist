import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

model = 'mnist-frozen-ir/mnist-frozen'
#model = 'savedmodel-ir/saved_model.xml'

win_name = 'Draw and Infer'
ratio = 20
size = 28*ratio
caption_size = ratio * 4
frame = np.zeros((size, size, 3), dtype=np.uint8)

def onMouse(event, x, y, flags, param):
	global frame
	pen_img = np.zeros(frame.shape, dtype=np.uint8)
	cv2.circle(pen_img, (x,y), ratio, (255,255,255), -1)
	if event==cv2.EVENT_MOUSEMOVE:                       # Mouse move event 
		if flags and cv2.EVENT_FLAG_LBUTTON:                 # Left button is pressing down
			frame |= pen_img                                   # Draw a filled circle
	elif event==cv2.EVENT_RBUTTONDOWN:                   # Right button down event
		frame = np.zeros((size, size, 3), dtype=np.uint8)    # Frame clear
	tmpimg = frame | pen_img
	cv2.imshow(win_name, tmpimg)


def main():
	global frame

	ie = IECore()
	net = ie.read_network(model=model+'.xml', weights=model+'.bin')
	input_name    = next(iter(net.input_info))    # input node name of the DL model
	output_name   = next(iter(net.outputs))   # output node name of the DL model
	print('Input node name=', input_name, ' Output node name=', output_name)
	batch, w = net.input_info[input_name].tensor_desc.dims
	print('Input shape = ', net.input_info[input_name].tensor_desc.dims)
	exec_net = ie.load_network(network=net, device_name='GNA', num_requests=1)
	del net

	cv2.namedWindow(win_name)
	cv2.setMouseCallback(win_name, onMouse, param=None)
	cv2.imshow(win_name, frame)

	while(cv2.waitKey(100)!=27):   # 27==ESC key
		# Image preprocess - shrink and convert to single channel image (monochrome)
		shrank_img = cv2.resize(frame, (28, 28))   # 28x28
		input_img, _, _ = cv2.split(shrank_img)

		result = exec_net.infer(inputs={input_name: input_img.ravel()})    # Infer
		result = result[output_name][0]
		result /= result.max()

		# Draw inference score bar chart
		caption = np.zeros((caption_size, size, 3), dtype=np.uint8)
		for i in range(10):
			prob = result[i]
			x1 = int((i+1)*(size/12))
			y1 = int(caption_size-prob*caption_size)
			x2 = int((i+1.5)*(size/12))
			y2 = int(caption_size)
			cv2.rectangle(caption, (x1, y1), (x2, y2), (255,0,0), -1)
			cv2.putText(caption, str(i), (x1, caption_size-ratio), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
		caption[:28,:28,:]=shrank_img
		cv2.imshow('score', caption)

if __name__ == '__main__':
	main()

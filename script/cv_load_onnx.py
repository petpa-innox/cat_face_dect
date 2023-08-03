import cv2
import numpy as np
from sewar.full_ref import uqi
# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1.0/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], swapRB=True,crop=False)
	# print(blob.shape)
	# img_blob =  blob[0].reshape(blob.shape[2],blob.shape[3],blob.shape[1])
	# img_blob = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
	# img_blob = img_blob.reshape(img_blob.shape[1],img_blob.shape[1],3)
	# cv2.imshow("blob",img_blob)
	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []
	nms_boxes = []
	# Rows.
	rows = outputs[0].shape[1]

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		nms_boxes.append([boxes[i],class_ids[i],confidences[i]])
	return nms_boxes

def get_vertex(box,shape):
	p_width = shape[0]
	p_height= shape[1] 
	left = box[0]
	top = box[1]
	width = box[2]
	height = box[3]
	right = left +width
	down = top +height
	left = max(0,min(shape[1],left))
	right = max(0,min(shape[1],right))#
	top = max(0,min(  shape[0],top))#
	down = max(0,min(shape[0],down))#
	return left,top,right,down

def pic_size_pad(input_img,target_size=224):
	img_ratio = target_size/max(input_img.shape[:2])
	input_img = cv2.resize(input_img,(0,0),fx=img_ratio,fy=img_ratio)
	delta_w = target_size - input_img.shape[1]
	delta_h = target_size - input_img.shape[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)
	out_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
	return out_img,input_img

modelWeights = "weight/second_train.onnx"
cat_face_path = ''
vdieo_path = 'data/video/2.mp4'
img_path  = 'cat.jpg'
std_cat_face = 'cf_0_0.jpg'

if __name__ == '__main__':
	classes = ['cat']
	cat_std = cv2.imread(std_cat_face)
	shape = (224,224)
	cat_std = cv2.resize(cat_std,shape)

	cap = cv2.VideoCapture(vdieo_path)
	net = cv2.dnn.readNet(modelWeights)
	if cv2.cuda.getCudaEnabledDeviceCount()>0:	
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	iter_ = 0
	while (cap.isOpened()):
		ret, frame = cap.read()
		# frame = cv2.imread(img_path)
		if ret == True:
			# # Give the weight files to the model and load the network using them.
			# # Process image.
			detections = pre_process(frame, net)
			boxes = post_process(frame, detections)
			
			for i, [box,class_id,confidences] in enumerate(boxes):
				left,top,right,down = get_vertex(box,frame.shape)
				head_img = frame[top:down,left:right]#height(y),width(x)
				head_img_pad,head_img = pic_size_pad(head_img)

				head_img = cv2.resize(head_img,shape)
				similarity = uqi(cat_std,head_img)
				label_1 = "{}:{:.2f}".format('sml', similarity)
				draw_label(head_img, label_1, 0, 0)
				cv2.imshow('head', head_img)
				# cv2.imwrite(cat_face_path+'/cf_{}_{}.jpg'.format(iter_,i),head_img)
				cv2.rectangle(frame, (left, top), (right, down), BLUE, 3*THICKNESS)
				label = "{}:{:.2f}".format(classes[class_id], confidences)
				draw_label(frame, label, left, top)
				
			# # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
			t, _ = net.getPerfProfile()
			label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
			# print(label)
			cv2.putText(frame, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
			cv2.imshow('Frame', frame)
			iter_+=1
			# 按q退出
			key = cv2.waitKey(1)
			if key== ord('q'):
				break
			elif key ==ord('n'):
				continue
			# break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()



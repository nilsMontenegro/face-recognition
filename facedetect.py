import cv2
import numpy
import sys

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def enlarge_rect(startX, startY, endX, endY, w, h):
	lenX = (endX - startX) / 4
	lenY = (endY - startY) / 4

	startX = int(max(startX - lenX, 0))
	startY = int(max(startY - lenY, 0))
	endX = int(min(endX + lenX, w))
	endY = int(min(endY + lenY, h))

	return (startX, startY, endX, endY)

def detect_face(filename):
	image = cv2.imread(filename)
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))


	net.setInput(blob)

	detections = net.forward()

	detections_array = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
		
		detections_array.append((confidence, box))

	print(detections_array[0])
	#detections_array.sort()
	detections_array = sorted(detections_array, key=lambda tup: tup[0])

	for (confidence, box) in detections_array[-1:]:
		(startX, startY, endX, endY) = box.astype("int")
		(startX_large, startY_large, endX_large, endY_large) = enlarge_rect(startX, startY, endX, endY, w, h)
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.rectangle(image, (startX_large, startY_large), (endX_large, endY_large),(0, 255, 0), 2)
		cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Output", image)
	cv2.waitKey(0)

for filename in sys.argv[1:]:
	print(filename)
	detect_face(filename)


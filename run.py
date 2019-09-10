# Reference: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Reference: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import tensorflow as tf
import numpy as np
import keras
import cv2
import os

newdir = 'graded_images' 

def load_model_VGG_trained():
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    new_output = model.output
    new_output = keras.layers.Flatten()(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(11, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    
    model.load_weights('model.h5')
    print("Loaded model from disk")
    
    #model.summary()
    model.compile(keras.optimizers.Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_own():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')
    ])
    
    model.load_weights('model.h5')
    print("Loaded model from disk")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    
    return model


def pyramid(img, scale, minSize):
    image = img.copy()
    yield (0,image)
    
    i = 0
    while True:
        i = i+1
        h = int(image.shape[0] / scale)
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, h))

        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield (i,image)

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
def non_max_suppression(boxes, overlapThresh):
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick]

            
def bounding_boxes(model, image, scale):
    bounding_boxes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    
    aspect_ratio = image.shape[0]/image.shape[1]
    if aspect_ratio <= 1:
        img = cv2.resize(image.copy(), (int(256*aspect_ratio), 256))
        kernel = (9,9)
    else:
        img = cv2.resize(image.copy(), (int(128*aspect_ratio), 256))
        kernel = (3,3)
        
    h_change = (image.shape[0]/img.shape[0])
    w_change = (image.shape[1]/img.shape[1])
        
    img = cv2.GaussianBlur(img.copy(),kernel,cv2.BORDER_DEFAULT)   
    
    for (scale_factor, resized) in pyramid(img, scale, (30, 30)):
        #print(resized.shape)
        for (x, y, window) in sliding_window(resized,stepSize=5, windowSize=(32, 32)):
            if window.shape[0] != 32 or window.shape[1] != 32:
                continue
            a = np.expand_dims(window, axis=0)
            a = a*(1 / 255.)
            pred = model.predict(a)
            label = np.argmax(pred)
            prob = np.max(pred)
            if prob > 0.999 and label != 10:
                rescale = pow(scale, scale_factor)
                x = int(x*rescale*w_change*1.1)
                y = int(y*rescale*h_change)
                size = int(32*rescale)
                bounding_boxes[int(label)].append((x, y, x + int(size*w_change*0.8), y + int(size*h_change), np.max(pred)))
    return bounding_boxes


def main():
    model = load_model_own()
    for i in range(1,7):
        scale = 1.1
        image = cv2.imread('images/' + str(i) + '.png')
        im = image.copy()
        
        boxes = bounding_boxes(model, image, scale)
        
        total_bboxes = {}
        for label in boxes:
            total_bboxes[label] = non_max_suppression(np.array(boxes[label]), 0.25)
        
        index = 0
        for label in total_bboxes:
            for box in total_bboxes[label]:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                prob = box[4]
                
                #print(label, prob)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 4)
                if index == 0:
                    font_size = round((x2-x1)*0.5/40, 2)
                cv2.putText(im,str(label) ,(x1, y1+10),cv2.FONT_HERSHEY_SIMPLEX,font_size,(0,0,255),4)
                cv2.imwrite(os.path.join(newdir, str(i) +'.png'), im)
                index = index+1
        print(str(i)  + '.png saved to graded_images!')

if __name__ == '__main__':
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    main()
        

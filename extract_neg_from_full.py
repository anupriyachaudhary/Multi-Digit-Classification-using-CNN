import h5py
import os
import numpy as np
import cv2
from random import randint

class SVHNnegative:
    
    def __init__(self, output_dir):
        self.h5File = None
        self.digit_struct_name = None
        self.digit_struct_bbox = None
        self.output_dir = output_dir

    def get_bounding_box(self, index):
        bbox = {}
        n = self.digit_struct_bbox[index].item()

        bbox['label'] = self.get_bbox(self.h5File[n]["label"])
        bbox['top'] = self.get_bbox(self.h5File[n]["top"])
        bbox['left'] = self.get_bbox(self.h5File[n]["left"])
        bbox['height'] = self.get_bbox(self.h5File[n]["height"])
        bbox['width'] = self.get_bbox(self.h5File[n]["width"])
        return bbox
    
    def get_image_filename(self, index):
        name = ''.join([chr(c[0]) for c in self.h5File[self.digit_struct_name[index][0]].value])
        return name
    
    def get_bbox(self, attribute):
        if len(attribute) > 1:
            attr = [self.h5File[attribute.value[j].item()].value[0][0] for j in range(len(attribute))]
        else:
            attr = [attribute.value[0][0]]
        return attr

    def get_digit_struct(self, index):
        struct = self.get_bounding_box(index)
        struct['name'] = self.get_image_filename(index)
        return struct

    def get_images_and_label_data(self):
        structs = []
        for i in range(len(self.digit_struct_name)):
            struct = self.get_digit_struct(i)
            if len(struct["label"]) <= 4:
                structs.append(struct)
        return structs

    def save_h5File_data(self, data, labels, name):
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), "w")
        h5f.create_dataset(name + "_images", data=data)
        h5f.create_dataset(name + "_labels", data=labels)

    def read_mat_file(self, data_dir):
        self.h5File = h5py.File(os.path.join(data_dir, "digitStruct.mat"), 'r')
        self.digit_struct_name = self.h5File['digitStruct']['name']
        self.digit_struct_bbox = self.h5File['digitStruct']['bbox']
        structs = self.get_images_and_label_data()

        return structs

    def process_digitStructFile(self, data_dir):
        structs = self.read_mat_file(data_dir)

        image_data = []
        labels = []
        index=0
        for i in range(len(structs)):
            file_name = os.path.join(data_dir, structs[i]['name'])
            top = structs[i]['top']
            left = structs[i]['left']
            height = structs[i]['height']
            width = structs[i]['width']
            
            bbox_whole = self.get_whole_bounding_box(file_name, top, left, height, width)
            isNeg, neg_image = self.create_negative_image(file_name, bbox_whole)
            if isNeg == True:
                image_data.append(neg_image)
                labels.append(10)
                index = index+1
                print(str(index))
        print("neg_images:", str(index))

        return np.array(image_data), np.array(labels)
    
    def get_whole_bounding_box(self, file_name, top, left, height, width):
        image = cv2.imread(file_name)

        image_top = np.amin(top)
        image_left = np.amin(left)
        image_height = np.amax(top) + height[np.argmax(top)] - image_top
        image_width = np.amax(left) + width[np.argmax(left)] - image_left

        box_left = np.amax([np.floor(image_left),0])
        box_top = np.amax([np.floor(image_top),0])
        box_right = np.amin([np.ceil(box_left + image_width), image.shape[1]])
        box_bottom = np.amin([np.ceil(image_top + image_height), image.shape[0]])
        
        return (int(box_left), int(box_top), int(box_right), int(box_bottom)) 
    
    def create_negative_image(self, file_name, bbox):
        isNeg = False
        isResize = False
        
        image = cv2.imread(file_name)

        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]
        
        v_overlap = 0.2*(box_bottom - box_top)
        h_overlap = 0.2*(box_right - box_left)
        
        case = randint(1, 8)
        
        if case == 1:
            bottom = int(box_top+v_overlap)
            right = int(box_left+h_overlap)
            top = bottom - 32
            left = right - 32
            if top > 0 and left > 0:
                isNeg = True
                
        elif case == 2:
            top = int(box_bottom-v_overlap)
            left = int(box_right-h_overlap)
            bottom = top + 32
            right = left + 32
            if bottom < image.shape[0] and right < image.shape[1]:
                isNeg = True
                
        elif case == 3:
            left = int(box_right-h_overlap)
            bottom = int(box_top+v_overlap)
            right = left+32
            top = bottom - 32
            
            if top > 0 and right < image.shape[1]:
                isNeg = True
                
        elif case == 4:
            top = int(box_bottom-v_overlap)
            right = int(box_left+h_overlap)
            left = right - 32
            bottom = top + 32
            if left > 0 and bottom < image.shape[0]:
                isNeg = True
                
        elif case == 5:
            left = box_left
            bottom = int(box_top+v_overlap)
            top = bottom-32
            right = left+32
            if top > 0 and right < image.shape[1]:
                isNeg = True
                
        elif case == 6:
            right = box_right
            top = int(box_bottom-v_overlap)
            bottom = top+32
            left = right-32
            if left > 0 and bottom < image.shape[0]:
                isNeg = True
                
        elif case == 7:
            right = int(box_left+h_overlap)
            bottom = box_bottom
            top = bottom-32
            left = right-32
            if top > 0 and left > 0:
                isNeg = True
        
        else:
            top = box_top
            left = int(box_right-h_overlap)
            bottom = top+32
            right = left+32
            if bottom < image.shape[0] and right < image.shape[1]:
                isNeg = True
        
        if isNeg == False:
            height = (box_bottom - box_top)
            right_margin = image.shape[1] - box_right
            left_margin = box_left
            
            if left_margin > right_margin and left_margin >= 33:
                top = box_top
                left = box_left - 32
                bottom = top + height
                right = box_left
                isNeg = True
                isResize = True
            elif left_margin > right_margin and left_margin < 33:
                top = box_top
                left = box_left - left_margin - 1
                bottom = top + height
                right = box_left
                isNeg = True
                isResize = True
            elif right_margin >= left_margin and right_margin >= 33:
                top = box_top
                left = box_right
                bottom = top + height
                right = left + 32
                isNeg = True
                isResize = True
            elif right_margin >= left_margin and right_margin < 33:
                top = box_top
                left = box_right
                bottom = top + height
                right = left + right_margin - 1
                isNeg = True
                isResize = True
            
        if isNeg == True:
            neg_image = image[top:bottom, left:right]
            if isResize == True:
                if neg_image.shape[0] > 8 and neg_image.shape[1] > 8:
                    neg_image = cv2.resize(neg_image, (32,32))
                else:
                    isNeg = False
            if neg_image.shape[0] != 32 and neg_image.shape[1] != 32:
                isNeg = False
            
        else:
            neg_image = ""

        return (isNeg, neg_image)


def main():
    svhn = SVHNnegative("data/cropped")

    # Train dataset
    train_neg_data, train_neg_labels = svhn.process_digitStructFile("data/full/train")
    svhn.save_h5File_data(train_neg_data, train_neg_labels, "train_neg")

    # Test dataset
    test_neg_data, test_neg_labels = svhn.process_digitStructFile("data/full/test")
    svhn.save_h5File_data(test_neg_data, test_neg_labels, "test_neg")


if __name__ == '__main__':
    main()

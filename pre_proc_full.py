import h5py
import os
import numpy as np
import cv2

class SVHNfull:
   
    def __init__(self, output_dir):
        self.h5File = None
        self.digit_struct_name = None
        self.digit_struct_bbox = None
        self.output_dir = output_dir

        self.NUM_LABELS = 11
        self.MAX_NUMBERS = 4
        self.MAX_LABELS = 5

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

    def get_digit_struct(self, n):
        struct = self.get_bounding_box(n)
        struct['name'] = self.get_image_name(n)
        return struct

    def get_images_and_labels_data(self):
        structs = []
        for i in range(len(self.digit_struct_name)):
            struct = self.get_digit_struct(i)
            if len(struct["label"]) <= self.MAX_NUMBERS:
                structs.append(struct)
        return structs

    def save_h5_data(self, data, labels, name):
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), "w")
        h5f.create_dataset(name + "_dataset", data=data)
        h5f.create_dataset(name + "_labels", data=labels)

    def read_mat_file(self, data_dir):
        self.h5File = h5py.File(os.path.join(data_dir, "digitStruct.mat"), 'r')
        self.digit_struct_name = self.h5File['digitStruct']['name']
        self.digit_struct_bbox = self.h5File['digitStruct']['bbox']
        structs = self.get_images_and_labels_data()

        return structs

    def extact_process_file(self, data_dir):
        structs = self.read_mat_file(data_dir)
        images_count = len(structs)

        image_data = []
        labels = []

        index=0
        no_of_neg_images = 0
        for i in range(images_count):
            lbls = structs[i]['label']
            file_name = os.path.join(data_dir, structs[i]['name'])
            top = structs[i]['top']
            left = structs[i]['left']
            height = structs[i]['height']
            width = structs[i]['width']
            
            bbox_whole = self.get_whole_bounding_box(file_name, top, left, height, width)
            labels.append(self.create_new_label(lbls))
            image_data.append(self.create_image_array(file_name, bbox_whole))
            index = index + 1
            
            isNeg, neg_image, label = self.create_negative_image(file_name, bbox_whole)
            if isNeg == True:
                image_data.append(neg_image)
                labels.append(label)
                index = index+1
                no_of_neg_images = no_of_neg_images + 1
            print(str(i))
            i=i+1
        print("neg_images:", str(no_of_neg_images))

        return np.array(image_data), np.array(labels)

    def create_new_label(self, labels):
        num_digits = len(labels)
        new_label = [10,10,10,10,0]

        for i in range(num_digits):
            digit = labels[i]
            if int(digit) == 10:
                new_label[i] = 0
            new_label[i] = digit
        
        if num_digits > 0:
            new_label[4] = num_digits
            
        return np.array(new_label)

    def create_image_array(self, file_name, bbox):
        # Load image
        image = cv2.imread(file_name)

        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]

        crop_img = image[box_top:box_bottom, box_left:box_right]
        resized_image = cv2.resize(crop_img, (64, 64))

        return resized_image
    
    def get_whole_bounding_box(self, file_name, top, left, height, width):
        image = cv2.imread(file_name)

        image_top = np.amin(top)
        image_left = np.amin(left)
        image_height = np.amax(top) + height[np.argmax(top)] - image_top
        image_width = np.amax(left) + width[np.argmax(left)] - image_left

        box_left = np.amax([np.floor(image_left - 0.1 * image_width),0])
        box_top = np.amax([np.floor(image_top - 0.1 * image_height),0])
        box_right = np.amin([np.ceil(box_left + 1.2 * image_width), image.shape[1]])
        box_bottom = np.amin([np.ceil(image_top + 1.2 * image_height), image.shape[0]])
        
        return (int(box_left), int(box_top), int(box_right), int(box_bottom)) 
    
    def create_negative_image(self, file_name, bbox):
        isNeg = False
        neg_image = np.zeros(1)
        labels_array = [10,10,10,10,0]
        
        # Load image
        image = cv2.imread(file_name)

        box_left = bbox[0]
        box_top = bbox[1]
        
        if box_left > 70 and (image.shape[0]-box_top) > 70:
            top = box_top + 1
            bottom = top + 64
            left = box_left - 64 -1
            right = left + 64
            neg_image = image[top:bottom, left:right]
            isNeg = True

        return (isNeg, neg_image, np.array(labels_array))


def main():
    svhn = SVHNfull("data")

    # Train dataset
    train_data, train_labels = svhn.extact_process_file("data/train")
    svhn.save_h5_data(train_data, train_labels, "train")

    # Test dataset
    test_data, test_labels = svhn.extact_process_file("data/test")
    svhn.save_h5_data(test_data, test_labels, "test")

    # Extra dataset
    #extra_data, extra_labels = svhn.extact_process_file("data/extra")
    #svhn.save_data(extra_data, extra_labels, "extra")


if __name__ == '__main__':
    main()

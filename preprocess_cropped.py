import h5py
import os
import numpy as np
from scipy.io import loadmat
import random

class SVHNsingle:
    
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save_h5_data(self, data, labels, name):
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), 'w')
        h5f.create_dataset(name + "_images", data=data)
        h5f.create_dataset(name + "_labels", data=labels)
        h5f.close()

    def load_h5_file(self, name):
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), "r")
        data = h5f[name + "_images"][:]
        labels = h5f[name + "_labels"][:]

        return data, labels

    def load_mat_file(self, data_dir, name):
        data = loadmat(data_dir + name + "_32x32.mat")
        return data['X'].transpose((3,0,1,2)), data['y'][:,0]
    
    def shuffle_data(self, images, labels):
        indices = [i for i in range(labels.shape[0])]
        random.shuffle(indices)
        return (images[indices,:,:,:], labels[indices,])
        
    def extract_process_file(self, data_dir, name):
        pos_images, pos_labels = self.load_mat_file(data_dir, name)
        pos_labels[pos_labels == 10] = 0
        
        neg_images, neg_labels = self.load_h5_file(name + "_neg")
        
        if name == "train":
            pos_extra_images, pos_extra_labels = self.load_mat_file(data_dir, "extra")
            pos_extra_labels[pos_extra_labels == 10] = 0
            
            size = int(pos_extra_labels.shape[0]*0.25)
            
            pos_extra_images = pos_extra_images[:size,:,:,:]
            pos_extra_labels = pos_extra_labels[:size,]
            
            images = np.vstack((np.vstack((pos_images, neg_images)), pos_extra_images))
            labels = np.concatenate([pos_labels, neg_labels, pos_extra_labels])
        else:
            images = np.vstack((pos_images, neg_images))
            labels = np.concatenate([pos_labels, neg_labels])
            
        images, labels = self.shuffle_data(images, labels)
        
        return images, labels

def main():
    svhn = SVHNsingle("data/cropped")

    # Train dataset
    train_data, train_labels = svhn.extract_process_file("data/cropped/", "train")
    size = int(0.9*train_labels.shape[0])
    
    train_d = train_data[:size,:,:,:]
    train_l = train_labels[:size,]
    valid_d = train_data[size:,:,:,:]
    valid_l = train_labels[size:,]
    
    svhn.save_h5_data(train_d, train_l, "train")
    svhn.save_h5_data(valid_d, valid_l, "valid")

    # Test dataset
    test_data, test_labels = svhn.extract_process_file("data/cropped/", "test")
    svhn.save_h5_data(test_data, test_labels, "test")

    # Extra dataset
    #extra_data, extra_labels = svhn.process_file("data/cropped/", "extra")
    #svhn.save_h5_data(extra_data, extra_labels, "extra")


if __name__ == '__main__':
    main()

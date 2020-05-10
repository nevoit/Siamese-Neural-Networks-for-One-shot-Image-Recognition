import os
import pickle

import tqdm
from PIL import Image
import numpy as np


class DataLoader(object):
    """
    Class for loading data from image files
    """

    def __init__(self, width, height, cells, data_path, output_path):
        """
        Proper width and height for each image.
        """
        self.width = width
        self.height = height
        self.cells = cells
        self.data_path = data_path
        self.output_path = output_path

    def _open_image(self, path):
        """
        Using the Image library we open the image in the given path. The path must lead to a .jpg file.
        We then resize it to 105x105 like in the paper (the dataset contains 250x250 images.)

        Returns the image as a numpy array.
        """
        image = Image.open(path)
        image = image.resize((self.width, self.height))
        data = np.asarray(image)
        data = np.array(data, dtype='float64')
        return data

    def convert_image_to_array(self, person, image_num, data_path, predict=False):
        """
        Given a person, image number and datapath, returns a numpy array which represents the image.
        predict - whether this function is called during training or testing. If called when training, we must reshape
        the images since the given dataset is not in the correct dimensions.
        """
        max_zeros = 4
        image_num = '0' * max_zeros + image_num
        image_num = image_num[-max_zeros:]
        image_path = os.path.join(data_path, 'lfw2', person, f'{person}_{image_num}.jpg')
        image_data = self._open_image(image_path)
        if not predict:
            image_data = image_data.reshape(self.width, self.height, self.cells)
        return image_data

    def load(self, set_name):
        """
        Writes into the given output_path the images from the data_path.
        dataset_type = train or test
        """
        file_path = os.path.join(self.data_path, 'splits', f'{set_name}.txt')
        print(file_path)
        print('Loading dataset...')
        x_first = []
        x_second = []
        y = []
        names = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in tqdm.tqdm(lines):
            line = line.split()
            if len(line) == 4:  # Class 0 - non-identical
                names.append(line)
                first_person_name, first_image_num, second_person_name, second_image_num = line[0], line[1], line[2], \
                                                                                           line[3]
                first_image = self.convert_image_to_array(person=first_person_name,
                                                          image_num=first_image_num,
                                                          data_path=self.data_path)
                second_image = self.convert_image_to_array(person=second_person_name,
                                                           image_num=second_image_num,
                                                           data_path=self.data_path)
                x_first.append(first_image)
                x_second.append(second_image)
                y.append(0)
            elif len(line) == 3:  # Class 1 - identical
                names.append(line)
                person_name, first_image_num, second_image_num = line[0], line[1], line[2]
                first_image = self.convert_image_to_array(person=person_name,
                                                          image_num=first_image_num,
                                                          data_path=self.data_path)
                second_image = self.convert_image_to_array(person=person_name,
                                                           image_num=second_image_num,
                                                           data_path=self.data_path)
                x_first.append(first_image)
                x_second.append(second_image)
                y.append(1)
            elif len(line) == 1:
                print(f'line with a single value: {line}')
        print('Done loading dataset')
        with open(self.output_path, 'wb') as f:
            pickle.dump([[x_first, x_second], y, names], f)


print("Loaded data loader")

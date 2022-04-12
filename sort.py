import pandas as pd
import os
import csv
import shutil


def main():

    train_dict = {}

    with open('train.csv', mode='r') as inp:
        reader = csv.reader(inp)
        train_dict = {rows[0]: rows[1] for rows in reader}

    # Make a new directory each unique value in the training dictionary and put the images with that value in the directory
    for key in train_dict:
        path = 'train_sorted/' + str(train_dict[key])
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy('train/' + key + '.jpg', path)


if (__name__ == '__main__'):
    main()

import datetime
import json
import os


class Experiment():
    def __init__(self, name, h, tau, errors_left, errors_right, additional_numerical_parameters = None):
        self.h = h
        self.temporal_resolution = tau
        self.numerical_parameters = additional_numerical_parameters
        self.errors_left = errors_left
        self.errors_right = errors_right
        self.time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S_%f")
        self.experiment_name = name

    def save(self, save_path):
        if not os.path.exists(save_path):  # create empty directory
            os.mkdir(save_path)

        filename = save_path + "/" + self.experiment_name + "_" + self.time + ".json"
        with open(filename, 'w') as outfile:
            s = json.dumps(self.__dict__, outfile)
            outfile.write(s)

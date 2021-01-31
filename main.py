import torch
from Plotter import Plotter
from TrainingPipeline import TrainingPipeline

if __name__ == '__main__':
    data_dir = "data"

    pipeline = TrainingPipeline()
    image_datasets, loaders = pipeline.initialize_data(data_dir)
    X_train, Y_train, X_train_f = pipeline.load_all_data(loaders)
    X_test, Y_test, X_test_f = pipeline.load_all_data(loaders, kind='val')

    # plot train data with labels
    Plotter.plot_data(image_datasets, X_train, Y_train)

    # train model for Approach1
    pipeline.run_approach(1, X_train_f, X_train, X_test_f, Y_train, image_datasets)

    # train model for Approach2
    #ipeline.run_approach(2, X_train_f, X_train, X_test_f, Y_train, image_datasets)

    # train model for Approach3
    #pipeline.run_approach(3, X_train_f, X_train, X_test_f, Y_train, image_datasets)
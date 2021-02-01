import torch

from GaussianNoiseTransform import GaussianNoiseTransform
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

    # # train model for Approach1
    # model1 = pipeline.run_approach(1, X_train_f, X_train, X_test_f, Y_train, image_datasets)
    # torch.save(model1.state_dict(),"models/model1.pth")
    #
    # #train model for Approach2
    # model2 = pipeline.run_approach(2, X_train_f, X_train, X_test_f, Y_train, image_datasets)
    # torch.save(model2.state_dict(), "models/model2.pth")
    #
    # #train model for Approach3
    # model3 = pipeline.run_approach(3, X_train_f, X_train, X_test_f, Y_train, image_datasets)
    # torch.save(model3.state_dict(), "models/model3.pth")

    # decide which model is better
    # models = pipeline.load_pretrained("models")
    # #pipeline.render_test_data(models, X_test_f)
    #
    # # Model3 seems to be the best
    # approach = 3
    # model = models[approach-1]
    #
    # #calculate statistics
    # Fh, Ffa = pipeline.compute_statistics(model,X_test_f)
    #
    # Plotter.plot_stats(Fh, Ffa)

    # corrupt all images with Gaussian noise
    image_datasets, loaders = pipeline.initialize_data(data_dir, sdev=0.01)
    X_test, Y_test, X_test_f = pipeline.load_all_data(loaders, kind='val')

    # plot train data with labels
    Plotter.plot_data(image_datasets, X_test, Y_test, kind='val')



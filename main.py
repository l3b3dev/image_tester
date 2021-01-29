import torch
from Plotter import Plotter
from TrainingPipeline import TrainingPipeline

if __name__ == '__main__':
    data_dir = "data"

    pipeline = TrainingPipeline()
    image_datasets, loaders = pipeline.initialize_data(data_dir)
    X_train, Y_train, X_train_f = pipeline.load_all_data(loaders)

    # plot train data with labels
    Plotter.plot_data(image_datasets, X_train, Y_train)

    # train model for Approach1
    approach = 1
    pipeline.init_optimizer(approach)
    model, loss_func, opt = pipeline.get_model(approach)
    loss_history = pipeline.train(X_train_f, torch.FloatTensor(
        [float(image_datasets['train'].classes[lookup]) for lookup in Y_train]),
                                  model, opt, loss_func, approach)
    Plotter.plot_losses(loss_history)

    # train model for Approach2
    approach = 2
    pipeline.init_optimizer(approach)
    model, loss_func, opt = pipeline.get_model(approach)
    loss_history = pipeline.train(X_train_f, pipeline.get_lbl_tensor(image_datasets, Y_train),
                                  model, opt, loss_func, approach)
    Plotter.plot_losses(loss_history)

    # testing
    X_test = X_train_f[0]
    result = model(X_test)
    # The index of the output pattern is found by locating the maximum value of y, then finding the indx j of that value
    y_pred = torch.argmax(model(
        X_test))
    p_val = y_pred.item() + 1
    print(p_val)

    # train model for Approach3
    approach = 3
    pipeline.init_optimizer(approach)
    model, loss_func, opt = pipeline.get_model(approach)
    loss_history = pipeline.train(X_train_f, X_train_f,
                                  model, opt, loss_func, approach, 100000)
    Plotter.plot_losses(loss_history)

    X_test = X_train_f[0]
    Y_test_pred = model(X_test)
    Y_pred = Y_test_pred.reshape(16, 16)
    Plotter.plot_sample(X_train[0][0], Y_pred)
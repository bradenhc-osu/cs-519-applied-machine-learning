from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from preprocess import read_data, preprocess, scale_y, separate_numerical
from train import rmse, rmsle
from operator import itemgetter
import numpy as np

if __name__ == "__main__":

    data_train = read_data('./data/my_train.csv')

    X_num_features, X_num, X_cat_features, X_cat = separate_numerical(data_train[:, 1:-1])
    feature_names = X_num_features + X_cat_features

    X_train_cat, m_cat = preprocess(X_cat, binarize=True, names=X_cat_features)
    X_train_num, m_num = preprocess(X_num, names=X_num_features)

    m = {**m_cat, **m_num}

    print(X_train_cat.shape, X_train_num.shape)

    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)
    y_train = scale_y(data_train[:, -1].transpose().astype(int))

    print("Number of features: %d" % len(X_train[0]))

    data_dev = read_data('./data/my_dev.csv')

    X_num_features, X_num, X_cat_features, X_cat = separate_numerical(data_dev[:, 1:-1])

    X_dev_cat = preprocess(X_cat, binarize=True, names=X_cat_features, mappings=m_cat)
    X_dev_num = preprocess(X_num, names=X_num_features, mappings=m_num)

    X_dev = np.concatenate((X_dev_cat, X_dev_num), axis=1)
    y_dev = scale_y(data_dev[:, -1].transpose().astype(int))

    # Train the model
    a = 22.9
    model = Ridge(alpha=a, fit_intercept=True).fit(X_train, y_train)

    # Get some statistics on our model
    print("Top ten most positive/negative weights")
    for i, w in enumerate(sorted(zip([k[2] for k in m.keys()], model.coef_), key=itemgetter(1))):
        if i < 10 or len(model.coef_) - i <= 10:
            print(w)
    print("Bias coefficient = %f" % model.intercept_)

    # Test on our validation set to get the RMSLE
    y_predict = model.predict(X_dev)
    error = rmsle(scale_y(y_dev, reverse=True), scale_y(y_predict, reverse=True))
    print("Dev error = %f " % error)

    # Test our model
    data_test = read_data('./data/test.csv')

    X_num_features, X_num, X_cat_features, X_cat = separate_numerical(data_test[:, 1:])

    X_test_cat = preprocess(X_cat, binarize=True, names=X_cat_features, mappings=m_cat)
    X_test_num = preprocess(X_num, names=X_num_features, mappings=m_num)

    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)

    y_predict = model.predict(X_test)
    y_predict = scale_y(y_predict, reverse=True)

    # Write test results back to the file
    with open("./data/my_submission_4.csv", "w") as file_prediction:
        file_prediction.write("Id,SalePrice\n")
        for i, row in enumerate(data_test):
            file_prediction.write("%s,%s\n" % (row[0], y_predict[i]))

from sklearn.linear_model import LinearRegression
from preprocess import read_data, preprocess, scale_y
from train import rmse, rmsle
from features import feature_names
from operator import itemgetter

if __name__ == "__main__":

    data_train = read_data('./data/my_train.csv')
    X_train, m = preprocess(data_train[:, 1:-1], binarize=True)
    y_train = scale_y(data_train[:, -1].transpose().astype(int))

    data_dev = read_data('./data/my_dev.csv')
    X_dev = preprocess(data_dev[:, 1:-1], mappings=m, binarize=True)
    y_dev = scale_y(data_dev[:, -1].transpose().astype(int))

    # Train the model
    model = LinearRegression(fit_intercept=True).fit(X_train, y_train)

    # Get some statistics on our model
    print("Top ten most positive/negative weights")
    for i, w in enumerate(sorted(zip([ k[2] for k in m.keys() ] , model.coef_), key=itemgetter(1))):
        if i < 10 or len(model.coef_) - i <= 10:
            print(w)
    print("Bias coefficient = %f" % model.intercept_)

    # Test on our validation set to get the RMSLE
    y_predict = model.predict(X_dev)
    error = rmsle(scale_y(y_dev, reverse=True), scale_y(y_predict, reverse=True))
    print("Dev error = %f " % error)

    # Test our model
    data_test = read_data('./data/test.csv')
    X_test = preprocess(data_test[:, 1:], mappings=m, binarize=True)
    y_predict = model.predict(X_test)
    y_predict = scale_y(y_predict, reverse=True)

    # Write test results back to the file
    with open("./data/my_submission.csv", "w") as file_prediction:
        file_prediction.write("Id,SalePrice\n")
        for i, row in enumerate(data_test):
            file_prediction.write("%s,%s\n" % (row[0], y_predict[i]))

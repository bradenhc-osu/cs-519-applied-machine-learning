from sklearn.neighbors import KNeighborsClassifier

def verify(X, Y, k):

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X[:, :-1], X[:,-1])

    predicted = model.predict(Y[:, :-1])

    errors = 0
    
    for i, y in enumerate(Y):
        if y[-1] != predicted[i]:
            errors += 1
    
    return errors / len(Y)
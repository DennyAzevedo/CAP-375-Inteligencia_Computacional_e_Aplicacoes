import activation_functions as func


def derived_sigmoid(x):
    return func.sigmoid(x) * (1 - func.sigmoid(x))


def derived_hyper_tang(x):
    return (1.0 - (func.hyper_tang(x) * func.hyper_tang(x)))

def derived_linear(x):
    return 1.0

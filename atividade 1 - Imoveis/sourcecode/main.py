import numpy as np
import random
import MLP as mlp
import load_dataset as lod


def main():
    # epochsArray = [x for x in range(50, 1000, 50)]
    #epochsArray = np.random.randint(1000, size=1000)

    epochsArray = [200 for i in range(0, 100)]

    dataset = lod.LoadDataset().loading()
    tent = 1
    mse = None
    best_case = None
    best_weights0 = None
    best_weights1 = None
    train = None
    valid = None
    test = None

    for epoch in epochsArray:
        np.random.shuffle(dataset)

        train = dataset[0:71]
        valid = dataset[71:95]
        test = dataset[95:119]

        weights0 = 2 * np.random.random((7, 28)) - 1
        weights1 = 2 * np.random.random((28, 1)) - 1
        lr = 0.2
        network = mlp.MLP(epochs=epoch, train_data=train, valid_data=valid, test_data=test, learning_rate=lr,
                          momentum=1, rmse_min=0.04,
                          stop_params="epochs", tent=tent, bias=1, weights0=weights0, weights1=weights1)
        network.train()
        nash_test, mse_r, w0, w1 = network.validation()

        if tent == 1:
            mse = mse_r

        print("Tentativa: " + str(tent) + " - NASH: " + str(nash_test) + " - MSE: " + str(mse_r) + " - LR: " + str(lr))
        if mse_r < mse:
            nash = nash_test
            mse = mse_r
            best_case = [tent, nash, mse]
            best_weights0 = w0
            best_weights1 = w1

        tent += 1
        # end_for
    print("-" * 80)
    print("Melhor caso")
    print("Tentativa Nº: " + str(best_case[0]) + " - NASH: " + str(best_case[1]) + " - MSE: " + str(best_case[2]))
    network = mlp.MLP(test_data=test, bias=1)
    network.test(weights0=best_weights0, weights1=best_weights1)


if __name__ == '__main__':
    main()

import ImportData as imp
import MLP as mlp
import load_dataset as lod


def main():
    imp.ImportData().importData()
    dataset = lod.LoadDataset().loading()

    train = dataset[0:30609]
    valid = dataset[30610:40812]
    test = dataset[40812:51016]

    tent = 1
    nash = None
    epochsArray = [200 for i in range(0, 10)]
    for epoch in epochsArray:

        network = mlp.MLP(epochs=epoch, train_data=train, valid_data=valid, test_data=test, learning_rate=0.01,
                          momentum=0.5, rmse_min=0.04,
                          stop_params="epochs", tent=tent, bias=1)
        network.train()
        nash_test = network.test()

        if tent == 1:
            nash = nash_test

        print("Melhor NASH: " + str(nash) + " - NASH Tentativa " + str(tent) + " : " + str(nash_test))
        if nash < nash_test:
            nash = nash_test
        tent += 1
        # end_for

if __name__ == '__main__':
    main()



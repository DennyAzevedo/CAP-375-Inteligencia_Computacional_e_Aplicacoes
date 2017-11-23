import ImportData as imp
import MLP as mlp
import load_dataset as lod


def main():
    imp.ImportData().importData()
    dataset = lod.LoadDataset().loading()

    train = dataset[0:30609]
    valid = dataset[30610:40812]
    test = dataset[40812:51016]

    #epochsArray = [x for x in range(50, 1000, 50)]
    epochsArray = [1000]
    for epoch in epochsArray:
        network = mlp.MLP(epochs=epoch, train_data=train, valid_data=valid, test_data=test, learning_rate=0.5, momentum=1, rmse_min=0.04,
                          stop_params="epochs")
        network.train()
        network.test()
        # end_for


if __name__ == '__main__':
    main()



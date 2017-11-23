import numpy as np

class ImportData:
    def __init__(self):
        pass

    def importData(self):
        input = np.loadtxt("dataset//pattern.in")
        inputT = np.transpose(input)
        inputT = np.where(inputT > 5, 5, inputT)
        print(len(inputT))

        np.savetxt("dataset//input.txt", inputT, fmt='%f')

        output = np.loadtxt("dataset//pattern.out")

        output = np.where(output > 5, 5, output)
        print(len(inputT))

        np.savetxt("dataset//output.txt", output, fmt='%f')
#
# test_and_or.py: creates various tic-tac-toe configurations 
#   for testing purposes
#
# Author: Derek Riley, 2020
#

from nn import NeuralNetwork
from typing import *
import numpy as np
import random


def create_or_nn_data():
    # input training data set for OR
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                  [1],
                  [1],
                  [1]])
    return x, y


def create_and_nn_data():
    # input training data set for OR
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                  [0],
                  [0],
                  [1]])
    return x, y


def load_tictactoe_csv(filepath):
    boards = []
    labels = []
    with open(filepath) as fl:
        for ln in fl:
            cols = ln.strip().split(",")
            board = []
            for s in cols[:-1]:
                if s == "o":
                    board += [0]
                elif s == "x":
                    board += [1]
                else:
                    board += [2]
            # board = [0 if s == "o" else 1 for s in cols[:-1]]
            label = [0] if cols[-1] == "Owin" else [1]
            labels.append(label)
            boards.append(board)
    x = np.array(boards)
    y = np.array(labels)
    return x, y


def test_and_nn_1():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 4, 1)
    nn.train(150)
    # print(nn.loss())
    assert nn.loss() < .04


def test_and_nn_2():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 4, 2)
    nn.train(150)
    # print(nn.loss())
    assert nn.loss() < .01


def test_and_nn_3():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 3, 1)
    nn.train(1500)
    # print(nn.loss())
    assert nn.loss() < .002


def test_or_nn_1():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 4, 1)
    nn.train(150)
    # print(nn.loss())
    assert nn.loss() < .3


def test_or_nn_2():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 10, 1)
    nn.train(1000)
    # print(nn.loss())
    assert nn.loss() < .002


def test_or_nn_3():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 1, 2)
    nn.train(1500)
    # print(nn.loss())
    assert nn.loss() < .0009


def test_nn_1():
    x, y = load_tictactoe_csv("tic-tac-toe.csv")
    nn = NeuralNetwork(x, y, 4, .1)
    nn.train(1000)
    # print(nn.loss())
    assert nn.loss() < .06


def test_nn_2():
    x, y = load_tictactoe_csv("tic-tac-toe.csv")
    nn = NeuralNetwork(x, y, 10, .1)
    nn.train(10000)
    # print(nn.loss())
    assert nn.loss() < .0025


def test_nn_3():
    x, y = load_tictactoe_csv("tic-tac-toeWBlanksSmall.csv")
    nn = NeuralNetwork(x, y, 10, .004)
    nn.train(10000)

    print("3 " + str(nn.loss()))
    print(nn.accuracy_calculator())
    assert nn.loss() < .1


def test_nn_4():
    x, y = load_tictactoe_csv("tic-tac-toeWBlanks.csv")
    nn = NeuralNetwork(x, y, 11, .00066)
    # 10, 0.0003, Epoch: 10000  -> 76%
    # 10, 0.0005, Epoch: 10000  -> 89%
    # 10, 0.0007, Epoch: 10000 -> 89.5%
    # 11, 0.00066, Epoch: 10000 -> 92.3%
    # 11, 0.00066, Epoch: 100,000 -> 96.9%
    # 11, 0.00066, Epoch: 200,000 -> 95.
    nn.train(100000)
    print("4 " + str(nn.loss()))
    print(nn.accuracy_calculator())
    assert nn.loss() < .001


def test_nn_5():
    x, y = load_tictactoe_csv("tic-tac-toeWBlanks.csv")
    nn = NeuralNetwork(x, y, 20, .01)
    nn.train(100000)
    print(nn.loss())
    print(nn.accuracy_calculator())
    assert nn.loss() < .01


def test_nn_blanks():
    random_split_data()
    x, y = load_tictactoe_csv("tic-tac-toeWBlanksTraining.csv")
    nn = NeuralNetwork(x, y, 11, .00066)
    nn.train(100000)
    boards = []
    labels = []
    with open("tic-tac-toeWBlanksValidation.csv") as file:
        for line in file:
            cols = line.strip().split(",")
            board = []
            for s in cols[:-1]:
                if s == "o":
                    board += [0]
                elif s == "x":
                    board += [1]
                else:
                    board += [2]
            label = [0] if cols[-1] == "Owin" else [1]
            labels.append(label)
            boards.append(board)
    lines = np.array(boards)
    outputs = np.array(labels)
    count = 0
    right = 0
    wrong = 0
    for line in lines:
        actual_output = outputs[count]
        calc_output = int(nn.inference(line) + .5)  # rounds to 0 or 1
        if actual_output == calc_output:
            right += 1
        else:
            wrong += 1
        count += 1
    print("Accuracy: " + str(right / (right + wrong)))


def run_all() -> None:
    """Runs all test cases"""
    test_or_nn_1()
    test_or_nn_2()
    test_or_nn_3()
    test_and_nn_1()
    test_and_nn_2()
    test_and_nn_3()
    test_nn_1()
    test_nn_2()
    test_nn_3()
    test_nn_4()
    test_nn_5()
    print("All tests pass.")


def test_show_results() -> None:
    """Prints outputs so that results can be examined"""
    x, y = load_tictactoe_csv("tic-tac-toe.csv")
    nn = NeuralNetwork(x, y)
    nn.train(1000)
    boards = []
    labels = []
    with open("tic-tac-toeFull.csv") as file:
        for line in file:
            cols = line.strip().split(",")
            board = [0 if s == "o" else 1 for s in cols[:-1]]
            label = 0 if cols[-1] == "Owin" else 1
            labels.append([label])
            boards.append(board)
    lines = np.array(boards)
    outputs = np.array(labels)
    count = 0
    for line in lines:
        print(line, end="     ")
        print("Actual Output: " + str(outputs[count]), end="     ")
        print("Calculated Output: " + str(int(nn.inference(line) + .5)))  # rounds to 0 or 1
        print()
        count += 1


def random_split_data() -> None:
    """Splits the tic-tac-toeWBlanks file into training and validation files. ~75%/25% split"""
    training_file = open("tic-tac-toeWBlanksTraining.csv", "w")
    validation_file = open("tic-tac-toeWBlanksValidation.csv", "w")
    with open("tic-tac-toeWBlanks.csv") as file:
        for line in file:
            rand = random.random()
            if rand > .75:
                validation_file.write(line)
            else:
                training_file.write(line)
    training_file.close()
    validation_file.close()


def main() -> int:
    """Main test program which prompts user for tests to run and displays any
    result.
    """

    # test_show_results()

    n = int(input("Enter test number (1-12; 0 = run all): "))
    if n == 0:
        run_all()
        return 0
    elif n == 1:
        result = test_or_nn_1()
    elif n == 2:
        result = test_or_nn_2()
    elif n == 3:
        result = test_or_nn_3()
    elif n == 4:
        result = test_and_nn_1()
    elif n == 5:
        result = test_and_nn_2()
    elif n == 6:
        result = test_and_nn_3()
    elif n == 7:
        result = test_nn_1()
    elif n == 8:
        result = test_nn_2()
    elif n == 9:
        result = test_nn_3()
    elif n == 10:
        result = test_nn_4()
    elif n == 11:
        result = test_nn_5()
    elif n == 12:
        result = test_nn_blanks()
    else:
        print("Error: unrecognized test number " + str(n))
    print("Test passes with result " + str(result))
    return 0


if __name__ == "__main__":
    exit(main())
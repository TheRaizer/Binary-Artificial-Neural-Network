import pickle

import scripts.train_tests.iris_train_test as iris
import scripts.train_tests.hrt_dis_train_test as heart


def get_layer_length(layer_index):
    try:
        length = int(input("(must be > 0), How many neurons should be in layer " + str(layer_index + 1) + ": ").strip())
        if length <= 0:
            raise ValueError()
        return length
    except ValueError:
        print("Not a valid input.")
        return get_layer_length(layer_index)


def get_positive_input(message, float_parse=False, allow_zero=False):
    sign = ">=" if allow_zero else ">"

    try:
        if float_parse:
            num = float(input("(must be " + sign + " 0), " + message).strip())
        else:
            num = int(input("(must be " + sign + " 0), " + message).strip())

        if (not allow_zero) and (num <= 0):
            raise ValueError()
        elif num < 0:
            raise ValueError()
        return num
    except ValueError:
        print("Not a valid input.")
        return get_positive_input(message, float_parse, allow_zero)


def run_test(data_type):
    print("\nTime to TEST!")
    if data_type == "heart":
        heart.test_heart_binary()
    elif data_type == "iris":
        iris.test_iris_binary()


def get_action_choice():
    print("\nTest saved parameters or train new ones?")
    print("- 'test'")
    print("- 'train'")

    # get input in lowercase as well as remove preceding and trailing spaces
    action = input("\nChoice: ").strip().lower()

    # recurse if action wasn't valid
    if (action != "train") and (action != "test"):
        print("invalid choice.")
        return get_action_choice()
    else:
        return action


def get_data_type(action):
    print("\nWhat type of data would you like to", action, "on?")
    print("(easy) - 'iris'")
    print("(hard) - 'heart'")

    # get input in lowercase as well as remove preceding and trailing spaces
    data_type = input("\nChoice: ").strip().lower()

    data_type = data_type.strip().lower()

    # recurse if data type was invalid.
    if (data_type != "iris") and (data_type != "heart"):
        print("invalid choice.")
        return get_data_type(action)
    else:
        return data_type


def input_loop():
    IRIS_INPUT_LENGTH = 4
    HRT_INPUT_LENGTH = 13

    while True:
        action = get_action_choice()
        data_type = get_data_type(action)

        if action == "test":
            run_test(data_type)
            continue

        dims = []
        print("")
        num_of_layers = get_positive_input("# of layers (excluding input and output layers)? ")

        # for every layer get the length
        for i in range(num_of_layers):
            length = get_layer_length(i)
            dims.append(length)

        # get other hyper parameters
        alpha = get_positive_input("Value of learning rate (recommended < 1)? ", True)
        iterations = get_positive_input("# of iterations (recommended >= 1000)? ")
        lambd = get_positive_input("Value of regularization parameter (recommended < 0.1)? ", True, True)

        print("\nWould you like to use mini batches? ")
        print("- 'yes'")
        print("- anything else")

        is_mini_batch = input("Choice: ").strip().lower() == "yes"
        batch_count = 0

        if is_mini_batch:
            batch_count = get_positive_input("What is the size of each batch? ")

        dims.append(1)
        if data_type == "heart":
            dims.insert(0, HRT_INPUT_LENGTH)
            heart.train_heart_binary(dims, alpha, iterations, is_mini_batch, batch_count, lambd)
        elif data_type == "iris":
            dims.insert(0, IRIS_INPUT_LENGTH)
            iris.train_iris_binary(dims, alpha, iterations, is_mini_batch, batch_count, lambd)


if __name__ == '__main__':
    # "\033[1m" is an ANSI escape sequence used to bold this line
    print("\033[1m" + "Before saving new parameters try 'test' to check the best parameters I trained for each data type." + "\033[0m")
    input_loop()


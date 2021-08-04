import scripts.train_tests.iris_train_test as iris
import scripts.train_tests.hrt_dis_train_test as heart


IRIS_INPUT_LENGTH = 4
HRT_INPUT_LENGTH = 13

IRIS_DATA_TYPE = "iris"
HRT_DATA_TYPE = "heart"

TRAIN = "train"
TEST = "test"

def get_layer_length(layer_index):
    """ Asks for a given layers length until the input is valid.

    Preconditions: 
    layer_index: int >= 0

    Parameters:
    layer_index: The index of the given layer whose length we will be obtaining.

    Postconditions:
    returns length: The valid inputted layer length.
    """

    # try to parse input to integer
    try:
        length = int(input("(must be > 0), How many neurons should be in layer " + str(layer_index + 1) + ": ").strip())
        if length <= 0:
            # if its a negative number raise an error
            raise ValueError()
        return length
    # catch any ValueErrors
    except ValueError:
        print("Not a valid input.")
        # recurse the method if there was an error
        return get_layer_length(layer_index)


def get_positive_input(message, float_parse=False, allow_zero=False):
    """ Obtains and returns a positive int from the user.

    Preconditions:
    message: non-empty string
    float_parse: bool defaulted to False
    allow_zero: bool defaulted to False

    Parameters:
    message: The message that is printed when obtaining the input.
    float_parse: Whether to parse input to float or int
    allow_zero: Whether to allow zero as an input

    Postconditions:
    num: The valid inputted number.
    """

    # use ternary operator to determine the sign to use
    sign = ">=" if allow_zero else ">"

    # try to parse input to either a float or int
    try:
        if float_parse:
            num = float(input("(must be " + sign + " 0), " + message).strip())
        else:
            num = int(input("(must be " + sign + " 0), " + message).strip())

        # raise a ValueError if input was invalid
        if (not allow_zero) and (num <= 0):
            raise ValueError()
        elif num < 0:
            raise ValueError()
        return num

    # catch any ValueErrors.
    except ValueError:
        print("Not a valid input.")
        # recurse the method until proper input was found
        return get_positive_input(message, float_parse, allow_zero)


def run_test(data_type):
    """ Runs model against test set of a given data type.

    Preconditions:
    data_type: Either "iris" or "heart"

    Postconditions:
    data_type: The type of data we will be testing the model on.

    Postconditions:
    Tests the model which outputs the accuracy it performed on the test set.
    """

    print("\nTime to TEST!")
    if data_type == HRT_DATA_TYPE:
        heart.test_heart_binary()
    elif data_type == IRIS_DATA_TYPE:
        iris.test_iris_binary()


def get_action_choice():
    """ Gets from the user the action they wish to perform and returns it.

    Postconditions:
    action: Either "train" or "test"
    """

    print("\nTest saved parameters or train new ones?")
    print("- 'test'")
    print("- 'train'")

    # get input in lowercase as well as remove preceding and trailing spaces
    action = input("\nChoice: ").strip().lower()

    # recurse if action wasn't valid
    if (action != TRAIN) and (action != TEST):
        print("invalid choice.")
        return get_action_choice()
    else:
        return action


def get_data_type(action):
    """ Gets the data type the user wishes to use and returns it.

    Preconditions:
    action: string that is either "train" or "test"

    Parameters:
    action: The action the user wishes to perform with the model

    Postconditions:
    data_type: string either "heart" or "iris"
    """

    print("\nWhat type of data would you like to", action, "on?")
    print("(easy) - 'iris' = detecting iris flower types")
    print("(hard) - 'heart' = detecting heart disease")

    # get input in lowercase as well as remove preceding and trailing spaces
    data_type = input("\nChoice: ").strip().lower()

    data_type = data_type.strip().lower()

    # recurse if data type was invalid.
    if (data_type != IRIS_DATA_TYPE) and (data_type != HRT_DATA_TYPE):
        print("invalid choice.")
        return get_data_type(action)
    else:
        if data_type == IRIS_DATA_TYPE:
            print("Iris data has input length of:", IRIS_INPUT_LENGTH)
        else:
            print("Heart data has input length of:", HRT_INPUT_LENGTH)
    return data_type


def input_loop():
    """ The constant loop that allows the user to configure the neural network

    Postconditions:
    Either tests or trains the neural network given the users hyper-parameter choices.
    """

    while True:
        action = get_action_choice()
        data_type = get_data_type(action)

        if action == TEST:
            run_test(data_type)
            continue

        # This will contain the length of each layer in the network (list of int)
        dims = []

        # create a empty line
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
        alpha_decay = get_positive_input("Value of learning rate decay (recommended < 1)? ", True, True)

        print("\nWould you like to use mini batches? ")
        print("- 'yes'")
        print("- anything else")

        is_mini_batch = input("Choice: ").strip().lower() == "yes"
        batch_count = 0

        if is_mini_batch:
            batch_count = get_positive_input("What is the size of each batch? ")

        # Append 1 because the last layer of the network must always be 1 neuron in length
        # this is due to the nature of binary classification.
        dims.append(1)

        if data_type == HRT_DATA_TYPE:
            # insert the proper input data length at the 0th index because the 0th layer must match the input length
            dims.insert(0, HRT_INPUT_LENGTH)
            heart.train_heart_binary(dims, alpha, iterations, is_mini_batch, batch_count, lambd, alpha_decay)
        elif data_type == IRIS_DATA_TYPE:
            # insert the proper input data length at the 0th index because the 0th layer must match the input length
            dims.insert(0, IRIS_INPUT_LENGTH)
            iris.train_iris_binary(dims, alpha, iterations, is_mini_batch, batch_count, lambd, alpha_decay)
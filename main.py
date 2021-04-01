import scripts.inputmanager as inp


if __name__ == '__main__':
    # "\033[1m" is an ANSI escape sequence used to bold this line
    print("\033[1m" + "Before saving new parameters try 'test' to check the best parameters I trained for each data type." + "\033[0m")
    inp.input_loop()


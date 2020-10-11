from docplex.cp.model import CpoModel

# Create CPO model
mdl = CpoModel()


def main():
    num_nodes = 10
    w = [mdl.binary_var(name=f"w_{j}") for j in range(num_nodes)]
    x = [mdl.binary_var(name=f"x_{i}_{j}") for i in range(num_nodes) for j in range(num_nodes)]


if __name__ == '__main__':
    main()
    print("done!")
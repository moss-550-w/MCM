import sys

def run(filename):
    with open(filename, "r") as f:
        code = f.read()
        exec(code)

def main():
    run("1.py")


if __name__ == "__main__":
    main()

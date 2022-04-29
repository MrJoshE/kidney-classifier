import os


def main():
    os.environ['KAGGLE_USERNAME'] = 'josheverett'  # Your Kaggle username
    # Your Kaggle API key
    os.environ['KAGGLE_KEY'] = 'e2116ffceb00d0e84caaa94ee3996ccb'
    # Your URN: submissions without a URN will not count
    os.environ['URN'] = '6621182'


if __name__ == '__main__':
    main()

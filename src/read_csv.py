import click
import pandas as pd

@click.command()
@click.argument('path_read', type = str)
@click.argument('path_save', type = str)
@click.option('--delim', type = str)
def main(path_read, delim=",", path_save):
    # Read the csv file from path_read, which can be a URL or filepath
    pd.read_csv(path_read, sep=delim).to_csv(path_save, index = False)
        
if __name__ == '__main__':
    main()
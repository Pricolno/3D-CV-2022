import click
import time
import sys

print("start")
with click.progressbar(range(5),
                       label='Calculating time',
                       ) as bar:
    for x in bar:
        #print(x)
        time.sleep(1)

print("finish")
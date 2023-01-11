import click
import time
import sys

from collections import OrderedDict

# print("start")
# with click.progressbar(range(5),
#                        label='Calculating time',
#                        ) as bar:
#     for x in bar:
#         #print(x)
#         time.sleep(1)
#
# print("finish")


od = OrderedDict({1:20, 2:30})
for v in od:
    print(v)
print("DEB")
for v in reversed(od):
    print(v)

print("DEBAG")
for i in range(10, -1, -1):
    print(i)

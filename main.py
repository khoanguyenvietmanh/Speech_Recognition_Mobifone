import argparse
import logging
import pickle
from enum import Enum
from collections import namedtuple
from collections import deque
from collections import defaultdict
from collections import ChainMap
from collections import Counter
from collections import OrderedDict
from collections import defaultdict


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

file_handler = logging.FileHandler("main.log")

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

parser = argparse.ArgumentParser(description="My first parse")

group = parser.add_mutually_exclusive_group()

group.add_argument('-q', '--quite', action="store_true", help="print quite")

parser.add_argument('-x', '--x', type=int, metavar='', default=64, help="X coordinate")

parser.add_argument('-y', '--y', type=int, metavar='', default=64, help="Y coordinate")

# parser.add_argument('-m', '--m', type=int, metavar='', required=True, help="M coordinate")
#
# parser.add_argument('-n', '--n', type=int, metavar='', required=True, help="N coordinate")

class Days(Enum):
    Temperature = 1
    Humidity = 2

args = parser.parse_args()

if __name__ == '__main__':
    # a = namedtuple("courses", 'name, technology')
    # s = a._make(['data science', 'python'])
    # a = {1 : "ereuka", 2 : "python"}
    # b = {3 : "ML", 4 : "AI"}
    # chain_map = ChainMap(a, b)

    # a = [1, 1, 2, 3, 2, 4, 3, 4, 1, 2]
    #
    # b = Counter(a)
    #
    # sub = {2:1, 6:1}
    # print(b)
    #
    # print(list(b.elements()))
    #
    # b.subtract(sub)
    #
    # print(b.most_common())
    #
    # d = OrderedDict()
    #
    # d[1] = 'd'
    # d[2] = 'e'
    # d[3] = 'f'

    a = defaultdict(int)

    a[1] = "python"
    a[2] = "C#"

    print(a[3])
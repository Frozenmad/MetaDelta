import os
from time import gmtime, strftime
import requests

DEBUG=0
INFO=1
WARN=2
ERROR=3
LEVEL = DEBUG

_idx2str = ['D', 'I', 'W', 'E']

get_logger = lambda x, filename='log.txt': Logger(x, filename)

class Logger():
    def __init__(self, name='', filename='log.txt') -> None:
        self.name = name
        if self.name != '':
            self.name = '[' + self.name + ']'

        self.debug = self._generate_print_func(DEBUG, filename=filename)
        self.info = self._generate_print_func(INFO, filename=filename)
        self.warn = self._generate_print_func(WARN, filename=filename)
        self.error = self._generate_print_func(ERROR, filename=filename)

    def _generate_print_func(self, level=DEBUG, filename='log.txt'):
        def prin(*args, end='\n'):
            if level >= LEVEL:
                strs = ' '.join([str(a) for a in args])
                str_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                print('[' + _idx2str[level] + '][' + str_time + ']' + self.name, strs, end=end)
                open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../' + filename)), 'a').write(
                    '[' + _idx2str[level] + '][' + str_time + ']' + self.name + strs + end
                )
        return prin

def safe_log(url, params):
    try:
        requests.get(url=url, params=params, timeout=1)
    except:
        pass

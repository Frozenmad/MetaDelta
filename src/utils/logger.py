import os
from time import gmtime, strftime

DEBUG=0
INFO=1
WARN=2
ERROR=3

LEVEL = DEBUG

_idx2str = ['D', 'I', 'W', 'E']

get_logger = lambda x:Logger(x)

class Logger():
    def __init__(self, name='') -> None:
        self.name = name
        if self.name != '':
            self.name = '[' + self.name + ']'

        self.debug = self._generate_print_func(DEBUG)
        self.info = self._generate_print_func(INFO)
        self.warn = self._generate_print_func(WARN)
        self.error = self._generate_print_func(ERROR)

    def _generate_print_func(self, level=DEBUG):
        def prin(*args, end='\n'):
            if level >= LEVEL:
                strs = ' '.join([str(a) for a in args])
                str_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                print('[' + _idx2str[level] + '][' + str_time + ']' + self.name, strs, end=end)
                open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../log.txt')), 'a').write(
                    '[' + _idx2str[level] + '][' + str_time + ']' + self.name + strs + end
                )
        return prin


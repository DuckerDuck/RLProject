from argparse import ArgumentParser, Action as ParseAction, Namespace
from os.path import isfile

import json

def is_type(string, t):
    try:
        t(string)
        return True
    except:
        return False

def fit_type(string, types):
    for t in types:
        if is_type(string, t):
            return t(string)

    return string

class DictArgs(ParseAction):
     def __call__(self, parser, namespace, values, option_string=None):
         d = dict()
         for kv in values:
             k,v = kv.split("=")
             d[k] = fit_type(v, [json.loads, int, float, str])

         setattr(namespace, self.dest, d)

class SettingsParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument(
            '--save_to',
            type = lambda x : x if x.endswith('.json') else x+'.json',
            default = None,
            help = 'file in which to save settings'
        )

        self.add_argument(
            '--load_from',
            type = lambda x : x if x.endswith('.json') else x+'.json',
            default = None,
            help = 'file from which to load settings'
        )

    def parse_args(self, *args, **kwargs):

        args = super().parse_args(*args, **kwargs)

        if args.load_from is not None and isfile(args.load_from):
            with open(args.load_from, 'rt') as f:
                args = Namespace()
                for k, v in json.load(f).items():
                    setattr(args, k, fit_type(v, [dict, int, float]))

        if args.save_to is not None:
            with open(args.save_to, 'wt') as f:
                json.dump(vars(args), f)

        return args


def get_mod_attr(mod, names):
    for name in names.split('.'):
        mod = getattr(mod, name)
    return mod

import json
from collections import defaultdict

from utils.settings import SettingsParser


class ResultsManager:
    """
    This class maintains a global reference to a ResultsWriter instance.
    To write a result from anywhere in the code simply import ResultsManager
    and call ResultsManager.get_writer(). Then all values added to the writer 
    should be saved into the current results file.
    """
    writer = None

    def setup_writer(output_file, settings):
        writer = ResultsWriter(output_file, settings)
        return writer

    def get_writer():
        if writer == None:
            raise Exception('Writer not initialized!')
        return writer


class ResultsWriter:
    def __init__(self, output_file: str, settings: SettingsParser):
        super().__init__()
        self.output_file = output_file
        self.results = defaultdict(list)
        self.set_settings(settings)


    def set_settings(self, settings):
        copy = vars(settings).copy()
        
        # Remove keys that can't be serialised
        del copy['device']
        self.results['__settings'] = copy


    def save(self):
        with open(self.output_file, 'wt') as f:
            json.dump(self.results, f)


    def add_value(self, name, value):
        self.results[name].append(value)

import os.path as path
import configparser


class Config:

    class __Config:
        config_path = './config.ini'
        paths = {}
        api = {}

        def __init__(self):
            config = configparser.ConfigParser()
            config.read(self.config_path)

            if 'Paths' in config:
                for key in config['Paths']:
                    self.paths[key] = path.join(path.curdir, config['Paths'][key])
            if 'Api' in config:
                for key in config['Api']:
                    self.api[key] = config['Api'][key]

    instance = None

    def __init__(self):
        if not Config.instance:
            Config.instance = Config.__Config()

    def __getattr__(self, item):
        return getattr(self.instance, item)

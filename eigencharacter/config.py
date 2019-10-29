import os
import json

class Config:
    __inst = None

    def __init__(self):
        if Config.__inst != None:
            raise Exception("Use getInstance")
        else:
            self.setup()
            Config.__inst = self        

    def setup(self):
        self.DEFAULT_FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
        self.FONT_SIZE = 64
        if os.path.exists("config.json"):
            self.load_config()
        else:
            self.save_config_template()

    @staticmethod
    def getInstance():
        if Config.__inst == None:
            Config()
        return Config.__inst

    def load_config(self):
        with open("config.json", "r", encoding="UTF-8") as fin:
            config_data = json.load(fin)
        for k, v in config_data.items():
            self.__dict__[k] = v

    def save_config_template(self):
        with open("config.json", "w", encoding="UTF-8") as fout:
            json.dump(self.__dict__, fout)

config = Config.getInstance()        

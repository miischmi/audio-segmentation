import json
from datetime import datetime

class JSON_Classifier():
    
    def __init__(self):
        pass

    def readJSON(self, path):
        """
        Method to read a JSON-file
        """
        with open(path) as jasonfile:
            data = json.load(jasonfile)
            for work in data['works']:
                for performance in work['performances']:
                    self.segments = []
                    for segment in performance['segments']:
                        start = strToSeconds(segment['start'])
                        end = strToSeconds(segment['end'])
                        self.segments.append({ 'id': segment['id'], 'start': start, 'ende': end })

def strToSeconds(str):
    hr, min, sec = map(int, str.split(':'))
    sec = (((hr * 60) + min) * 60) + sec
    return sec



                

 
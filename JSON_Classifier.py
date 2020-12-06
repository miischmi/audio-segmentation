import json

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
                        self.segments.append({ 'id': segment['id'], 'start': segment['start'], 'ende': segment['end'] })

               

                   


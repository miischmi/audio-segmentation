import json
from datetime import datetime

class JSON_Classifier():
    
    def readJSON(self, path):
        """Reading the JSON file with the metadata of the recording

        Args:
            id: ID of the Segment
            start: Start of the segment in seconds
            end: end of the segment in seconds
        
        Returns: 
            a map of segment-metadata
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



                

 
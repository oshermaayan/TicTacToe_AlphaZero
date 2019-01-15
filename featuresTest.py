import json
import numpy as np
from FeatureExtractor import FeatureExtractor

json_path = "json_6x6.json"

with open(json_path, "r") as json_file:
    data = json.load(json_file)
    board = np.array(data["position"])

fe = FeatureExtractor(streak_size=4)
tst1 = fe.extractFeatures(board=board,index=(1,5))

print(tst1)
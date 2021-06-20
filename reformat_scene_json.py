import json

final_scene = {}
with open('data/scene_recognizer/scene.json') as f:
    scene = json.load(f)

for key in scene:
    scene_dets = scene[key]
    key_comb = key.split('/')[-1].split('.')[0]
    final_scene[key_comb] = scene_dets

with open("scene_dets.json", "w") as f:
    json.dump(final_scene, f)
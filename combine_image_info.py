import numpy as np
import json
import os


def combine(key, text, scene):
    #with open('ocr_with_bbox.json') as f:
    #  text = json.load(f)
    #with open('data/scene_recognizer/scene.json') as f:
    #    scene = json.load(f)

    #combo = text.copy()
    #print(len(list(scene.keys())))
    #for key in scene:
    #    scene_dets = scene[key]
    #    key_comb = key.split('/')[-1].split('.')[0]
    #    if key_comb in combo:
    #        combo[key_comb].append(scene_dets)
    #   else:
    #       pass
    #       #print(key_comb)
    combo = []
    combo = text.copy()
    combo.append(scene)
    #print("Scene info entered")
    feats_dir = r'data/frcnn'
    #for key in combo:
    feats_file = os.path.join(feats_dir, key+'.npy')
    feats_info = os.path.join(feats_dir, key+'_info.npy')
    if os.path.exists(feats_file) and os.path.exists(feats_info):
        img_feats = np.load(feats_file)
        combo.append(img_feats)
        img_info = np.load(feats_info, allow_pickle=True)
        combo.append(img_info)
        #for key in combo:
        #    print(key, combo[key])
        #with open("combo.npy", 'w') as fp:
        #np.save('combo.npy', combo, allow_pickle=True)
        return combo
    return None
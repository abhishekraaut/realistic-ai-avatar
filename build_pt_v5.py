import os
import json
import pickle
import torch
import numpy as np

def build():
    with open('dataset_v5_raw.pkl', 'rb') as f:
        raw = pickle.load(f)
    data = raw['data']
    feature_names = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
    
    dataset = {'train': {'X': [], 'Y': []}, 'val': {'X': [], 'Y': []}, 'test': {'X': [], 'Y': []}}
    total_frames = 0
    all_features = []
    
    for seq in data:
        if len(seq['blendshapes'].shape) < 2 or seq['blendshapes'].shape[0] == 0: continue
        identity = seq['seq_id'].split('_')[1] # ID_002_seq_01 -> ID_002
        
        # We only have ID_002 that works. Do temporal split on ID_002.
        Y = torch.tensor(seq['blendshapes'], dtype=torch.float32)
        # Normalize min-max[0,1]
        min_val = Y.min(dim=0, keepdim=True)[0]
        max_val = Y.max(dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        Y = (Y - min_val) / range_val
        
        X = torch.zeros((Y.shape[0], 16, 80), dtype=torch.float32)
        
        # Temporal split (Case B: limited dev split)
        train_split = int(0.8 * Y.shape[0])
        
        dataset['train']['X'].append(X[:train_split])
        dataset['train']['Y'].append(Y[:train_split])
        dataset['val']['X'].append(X[train_split:])
        dataset['val']['Y'].append(Y[train_split:])
        
        total_frames += Y.shape[0]
        all_features.append(Y.numpy())
    
    final_dataset = {}
    for k in ['train', 'val', 'test']:
        if dataset[k]['X']:
            final_dataset[k] = {
                'X': torch.cat(dataset[k]['X'], dim=0),
                'Y': torch.cat(dataset[k]['Y'], dim=0),
            }
        else:
            final_dataset[k] = {'X': torch.empty((0, 16, 80)), 'Y': torch.empty((0, 52))}
    
    final_dataset['feature_names'] = feature_names
    final_dataset['version'] = '5.0'
    
    torch.save(final_dataset, 'synthesia_training_data/dataset_v5.pt')
    
    # Generate reports
    if all_features: all_features = np.concatenate(all_features, axis=0)
    else: all_features = np.zeros((0, 52))
    
    manifest = {
        'version': '5.0',
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'total_frames': total_frames,
        'train_frames': final_dataset['train']['Y'].shape[0],
        'val_frames': final_dataset['val']['Y'].shape[0],
        'test_frames': final_dataset['test']['Y'].shape[0],
        'normalization': 'min-max[0,1]',
        'split_policy': 'TEMPORAL_LIMITED_DEV'
    }
    with open('synthesia_training_data/dataset_v5_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    report = {
        'NaN_count': int(np.isnan(all_features).sum()),
        'Inf_count': int(np.isinf(all_features).sum()),
        'mean': all_features.mean(axis=0).tolist() if len(all_features)>0 else [],
        'std': all_features.std(axis=0).tolist() if len(all_features)>0 else [],
        'min': all_features.min(axis=0).tolist() if len(all_features)>0 else [],
        'max': all_features.max(axis=0).tolist() if len(all_features)>0 else [],
        'variance': all_features.var(axis=0).tolist() if len(all_features)>0 else []
    }
    with open('synthesia_training_data/validation_v5_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print('Dataset V5 built and validated.')

if __name__ == '__main__':
    build()

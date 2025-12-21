import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_audface_data(basedir, testskip=1, test_file=None, aud_file=None, test_size=-1):
    if test_file is not None:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        aud_features = np.load(os.path.join(basedir, aud_file))
        cur_id = 0
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            aud_id = cur_id
            auds.append(aud_features[aud_id])
            cur_id = cur_id + 1
            if cur_id == aud_features.shape[0] or cur_id == test_size:
                break
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])
        return poses, auds, bc_img, [H, W, focal, cx, cy]

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_com_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    aud_features = np.load(os.path.join(basedir, 'aud.npy'))
    counts = [0]
    for s in splits:
        meta = metas[s]
        com_imgs = []
        poses = []
        auds = []
        sample_rects = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            filename = os.path.join(basedir, 'com_imgs',
                                    str(frame['img_id']) + '.jpg')
            com_imgs.append(filename)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(
                aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        com_imgs = np.array(com_imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + com_imgs.shape[0])
        all_com_imgs.append(com_imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(sample_rects)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    com_imgs = np.concatenate(all_com_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)

    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    return com_imgs, poses, auds, bc_img, [H, W, focal, cx, cy], \
        sample_rects, i_split


# BIT Note: 
#   This is problematic. When driving talking face with any audio (which doesn't have a json file), 
#   the length of default test_pose_file still has an effect, so need to make sure test_pose_file 
#   is long enough to cover the audio length of aud_file.
def load_test_data(basedir, aud_file, test_pose_file='transforms_train.json',
                   testskip=1, test_size=-1, aud_start=0):
    
    # BIT Test : Debug prints for troubleshooting
    print(f"[load_test_data] basedir={basedir}")
    print(f"[load_test_data] aud_file={aud_file}")
    print(f"[load_test_data] test_pose_file={test_pose_file}")
    print(f"[load_test_data] testskip={testskip}, test_size={test_size}, aud_start={aud_start}")

    with open(os.path.join(basedir, test_pose_file)) as fp:
        meta = json.load(fp)

    # BIT Test
    frames_total = len(meta.get('frames', []))
    frames_after_skip = len(meta.get('frames', [])[::testskip]) if testskip > 0 else frames_total
    print(f"[load_test_data] pose frames total={frames_total}, after testskip={frames_after_skip}")

    poses = []
    auds = []
    aud_features = np.load(aud_file)

    # BIT Test
    aud_len = aud_features.shape[0]
    est_duration_sec = aud_len * 0.040  # assuming 25 fps = 40 ms per frame for DeepSpeech
    print(f"[load_test_data] audio feature frames={aud_len}, est_duration≈{est_duration_sec:.2f}s (25 fps assumed)")

    # Rough expectation of frames to render
    remaining_aud = max(0, aud_len - max(0, aud_start))
    expected_frames = frames_after_skip if test_size == -1 else min(frames_after_skip, max(0, test_size))
    expected_frames = min(expected_frames, remaining_aud)
    print(f"[load_test_data] expected frames to render≈{expected_frames}")

    aud_ids = []
    cur_id = 0
    for frame in meta['frames'][::testskip]:
        poses.append(np.array(frame['transform_matrix']))
        auds.append(
            aud_features[min(aud_start+cur_id, aud_features.shape[0]-1)])
        aud_ids.append(aud_start+cur_id)
        cur_id = cur_id + 1
        if cur_id == test_size or cur_id == aud_features.shape[0]:
            break
    poses = np.array(poses).astype(np.float32)
    auds = np.array(auds).astype(np.float32)
    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
    H, W = bc_img.shape[0], bc_img.shape[1]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    with open(os.path.join(basedir, 'transforms_train.json')) as fp:
    #with open(os.path.join(basedir, test_pose_file)) as fp: # BIT Test: use same pose file as test
        meta_torso = json.load(fp)
    torso_pose = np.array(meta_torso['frames'][0]['transform_matrix'])
    return poses, auds, bc_img, [H, W, focal, cx, cy], aud_ids, torso_pose

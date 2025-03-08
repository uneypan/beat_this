import argparse
from pathlib import Path
import sys
import os
import numpy as np
import torch
import mir_eval
import nevergrad as ng
from pytorch_lightning import seed_everything
from beat_this.dataset import BeatDataModule
from beat_this.inference import load_model, load_checkpoint
from temgo import BFBeatTracker
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# 设置随机种子，保证实验可复现
seed_everything(0, workers=True)


# 全局缓存字典，使用顺序索引作为 key
onset_cache = {}
gt_cache = {}


def preprocess_onset_envelopes(device, stage):
    """
    预处理所有数据的 onset_envelope 与 ground_truth，
    采用顺序索引（样本在 dataloader 中的顺序）作为 key
    """
    print("Precomputing onset_envelopes and ground_truth...")
    if stage == "train":
        train_dataloader = datamodule.train_dataloader()
    elif stage == "val":
        train_dataloader = datamodule.val_dataloader()
    elif stage == "test":
        train_dataloader = datamodule.test_dataloader()
    elif stage == "no_val":
        train_dataloader = datamodule.train_dataloader()
    
    with torch.inference_mode():
        with torch.autocast(enabled=True, device_type=device.type):
            # 采用 enumerate 顺序遍历，确保索引稳定
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100):
                # 每个 batch 只有一个样本（batch_size=1）
                spect = batch["spect"].to(device)[0]
                truth_beat = batch["truth_beat"][0].cpu()
                key = i  # 使用样本顺序作为键
                if key not in onset_cache:
                    out = spect2frames(spect.unsqueeze(0))
                    onset_envelope = out['beat'].cpu().double().sigmoid().numpy()
                    onset_cache[key] = onset_envelope
                if key not in gt_cache:
                    beat_frames = torch.nonzero(truth_beat).squeeze(-1)
                    ground_truth = (beat_frames / 50.0).numpy()
                    gt_cache[key] = ground_truth


def datamodule_setup(checkpoint, num_workers, datasplit):
    """创建数据模块，设置 batch_size 为 1 保证顺序索引稳定"""
    print(f"Creating {datasplit} datamodule")
    data_dir = "/tmp/data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule_hparams["batch_size"] = 1  # 每个 batch 只有一个样本
    datamodule_hparams["augmentations"] = {} # 禁用数据增强
    if datasplit == "no_val":
        datamodule_hparams["no_val"] = True
    datamodule = BeatDataModule(**datamodule_hparams)
    if datasplit == "train" or datasplit == "no_val":
        datamodule.setup(stage="fit")
    elif datasplit == "val":
        datamodule.setup(stage="validate")
    elif datasplit == "test":
        datamodule.setup(stage="test")
    return datamodule


def objective(winsize, P1, P2, align, maxbpm, minbpm, correct, offset):
    """用于 Nevergrad 评估参数效果的目标函数，
    直接从缓存中读取 onset_envelope 和 ground_truth"""
    print(f"Objective: {locals()}")

    def process_batch(onset_envelope, ground_truth):
        """处理单个样本的 F-measure 计算"""
        beat_tracker = BFBeatTracker(
            mode='offline', 
            debug=0, 
            fps=50, 
            winsize=winsize, 
            P1=P1, 
            P2=P2,
            align=align, 
            maxbpm=maxbpm,
            minbpm=minbpm,
            correct=correct,
            multiscale=False
        )
        predicted_beats = np.asarray(beat_tracker(onset_envelope)) + offset
        continuity = mir_eval.beat.continuity(ground_truth, predicted_beats)
        f_measure = mir_eval.beat.f_measure(ground_truth, predicted_beats)
        res = [f_measure, continuity[0], continuity[1], continuity[2], continuity[3]]
        return res

    res = []
    # 使用 as_completed 来迭代已完成的任务，并显示进度
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for key in range(len(onset_cache)):
            onset_envelope = onset_cache[key][0]
            ground_truth = gt_cache[key]
            futures.append(executor.submit(process_batch, onset_envelope, ground_truth))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks", ncols=100):
            res.append(future.result())

    res = np.asarray(res)
    F1, CMLc, CMLt, AMLc, AMLt = res.mean(axis=0)
    print(f"Objective result: F1={F1}, CMLt={CMLt} \n")
    return -(F1 + CMLt)


def main(args):
    for i_model, checkpoint_path in enumerate(args.models):

        global cache_file
        cache_file = f"onset_cache_{checkpoint_path}.npy"  # 根据模型命名缓存文件
        print(f"Using cache file: {cache_file}")

        print(f"Model {i_model+1}/{len(args.models)}")
        global spect2frames
        device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
        spect2frames = load_model(checkpoint_path, device)
        spect2frames.eval()

        checkpoint = load_checkpoint(checkpoint_path)
        global datamodule
        datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)

        # 预处理 onset_envelopes 和 ground_truth 用于训练
        gt_cache.clear()
        onset_cache.clear()
        preprocess_onset_envelopes(device, args.datasplit)


        all_et, all_ed = [], []
        for _, gt in gt_cache.items():
            if  len(gt) < 3: continue
            ibi = np.diff(gt) # inter-beat intervals
            pred = gt[1:-1] + np.mean(ibi) # predicted beat times
            et = gt[2:] - pred # error in timing
            [all_et.append(e) for e in et]
            [all_ed.append(e) for e in ibi]

        all_Pt = np.var(all_et, ddof=1)
        all_Pd = np.var(all_ed, ddof=1)
        
        print(f"all_Pt: {all_Pt}") # 0.00918
        print(f"all_Pd: {all_Pd}") # 0.0547
        

        # 定义搜索空间
        parametrization = ng.p.Instrumentation(
            winsize=1.3,#ng.p.Scalar(lower=1.25, upper=1.4),
            P1=0.02, #ng.p.Log(lower=0.015, upper=0.05),
            P2=0.2, #ng.p.Log(lower=0.20, upper=0.40),
            align=False, # ng.p.Choice([False, True]),
            maxbpm=210,#ng.p.Scalar(lower=200, upper=280),
            minbpm=55,# ng.p.Scalar(lower=40, upper=70),
            correct=False, #ng.p.Choice([False, True]),
            offset=ng.p.Scalar(lower=0.000, upper=0.030),
        )
        

        # 选择优化算法
        optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, 
                                             budget=args.budget,
                                            num_workers=args.num_workers,
                                            )
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            recommendation = optimizer.minimize(objective, 
                                                max_time=args.max_time,
                                                executor=executor,
                                                verbosity=2,
                                                ) 
            
        print(f"best_params_({checkpoint_path}):", recommendation.value)
        # 以文本格式保存
        with open(f"best_params_{checkpoint_path}.txt", "w") as f:
            f.write(str(recommendation.value[1])+ "\n")

        # 重新加载模型，清空缓存，重新预处理验证集
        gt_cache.clear()
        onset_cache.clear()
        if args.datasplit == "test":
            preprocess_onset_envelopes(device, stage="test")
        else:
            preprocess_onset_envelopes(device, stage="val")
        # test the best params
        winsize, P1, P2, align, maxbpm, minbpm, correct, offset = tuple(recommendation.value[1].values())
        res = objective(winsize, P1, P2, align, maxbpm, minbpm, correct, offset)
        print(f"Test result: {res} \n")
        with open(f"best_params_{checkpoint_path}.txt", "a") as f:
            f.write(f"Test result: {res}\n")        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Local checkpoint files to use",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=("train", "val", "test","no_val"),
        default="train",
        help="data split to use: train, val or test (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of data loading workers"
    )
    parser.add_argument(
        "--budget", type=int, default=50, help="number of optimization steps"
    )
    parser.add_argument(
        "--max_time", type=float, default=None, help="max optimazation time in seconds"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args)

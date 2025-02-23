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

def save_cache(cache_file):
    """将 onset_cache 保存到磁盘"""
    print(f"Saving onset_cache to {cache_file} ...")
    np.save(cache_file, onset_cache)

def load_cache(cache_file):
    """从磁盘加载 onset_cache"""
    global onset_cache
    if os.path.exists(cache_file):
        print(f"Loading onset_cache from {cache_file} ...")
        onset_cache = np.load(cache_file, allow_pickle=True).item()
    else:
        onset_cache = {}

def save_gt_cache(cache_file):
    """将 gt_cache 保存到磁盘，缓存文件名与 onset_cache 文件名关联"""
    gt_cache_file = cache_file.replace("onset_cache", "gt_cache")
    print(f"Saving gt_cache to {gt_cache_file} ...")
    np.save(gt_cache_file, gt_cache)

def load_gt_cache(cache_file):
    """从磁盘加载 gt_cache"""
    global gt_cache
    gt_cache_file = cache_file.replace("onset_cache", "gt_cache")
    if os.path.exists(gt_cache_file):
        print(f"Loading gt_cache from {gt_cache_file} ...")
        gt_cache = np.load(gt_cache_file, allow_pickle=True).item()
    else:
        gt_cache = {}

def preprocess_onset_envelopes(device):
    """
    预处理所有数据的 onset_envelope 与 ground_truth，
    采用顺序索引（样本在 dataloader 中的顺序）作为 key
    """
    print("Precomputing onset_envelopes and ground_truth...")
    train_dataloader = datamodule.train_dataloader()
    
    with torch.inference_mode():
        with torch.autocast(enabled=True, device_type=device.type):
            # 采用 enumerate 顺序遍历，确保索引稳定
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
    save_cache(cache_file)
    save_gt_cache(cache_file)

def objective(winsize, P1, P2):
    """用于 Nevergrad 评估参数效果的目标函数，
    直接从缓存中读取 onset_envelope 和 ground_truth"""

    def process_batch(onset_envelope, ground_truth):
        """处理单个样本的 F-measure 计算"""
        beat_tracker = BFBeatTracker(
            mode='offline', 
            debug=0, 
            fps=50, 
            winsize=winsize, 
            P1=P1, 
            P2=P2,
            align=True, 
            maxbpm=210, 
            minbpm=55
        )
        predicted_beats = np.asarray(beat_tracker(onset_envelope))
        return mir_eval.beat.f_measure(ground_truth, predicted_beats)

    total_f_measure = 0
    num_batches = 0
    
    # 使用 as_completed 来迭代已完成的任务，并显示进度
    with ThreadPoolExecutor() as executor:
        futures = []
        for key in range(len(onset_cache)):
            onset_envelope = onset_cache[key][0]
            ground_truth = gt_cache[key]
            futures.append(executor.submit(process_batch, onset_envelope, ground_truth))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            total_f_measure += future.result()
            num_batches += 1

    avg_f_measure = total_f_measure / num_batches
    return -avg_f_measure

def datamodule_setup(checkpoint, num_workers, datasplit):
    """创建数据模块，设置 batch_size 为 1 保证顺序索引稳定"""
    print(f"Creating {datasplit} datamodule")
    data_dir = "/tmp/data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = 'val'
    datamodule_hparams["data_dir"] = data_dir
    datamodule_hparams["batch_size"] = 1  # 每个 batch 只有一个样本
    datamodule_hparams["augmentations"] = {}
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="fit")
    return datamodule

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

        # 尝试加载缓存文件
        load_cache(cache_file)
        load_gt_cache(cache_file)
        # 如果缓存为空，则预处理并存储
        if not onset_cache or not gt_cache:
            preprocess_onset_envelopes(device)

        # 定义搜索空间
        parametrization = ng.p.Instrumentation(
            winsize=ng.p.Scalar(lower=0.5, upper=1.5),
            P1=ng.p.Log(lower=0.001, upper=0.5),
            P2=ng.p.Log(lower=0.5, upper=2.0),
        )

        # 选择优化算法
        optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, 
                                             budget=args.budget,
                                            )
        recommendation = optimizer.minimize(objective, 
                                            verbosity=2, 
                                            max_time=args.max_time
                                            )
        print(f"best_params_({checkpoint_path}):", recommendation.value)
        # 以文本格式保存
        with open(f"best_params_{checkpoint_path}.txt", "w") as f:
            f.write(str(recommendation.value))


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
        choices=("train", "val", "test"),
        default="val",
        help="data split to use: train, val or test (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of data loading workers"
    )
    parser.add_argument(
        "--budget", type=int, default=20, help="number of optimization steps"
    )
    parser.add_argument(
        "--max_time", type=float, default=3600.0, help="max optimazation time in seconds"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args)

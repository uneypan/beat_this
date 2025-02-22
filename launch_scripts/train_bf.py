import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from pytorch_lightning import seed_everything
import torch
import mir_eval
import nevergrad as ng
from beat_this.dataset import BeatDataModule
from beat_this.inference import load_model, load_checkpoint
from temgo import BFBeatTracker
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 设置随机种子，保证实验可复现
seed_everything(0, workers=True)

# 定义搜索空间
parametrization = ng.p.Instrumentation(
    winsize=ng.p.Scalar(lower=0.5, upper=1.5),
    P1=ng.p.Log(lower=0.001, upper=0.1),
    P2=ng.p.Log(lower=0.5, upper=2.0),
)

# 选择优化算法
optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=50)


def objective(winsize, P1, P2):
    """用于 Nevergrad 评估参数效果的目标函数"""


    # 获取数据
    train_dataloader = datamodule.train_dataloader()
    
    # 计算 F-measure 作为优化目标
    def process_batch(onset_envelope, truth_beat):
        """ 处理单个样本的 F-measure 计算 """
        beat_frames = torch.nonzero(truth_beat).squeeze(-1)
        ground_truth = (beat_frames / 50.0).numpy()
        # 直接实例化 BFBeatTracker
        beat_tracker = BFBeatTracker(
            mode='offline', debug=0, fps=50, winsize=winsize, P1=P1, P2=P2,
            align=True, maxbpm=210, minbpm=55
        )        
        predicted_beats = np.asarray(beat_tracker(onset_envelope))  # 预测节拍
        return mir_eval.beat.f_measure(ground_truth, predicted_beats)

    total_f_measure = 0
    num_batches = 0

    with ThreadPoolExecutor() as executor:
        
        futures = []
        for batch in tqdm(train_dataloader):
            with torch.inference_mode():
                with torch.autocast(enabled=True, device_type=torch.device(args.gpu).type):
                    out = spect2frames(batch["spect"].to(device=args.gpu))  # GPU 计算
            onset_envelopes = out['beat'].cpu().double().sigmoid().numpy()  # 转回 CPU

            # 多线程处理 onset_envelopes
            for onset_envelope, truth_beat in zip(onset_envelopes, batch["truth_beat"]):
                futures.append(executor.submit(process_batch, onset_envelope, truth_beat.cpu()))

            # 收集多线程计算结果
            for future in futures:
                total_f_measure += future.result()
                num_batches += 1
            futures = []

    avg_f_measure = total_f_measure / num_batches  # 计算平均 F-measure

    return -avg_f_measure  # 目标是最大化 F-measure，因此取负



def main(args):
    for i_model, checkpoint_path in enumerate(args.models):
        print(f"Model {i_model+1}/{len(args.models)}")

        global spect2frames  # 使 spect2frames 在 objective() 内部可访问
        device = f"cuda:{args.gpu}" if  args.gpu >= 0 else "cpu"
        spect2frames = load_model(checkpoint_path, device)
        spect2frames.eval()

        checkpoint = load_checkpoint(checkpoint_path)
        global datamodule  # 使 datamodule 在 objective() 内部可访问
        datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)

        # 运行 Nevergrad 优化
        recommendation = optimizer.minimize(objective)
        print("最佳参数:", recommendation.value)


def datamodule_setup(checkpoint, num_workers, datasplit):
    """创建数据模块"""
    print(f"Creating {datasplit} datamodule")
    data_dir = "/tmp/data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = 'val'
    datamodule_hparams["data_dir"] = data_dir
    datamodule_hparams["batch_size"] = 64
    datamodule_hparams["augmentations"] = {}
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="fit")
    return datamodule


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
        "--num_workers", type=int, default=8, help="number of data loading workers "
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args)

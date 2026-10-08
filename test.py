import argparse
import os
import re
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.BPFNet import Network
from data import get_dataloader
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


def parse_epoch_from_ckpt(ckpt_path: str):
    """
    从 Net_epoch_78.pth 这类文件名中解析 epoch。
    如果解析失败，返回 0。
    """
    base = os.path.basename(ckpt_path)
    match = re.search(r'Net_epoch_(\d+)\.pth', base)
    if match:
        return int(match.group(1))
    return 0


def _adapt_state_dict_for_model(model, state):
    """
    兼容 all-stage RAFA / 旧 stage3 RAFA checkpoint。

    关键点：
    - 不直接访问 LazyLinear 未初始化参数的 shape
    - polar3.* 可以映射到 polar.3.*
    - 标量 rafa_beta_logit 可以映射到长度为4的 rafa_beta_logit
    """
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model_state = model.state_dict()

    # 旧版 stage3-only: polar3.* -> 新版 all-stage: polar.3.*
    has_new_polar = any(k.startswith("polar.") for k in model_state.keys())
    has_old_polar3 = any(k.startswith("polar3.") for k in state.keys())

    if has_new_polar and has_old_polar3:
        for k, v in list(state.items()):
            if k.startswith("polar3."):
                new_k = k.replace("polar3.", "polar.3.", 1)
                if new_k in model_state:
                    state[new_k] = v

    # 旧版标量 beta -> 新版 4-stage beta
    if "rafa_beta_logit" in state and "rafa_beta_logit" in model_state:
        src = state["rafa_beta_logit"]
        tgt = model_state["rafa_beta_logit"]

        try:
            src_shape = tuple(src.shape)
            tgt_shape = tuple(tgt.shape)
        except RuntimeError:
            src_shape, tgt_shape = None, None

        if src_shape is not None and tgt_shape is not None and src_shape != tgt_shape:
            if src.numel() == 1 and tgt.numel() == 4:
                new_beta = tgt.clone()
                # 旧版只训练过 stage3，所以只继承到 stage3。
                # stage0/1/2 保持当前初始化。
                new_beta[3] = src.reshape(-1)[0].to(dtype=new_beta.dtype)
                state["rafa_beta_logit"] = new_beta
            elif src.numel() == tgt.numel():
                state["rafa_beta_logit"] = src.reshape_as(tgt)
            else:
                print(
                    "⚠️ Drop incompatible key: rafa_beta_logit, "
                    f"ckpt shape={src_shape}, model shape={tgt_shape}"
                )
                state.pop("rafa_beta_logit", None)

    def safe_shape(t):
        """
        LazyLinear 未初始化参数访问 .shape 会 RuntimeError。
        这里返回 None，然后跳过 shape 检查，交给 load_state_dict 初始化。
        """
        try:
            return tuple(t.shape)
        except RuntimeError as e:
            if "uninitialized parameter" in str(e).lower():
                return None
            if "uninitialized" in str(e).lower():
                return None
            raise

    dropped = []

    for k in list(state.keys()):
        if k not in model_state:
            continue

        src_shape = safe_shape(state[k])
        tgt_shape = safe_shape(model_state[k])

        # tgt_shape 为 None 通常是 LazyLinear 参数，不能手动比 shape
        if src_shape is None or tgt_shape is None:
            continue

        if src_shape != tgt_shape:
            dropped.append((k, src_shape, tgt_shape))
            state.pop(k)

    if dropped:
        print("⚠️ Dropped shape-mismatched keys, show up to 20:")
        for k, src_shape, tgt_shape in dropped[:20]:
            print(f"  {k}: ckpt {src_shape} -> model {tgt_shape}")

    return state

def load_checkpoint(model, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 模型文件未找到: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)

    state = _adapt_state_dict_for_model(model, state)

    load_info = model.load_state_dict(state, strict=False)

    print(f"✅ 权重加载成功: {ckpt_path}")

    if len(load_info.missing_keys) > 0:
        print(
            f"⚠️ Missing keys ({len(load_info.missing_keys)}): show up to 20\n  "
            + "\n  ".join(load_info.missing_keys[:20])
        )

    if len(load_info.unexpected_keys) > 0:
        print(
            f"⚠️ Unexpected keys ({len(load_info.unexpected_keys)}): show up to 20\n  "
            + "\n  ".join(load_info.unexpected_keys[:20])
        )

    model.eval()
    return model


def _move_ref_feats_to_device(ref_feats, device, dtype):
    if not (isinstance(ref_feats, (list, tuple)) and len(ref_feats) == 4):
        raise TypeError("ref_feats 必须是长度为4的 tuple/list: (r1, r2, r3, r4)")

    out = []

    for t in ref_feats:
        if not isinstance(t, torch.Tensor):
            raise TypeError("ref_feats 中的每个元素都必须是 torch.Tensor")

        if t.dim() == 3:
            t = t.unsqueeze(0)

        out.append(
            t.to(
                device=device,
                dtype=dtype,
                non_blocking=True
            )
        )

    return tuple(out)


def _gt_to_numpy(gts):
    """
    将 dataloader 产出的 gts 转为 [H,W] float32 numpy，并归一化到 [0,1]。
    """
    if isinstance(gts, list):
        gt_np = gts[0]
        if isinstance(gt_np, torch.Tensor):
            gt_np = gt_np.squeeze().cpu().numpy()
        gt_np = np.asarray(gt_np)

    elif isinstance(gts, np.ndarray):
        gt_np = gts

    elif isinstance(gts, torch.Tensor):
        gt_np = gts.squeeze().cpu().numpy()

    else:
        gt_np = np.asarray(gts)

    gt_np = gt_np.astype(np.float32)

    if gt_np.ndim == 3:
        if gt_np.shape[-1] == 1:
            gt_np = gt_np[..., 0]
        elif gt_np.shape[0] == 1:
            gt_np = gt_np[0, ...]

    if gt_np.ndim != 2:
        raise RuntimeError(f"GT shape unexpected: {gt_np.shape}")

    gt_np = gt_np / (gt_np.max() + 1e-8)

    return gt_np


@torch.no_grad()
def test_model(test_loader, model, device):
    Sm = Smeasure()
    Em = Emeasure()
    Fm = Fmeasure()
    wFm = WeightedFmeasure()
    mae = MAE()

    model.eval()

    with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
        for batch in test_loader:
            images, gts, ref_feats, _ = batch

            if isinstance(images, torch.Tensor) and images.size(0) != 1:
                raise RuntimeError("该评测脚本默认 test batchsize=1，以兼容 numpy GT。")

            images = images.to(device, non_blocking=True).float()
            ref_feats = _move_ref_feats_to_device(
                ref_feats,
                images.device,
                torch.float32
            )

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(
                    images,
                    ref_feats,
                    y=None,
                    training=False
                )

                s3, s2, s1, s0 = outputs[:4]

            gt_np = _gt_to_numpy(gts)
            h, w = gt_np.shape[:2]

            pred = F.interpolate(
                s0,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            pred = torch.sigmoid(pred)
            pred = pred.squeeze().detach().cpu().numpy().astype(np.float32)

            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            pred_img = pred * 255.0
            gt_img = gt_np * 255.0

            Sm.step(pred=pred_img, gt=gt_img)
            Em.step(pred=pred_img, gt=gt_img)
            Fm.step(pred=pred_img, gt=gt_img)
            wFm.step(pred=pred_img, gt=gt_img)
            mae.step(pred=pred_img, gt=gt_img)

            pbar.update(1)

    results = {
        "Smeasure": float(Sm.get_results()["sm"]),
        "WeightedFmeasure": float(wFm.get_results()["wfm"]),
        "MAE": float(mae.get_results()["mae"]),
        "adpEm": float(Em.get_results()["em"]["adp"]),
        "meanEm": float(Em.get_results()["em"]["curve"].mean()),
        "maxEm": float(Em.get_results()["em"]["curve"].max()),
        "adpFm": float(Fm.get_results()["fm"]["adp"]),
        "meanFm": float(Fm.get_results()["fm"]["curve"].mean()),
        "maxFm": float(Fm.get_results()["fm"]["curve"].max()),
    }

    results["Score"] = (
        results["Smeasure"]
        + results["WeightedFmeasure"]
        - results["MAE"]
    )

    return results


def save_results_csv(results_all, csv_path):
    if not results_all:
        return

    epochs = sorted(results_all.keys())
    metric_names = list(results_all[epochs[0]].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch"] + metric_names)

        for epoch in epochs:
            row = [epoch] + [results_all[epoch][k] for k in metric_names]
            writer.writerow(row)

    print(f"📄 CSV 结果已保存: {csv_path}")


def draw_curve(results_all, out_dir, exp_name):
    epochs = sorted(results_all.keys())

    S_list = [results_all[e]['Smeasure'] for e in epochs]
    Wf_list = [results_all[e]['WeightedFmeasure'] for e in epochs]
    MAE_list = [results_all[e]['MAE'] for e in epochs]
    Score_list = [results_all[e]['Score'] for e in epochs]

    plt.figure(figsize=(9, 5))

    plt.plot(epochs, S_list, 'o-', label='Smeasure')
    plt.plot(epochs, Wf_list, 's-', label='WeightedF')
    plt.plot(epochs, MAE_list, '^-', label='MAE')
    plt.plot(epochs, Score_list, 'd-', label='Score = S + Wf - MAE')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(f'{exp_name} Performance Curve')
    plt.grid(True)
    plt.legend()

    curve_path = os.path.join(out_dir, f'{exp_name}_curve.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 曲线已保存: {curve_path}")

    best_epoch = max(results_all, key=lambda e: results_all[e]['Score'])
    best_metrics = results_all[best_epoch]

    print(f"\n🏆 最优模型 by Score: epoch {best_epoch}")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='UAT_main_RAFA_ALLSTAGE_FINETUNE_68to78')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--imgsize', type=int, default=352)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/dataset/R2C7K')

    # 注意：这里是 checkpoint 所在目录，不是总 snapshot 目录
    parser.add_argument(
        '--save_root',
        type=str,
        default='/root/autodl-tmp/snapshot/UAT_main_RAFA_ALLSTAGE_FINETUNE_68to78'
    )

    parser.add_argument(
        '--epochs',
        type=str,
        default='70,72,74,76,78',
        help='多个 epoch 用逗号分隔，例如 70,72,74,76,78'
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default='',
        help='单个权重测试路径；如果设置该参数，则忽略 --save_root 和 --epochs'
    )

    parser.add_argument('--pvt_weights', type=str, default='./pvt_weights/pvt_v2_b2.pth')

    # -----------------------------
    # 新版 RAFA 弱注入参数
    # -----------------------------
    parser.add_argument(
        '--use_polar',
        type=int,
        default=1,
        help='1: 启用 BTCAF 后 RAFA 弱注入；0: 关闭 RAFA'
    )

    parser.add_argument(
        '--rafa_stages',
        type=str,
        default='0123',
        help='启用 RAFA 的 stage，例如 3 / 23 / 123 / 0123。all-stage 用 0123'
    )

    parser.add_argument(
        '--rafa_max_beta',
        type=float,
        default=0.01,
        help='RAFA 弱残差注入最大比例，应与训练时保持一致'
    )

    parser.add_argument(
        '--rafa_beta_init',
        type=float,
        default=0.001,
        help='RAFA 弱残差注入初始比例，应与训练时保持一致；加载 checkpoint 后主要影响缺失的新 stage 参数'
    )

    parser.add_argument(
        '--rafa_gamma_init',
        type=float,
        default=0.1,
        help='RAFA 内部 gamma 初始值，应与训练时保持一致；加载 checkpoint 后主要影响缺失的新 stage 参数'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='/root/autodl-tmp/snapshot/eval_results'
    )

    opt = parser.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True

    print(f"✅ Using device: {device}, visible gpu: {opt.gpu_id}")

    test_loader = get_dataloader(
        opt.data_root,
        opt.shot,
        opt.imgsize,
        opt.batchsize,
        opt.num_workers,
        mode='test'
    )

    exp_name = (
        f'{opt.model_name}'
        f'_usepolar{opt.use_polar}'
        f'_stages{opt.rafa_stages}'
        f'_betamax{opt.rafa_max_beta}'
        f'_betainit{opt.rafa_beta_init}'
    )

    exp_dir = os.path.join(opt.out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    results_all = {}

    if opt.ckpt:
        epochs = [parse_epoch_from_ckpt(opt.ckpt)]
    else:
        epochs = [int(e.strip()) for e in opt.epochs.split(',') if e.strip()]

    results_file_path = os.path.join(exp_dir, f'results_{exp_name}.txt')
    results_csv_path = os.path.join(exp_dir, f'results_{exp_name}.csv')

    if os.path.exists(results_file_path):
        old_path = results_file_path.replace(".txt", "_old.txt")
        os.rename(results_file_path, old_path)
        print(f"⚠️ 旧 txt 结果已重命名: {old_path}")

    if os.path.exists(results_csv_path):
        old_path = results_csv_path.replace(".csv", "_old.csv")
        os.rename(results_csv_path, old_path)
        print(f"⚠️ 旧 csv 结果已重命名: {old_path}")

    for epoch in epochs:
        if opt.ckpt:
            ckpt_path = opt.ckpt
        else:
            ckpt_path = os.path.join(
                opt.save_root,
                f'Net_epoch_{epoch}.pth'
            )

        if not os.path.exists(ckpt_path):
            print(f"⚠️ 权重文件不存在: {ckpt_path}")
            continue

        print(f"\n==============================")
        print(f"🚀 Evaluating epoch {epoch}")
        print(f"📌 Checkpoint: {ckpt_path}")
        print(f"==============================")

        model = Network(opt).to(device)
        model = load_checkpoint(model, ckpt_path)

        metrics = test_model(test_loader, model, device)
        results_all[epoch] = metrics

        with open(results_file_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch}: {metrics}\n")

        print(f"✅ Epoch {epoch} Score = {metrics['Score']:.4f}")

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        del model
        torch.cuda.empty_cache()

    if results_all:
        save_results_csv(results_all, results_csv_path)
        draw_curve(results_all, exp_dir, exp_name)
    else:
        print("⚠️ 没有任何 epoch 被成功评测，请检查 save_root / epochs / ckpt 是否正确。")


if __name__ == "__main__":
    main()

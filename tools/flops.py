"""
measure_model.py
统计 ConnectUnet 的参数量、FLOPs 和峰值显存。

依赖：pip install torch
运行：python measure_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from connectunets import ConnectUnet


# =========================================================
# 1) FLOPs 统计：monkey-patch F.conv2d / F.linear
#    这样对 Convde 中的 F.conv2d(x, de_f) 也能精确统计
# =========================================================
_orig_conv2d = F.conv2d
_orig_linear = F.linear
_flops_counter = {"total": 0}


def _conv2d_with_flops(input, weight, bias=None,
                        stride=1, padding=0, dilation=1, groups=1):
    out = _orig_conv2d(input, weight, bias, stride, padding, dilation, groups)
    B = input.shape[0]
    H, W = out.shape[-2], out.shape[-1]
    Cout = weight.shape[0]
    Cin_g = weight.shape[1]
    kh, kw = weight.shape[-2], weight.shape[-1]
    # 1 次乘加 = 2 FLOPs
    _flops_counter["total"] += 2 * B * H * W * Cout * Cin_g * kh * kw
    return out


def _linear_with_flops(input, weight, bias=None):
    out = _orig_linear(input, weight, bias)
    in_features = weight.shape[1]
    _flops_counter["total"] += 2 * out.numel() * in_features
    return out


@contextmanager
def flops_ctx():
    """进入上下文后，所有 F.conv2d / F.linear 都会被统计"""
    F.conv2d = _conv2d_with_flops
    F.linear = _linear_with_flops
    _flops_counter["total"] = 0
    try:
        yield _flops_counter
    finally:
        F.conv2d = _orig_conv2d
        F.linear = _orig_linear


# =========================================================
# 2) 峰值显存
# =========================================================
def peak_memory(model, x, train=False):
    model.train(train)
    x = x.clone().detach()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if train:
        # 前向 + 反向（真实训练的显存峰值）
        seg_list, feat = model(x)
        loss = sum(o.mean() for o in seg_list) + feat.mean()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        model.zero_grad(set_to_none=True)
        del loss, seg_list, feat
    else:
        with torch.no_grad():
            out = model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        del out

    torch.cuda.empty_cache()
    return peak


# =========================================================
# 3) 主流程
# =========================================================
def main():
    if not torch.cuda.is_available():
        print("⚠ CUDA 不可用，仅统计 FLOPs / 参数量，不测量显存")
        device = "cpu"
    else:
        device = "cuda"

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False  # 显存测量更稳定

    input_shape = (1, 3, 512, 512)
    input_tensor = torch.randn(*input_shape, device=device)

    model = ConnectUnet(unetn=1, base_filters=32).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device            : {device}")
    print(f"Total params      : {total_params:,}")
    print(f"Trainable params  : {train_params:,}")

    # ---------- 推理 FLOPs (eval 模式, 无梯度) ----------
    model.eval()
    with torch.no_grad():
        with flops_ctx() as counter:
            _ = model(input_tensor)
    inf_flops = counter["total"]
    print(f"FLOPs (inference) : {inf_flops / 1e9:.3f} GFLOPs  ({inf_flops / 1e12:.3f} TFLOPs)")
    del _
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- 训练 FLOPs (train 模式, Convde 内注意力会执行) ----------
    model.train()
    with flops_ctx() as counter:
        _ = model(input_tensor)
    train_flops = counter["total"]
    model.zero_grad(set_to_none=True)
    print(f"FLOPs (training)  : {train_flops / 1e9:.3f} GFLOPs  "
          f"(forward only, 包含 Convde 内部注意力)")
    del _
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- 峰值显存 ----------
    if device == "cuda":
        eval_peak = peak_memory(model, input_tensor, train=False)
        print(f"Peak GPU mem (inference)      : {eval_peak:.3f} GB")

        train_peak = peak_memory(model, input_tensor, train=True)
        print(f"Peak GPU mem (training fwd+bwd): {train_peak:.3f} GB")


if __name__ == "__main__":
    main()
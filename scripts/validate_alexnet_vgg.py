"""
validate_alexnet_vgg.py

运行前验证 AlexNet / VGG CIFAR 版本的接口正确性：
  1. forward 输出 shape
  2. get_feature(f_idx) 输出 shape（DSDM last_feature）
  3. 参数量
  4. 一步 SGD 训练不报错

用法：
    conda run -n sp python scripts/validate_alexnet_vgg.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

PASS = "✅"
FAIL = "❌"


def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    if not cond:
        raise AssertionError(msg)


def validate_model(name, model, f_idx, expected_feat_shape, nclass=100):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    model.eval()
    x = torch.zeros(2, 3, 32, 32)

    # --- forward ---
    with torch.no_grad():
        out = model(x)
    check(out.shape == (2, nclass),
          f"forward output shape = {tuple(out.shape)}，期望 (2, {nclass})")

    # --- get_feature ---
    # get_feature 返回 (feature_list, None)，feature_list[0] 是对应层的 tensor
    with torch.no_grad():
        feat_list, _ = model.get_feature(x, f_idx, f_idx)
    feat = feat_list[0]
    check(feat.shape == (2, *expected_feat_shape),
          f"get_feature(idx={f_idx}) shape = {tuple(feat.shape)}，期望 (2, {expected_feat_shape[0]})")

    # --- 参数量 ---
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  ℹ  参数量: {nparams:,}")

    # --- 一步 SGD 训练 ---
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    labels = torch.zeros(2, dtype=torch.long)
    optimizer.zero_grad()
    loss = criterion(model(x), labels)
    loss.backward()
    optimizer.step()
    check(True, f"一步 SGD 训练完成，loss={loss.item():.4f}")

    print(f"  {PASS}  {name} 所有检查通过\n")


def main():
    print("\n[validate_alexnet_vgg.py]  AlexNet / VGG CIFAR 接口验证")

    # --- AlexNet ---
    import DSDM.models.alexnet_cifar as AN
    alexnet = AN.alexnet_cifar(num_classes=100, nch=3)
    # f_idx=7 对应 logits 前一层 Linear(1024→512)，输出 [B,512]
    validate_model("AlexNetCIFAR", alexnet, f_idx=7,
                   expected_feat_shape=(512,), nclass=100)

    # --- VGG ---
    import DSDM.models.vgg_cifar as VN
    vgg = VN.vgg_cifar(num_classes=100, nch=3)
    # f_idx=10 对应 logits 前一层 Linear(512→512)，输出 [B,512]
    validate_model("VGG11-CIFAR", vgg, f_idx=10,
                   expected_feat_shape=(512,), nclass=100)

    # --- 与 DSDM 蒸馏接口一致性检查 ---
    print("=" * 50)
    print("  DSDM 蒸馏参数一致性检查")
    print("=" * 50)
    alexnet.eval()
    vgg.eval()
    x = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        af_list, _ = alexnet.get_feature(x, 7, 7)
        vf_list, _ = vgg.get_feature(x, 10, 10)
    af, vf = af_list[0], vf_list[0]
    check(af.shape == (2, 512),
          f"AlexNet get_feature(7,7) = {tuple(af.shape)}")
    check(vf.shape == (2, 512),
          f"VGG get_feature(10,10) = {tuple(vf.shape)}")

    print(f"\n{PASS}  全部验证通过，AlexNet 和 VGG 可用于社会化学习 pipeline。\n")


if __name__ == "__main__":
    main()

"""
作用：
    统一管理项目中的 model pool，根据配置名称构建不同 backbone。
    当前优先复用 DSDM 体系中的 conv / resnet / resnet_ap，保证与蒸馏侧模型风格一致。

主要部分：
    1. build_backbone(name, ...):
       - 根据字符串名称返回对应 backbone 实例
    2. get_default_feature_dim(name):
       - 返回当前 backbone 的默认特征维度（先做占位，后续可精细化）

输入输出：
    - 输入：模型名称、数据集名称、类别数、图像大小、归一化类型等
    - 输出：对应的 backbone 模型实例
"""

from typing import Tuple

from src.models.backbones.convnet import ConvNet
from src.models.backbones.resnet import ResNet
from src.models.backbones.resnet_ap import ResNetAP


def build_backbone(
    name: str,
    dataset: str,
    num_classes: int,
    image_size: Tuple[int, int] = (32, 32),
    norm_type: str = "instance",
):
    """
    根据名称构建 backbone。

    参数：
        name:
            模型名称，可选：
            - "conv"
            - "resnet"
            - "resnet_ap"

        dataset:
            数据集名称，例如 "cifar10" / "cifar100"

        num_classes:
            全局类别数

        image_size:
            输入图像尺寸，默认 (32, 32)

        norm_type:
            归一化方式，默认 "instance"

    返回：
        对应 backbone 模型实例
    """
    if name == "conv":
        return ConvNet(
            num_classes=num_classes,
            net_norm=norm_type,
            net_depth=3,
            net_width=128,
            channel=3,
            net_act="relu",
            net_pooling="avgpooling",
            im_size=image_size,
        )

    if name == "resnet":
        return ResNet(
            dataset=dataset,
            depth=20 if dataset.startswith("cifar") else 18,
            num_classes=num_classes,
            norm_type=norm_type,
            size=image_size[0],
            nch=3,
        )

    if name == "resnet_ap":
        return ResNetAP(
            dataset=dataset,
            depth=20 if dataset.startswith("cifar") else 18,
            num_classes=num_classes,
            norm_type=norm_type,
            size=image_size[0],
            nch=3,
        )

    raise ValueError(f"未知 backbone 名称: {name}")


def get_default_feature_dim(name: str) -> int:
    """
    返回 backbone 默认特征维度。

    说明：
        这里先提供一个占位接口，后续在 agent_model 中如果需要更精确的
        feature_dim 管理，可以再细化。

    参数：
        name: backbone 名称

    返回：
        默认特征维度
    """
    if name == "conv":
        return 2048   # 128 * 4 * 4（32x32, 3次pool后的典型展平大小）
    if name in ["resnet", "resnet_ap"]:
        return 256
    raise ValueError(f"未知 backbone 名称: {name}")
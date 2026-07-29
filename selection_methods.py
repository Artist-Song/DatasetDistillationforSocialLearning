from collections import defaultdict
import hashlib
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


FAST_OFFICIAL_COMMIT = "6a218fcfdc93838634921399b0de6a36cdd29756"


def _get_target(train_set, index):
    """读取数据集样本标签，兼容 targets 和 labels。"""
    if hasattr(train_set, "targets"):
        target = train_set.targets[index]
    elif hasattr(train_set, "labels"):
        target = train_set.labels[index]
    else:
        _, target = train_set[index]
    return int(target)


def _collect_indices_by_class(train_set):
    """按类别收集真实训练集样本索引。"""
    indices_by_class = defaultdict(list)
    for index in range(len(train_set)):
        indices_by_class[_get_target(train_set, index)].append(index)
    return indices_by_class


def _stack_samples(train_set, indices):
    """根据索引从真实训练集中堆叠图片和标签。"""
    images = []
    labels = []
    for index in indices:
        image, label = train_set[index]
        images.append(image.detach().cpu() if torch.is_tensor(image) else image)
        labels.append(int(label))
    return torch.stack(images).float(), torch.tensor(labels, dtype=torch.long)


def build_heuristic_packet(args, train_set):
    """每类随机选择 ipc 张真实样本，构建 Heuristic packet。"""
    generator = torch.Generator().manual_seed(int(args.seed))
    indices_by_class = _collect_indices_by_class(train_set)
    selected = []
    class_ids = sorted(indices_by_class.keys())
    for class_id in class_ids:
        candidates = indices_by_class[class_id]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected.extend([candidates[i] for i in order[: args.ipc]])
    images, labels = _stack_samples(train_set, selected)
    return images, labels, class_ids


def _score_confidence(args, images, guide_models, device):
    """用 guide ensemble 计算每张图片的平均最大 softmax 置信度。"""
    batch = images.to(device)
    try:
        from data import MEANS, STDS

        mean = torch.tensor(MEANS[args.dataset], device=device).view(1, -1, 1, 1)
        std = torch.tensor(STDS[args.dataset], device=device).view(1, -1, 1, 1)
        batch = (batch - mean) / std
    except Exception:
        pass
    scores = torch.zeros(batch.shape[0], device=device)
    for model in guide_models:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
            scores += probs.max(dim=1).values
    scores = scores / max(1, len(guide_models))
    return scores.detach().cpu()


def build_importance_packet(args, train_set, guide_models):
    """使用 guide ensemble 选择低置信度样本，构建 Importance packet。"""
    if not guide_models:
        images, labels, class_ids = build_heuristic_packet(args, train_set)
        return images, labels, class_ids

    device = torch.device("cuda" if getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    for model in guide_models:
        model.to(device)

    indices_by_class = _collect_indices_by_class(train_set)
    selected = []
    class_ids = sorted(indices_by_class.keys())
    for class_id in class_ids:
        candidates = indices_by_class[class_id]
        images, _ = _stack_samples(train_set, candidates)
        scores = _score_confidence(args, images, guide_models, device)
        order = torch.argsort(scores, descending=False).tolist()
        selected.extend([candidates[i] for i in order[: args.ipc]])
    images, labels = _stack_samples(train_set, selected)
    return images, labels, class_ids


def _load_fast_official_selector(repo_path):
    """以独立包名加载 FAST 官方核心，避免与其他仓库的 modules 包冲突。"""
    modules_dir = Path(repo_path).resolve() / "modules"
    init_path = modules_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"缺少 FAST 官方源码: {init_path}")

    package_name = "_fast_official_modules"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(modules_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 FAST 官方源码: {init_path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    module = importlib.import_module(f"{package_name}.per_class")
    return module.run_per_class_selection


def _validate_fast_positions(positions, num_candidates, ipc, class_id):
    """检查 FAST 返回的类内索引满足精确 IPC、唯一性和范围约束。"""
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size != int(ipc):
        raise ValueError(f"FAST class {class_id} 返回 {positions.size} 张，期望 IPC={ipc}")
    if np.unique(positions).size != positions.size:
        raise ValueError(f"FAST class {class_id} 返回了重复索引")
    if positions.size and (positions.min() < 0 or positions.max() >= int(num_candidates)):
        raise ValueError(f"FAST class {class_id} 返回了越界索引")
    return positions


def _fast_repo_provenance(args):
    repo_path = Path(getattr(args, "fast_repo_path", "external_baselines/repos/FAST")).resolve()
    expected_commit = str(getattr(args, "fast_commit", FAST_OFFICIAL_COMMIT))
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise ValueError(f"FAST HEAD={actual_commit} differs from configured commit={expected_commit}")
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(repo_path), "diff", "--binary", "--no-ext-diff"], text=True
    )
    patch_sha256 = hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
    return {
        "repo_path": str(repo_path),
        "official_commit": actual_commit,
        "tracked_patch_present": bool(tracked_diff),
        "tracked_patch_sha256": patch_sha256,
    }


def _fast_cache_path(args, class_id, provenance):
    cache_root = Path(
        getattr(args, "fast_cache_root", "external_baselines/outputs/fast_cache")
    )
    commit = provenance["official_commit"][:12]
    patch = provenance["tracked_patch_sha256"][:12] if provenance["tracked_patch_present"] else "clean"
    return (
        cache_root
        / f"{args.dataset}_pixels_{commit}_{patch}"
        / f"ipc{int(args.ipc)}"
        / f"class_{int(class_id):03d}.npz"
    )


def _select_fast_class_positions(args, class_id, class_images, provenance):
    """运行 FAST 官方 per-class/pixels/minmax 选择并缓存类内索引。"""
    num_candidates = int(class_images.shape[0])
    cache_path = _fast_cache_path(args, class_id, provenance)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if int(cached["num_candidates"]) != num_candidates:
                raise ValueError(f"FAST cache 样本数不匹配: {cache_path}")
            if str(cached["fast_commit"].item()) != provenance["official_commit"]:
                raise ValueError(f"FAST cache commit 不匹配: {cache_path}")
            if str(cached["tracked_patch_sha256"].item()) != provenance["tracked_patch_sha256"]:
                raise ValueError(f"FAST cache patch SHA 不匹配: {cache_path}")
            positions = cached["selected_positions"]
        return _validate_fast_positions(positions, num_candidates, args.ipc, class_id), cache_path, True

    repo_path = getattr(args, "fast_repo_path", "external_baselines/repos/FAST")
    selector = _load_fast_official_selector(repo_path)
    images_np = class_images.detach().cpu().numpy().astype(np.float32, copy=False)
    x_pixels = images_np.reshape(num_candidates, -1)
    y_class = np.full(num_candidates, int(class_id), dtype=np.int64)
    algorithm_seed = int(getattr(args, "fast_seed", 0))
    np.random.seed(algorithm_seed)
    selected, _, _ = selector(
        X=x_pixels,
        y_all=y_class,
        retain_ratio=float(args.ipc) / float(num_candidates),
        method="minmax",
        rff=None,
        stage2_mmd=0.0,
        per_class_retain_ratio=float(args.ipc) / float(num_candidates),
        min_samples_per_class=int(args.ipc),
        outdir=None,
        save_stage_files=False,
        verbose=False,
        viz_manager=None,
    )
    positions = _validate_fast_positions(selected, num_candidates, args.ipc, class_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        selected_positions=positions,
        class_id=np.int64(class_id),
        num_candidates=np.int64(num_candidates),
        ipc=np.int64(args.ipc),
        algorithm_seed=np.int64(algorithm_seed),
        fast_commit=np.asarray(str(getattr(args, "fast_commit", FAST_OFFICIAL_COMMIT))),
        tracked_patch_sha256=np.asarray(provenance["tracked_patch_sha256"]),
        x_source=np.asarray("pixels"),
    )
    return positions, cache_path, False


def build_fast_packet(args, train_set):
    """用 FAST 官方 pixels/per-class/minmax 模式选择真实图 packet。"""
    provenance = _fast_repo_provenance(args)
    indices_by_class = _collect_indices_by_class(train_set)
    class_ids = sorted(indices_by_class.keys())
    selected = []
    cache_hits = 0
    cache_paths = []
    for class_id in class_ids:
        candidates = indices_by_class[class_id]
        if len(candidates) < int(args.ipc):
            raise ValueError(f"class {class_id} 只有 {len(candidates)} 张，无法选择 IPC={args.ipc}")
        class_images, _ = _stack_samples(train_set, candidates)
        positions, cache_path, cache_hit = _select_fast_class_positions(
            args, class_id, class_images, provenance
        )
        selected.extend(candidates[int(position)] for position in positions)
        cache_paths.append(str(cache_path))
        cache_hits += int(cache_hit)

    images, labels = _stack_samples(train_set, selected)
    dataset_indices = [
        int(train_set.indices[index]) if hasattr(train_set, "indices") else int(index)
        for index in selected
    ]
    meta = {
        "selector": "FAST",
        "official_commit": str(getattr(args, "fast_commit", FAST_OFFICIAL_COMMIT)),
        "repo_path": provenance["repo_path"],
        "tracked_patch_present": provenance["tracked_patch_present"],
        "tracked_patch_sha256": provenance["tracked_patch_sha256"],
        "official_mode": "pixels/per_class/minmax",
        "algorithm_seed": int(getattr(args, "fast_seed", 0)),
        "selected_dataset_indices": dataset_indices,
        "cache_hits": cache_hits,
        "cache_entries": len(cache_paths),
    }
    return images, labels, class_ids, meta


def build_full_real_packet(args, train_set):
    """使用 agent 的全部真实训练样本构建 full_real packet。"""
    indices = list(range(len(train_set)))
    images, labels = _stack_samples(train_set, indices)
    class_ids = sorted({int(label) for label in labels.tolist()})
    return images, labels, class_ids

"""扫描本地 WeedMap 目录，帮助理解文件结构；不下载或修改数据。"""

import argparse
import re
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
KEYWORDS = ("RGB", "NDVI", "NIR", "RE", "G", "R", "B", "GroundTruth", "iMap", "mask")


def filename_keywords(filename):
    """识别文件名中的候选关键词，不把普通单词中的 r/g/b 当作波段。"""
    # 把下划线、横线、空格等分隔符作为边界，并拆开常见驼峰命名。
    stem = Path(filename).stem
    separated = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    tokens = set(re.findall(r"[a-z]+", separated.lower()))
    found = set()
    for keyword in KEYWORDS:
        if keyword in ("RE", "G", "R", "B"):
            # 短波段名要求是独立标记，例如 tile_R_001.tif。
            matched = keyword.lower() in tokens
        else:
            matched = keyword.lower() in stem.lower()
        if matched:
            found.add(keyword)
    return found


def main():
    parser = argparse.ArgumentParser(description="递归检查本地 WeedMap 数据结构。")
    parser.add_argument("--data-root", type=Path, default=Path("data/weedmap"),
                        help="数据目录，默认 data/weedmap（相对于当前工作目录）")
    args = parser.parse_args()
    root = args.data_root.expanduser()

    if not root.exists():
        print(f"数据路径不存在：{root.resolve()}")
        print("目前不需要真实数据。以后请把 WeedMap Tiles 数据放到 data/weedmap/，")
        print("再运行本脚本；也可以通过 --data-root 指定实际目录。")
        return
    if not root.is_dir():
        print(f"指定路径不是文件夹：{root.resolve()}，请通过 --data-root 指定目录。")
        return

    suffix_counts = Counter()
    keyword_counts = Counter()
    examples = []
    total_files = 0
    try:
        # 递归扫描子目录；仅保留前 30 个示例，避免保存全部文件路径。
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            total_files += 1
            if len(examples) < 30:
                examples.append(path.relative_to(root))
            suffix_counts[path.suffix.lower()] += 1
            # 每个文件对同一个关键词最多计数一次；可同时匹配多个关键词。
            keyword_counts.update(filename_keywords(path.name))
    except OSError as error:
        print(f"扫描遇到文件访问错误：{error}；以下结果可能不完整。")

    print(f"扫描目录：{root.resolve()}")
    print(f"文件总数：{total_files}")
    print("\n常见图片后缀数量（不区分大小写）：")
    for suffix in IMAGE_SUFFIXES:
        print(f"  {suffix}: {suffix_counts[suffix]}")
    print("\n前 30 个文件路径示例（相对于数据目录）：")
    for path in examples:
        print(f"  {path}")
    if not examples:
        print("  未发现文件。")
    print("\n根据文件名推测的通道/标签文件数量：")
    for keyword in KEYWORDS:
        print(f"  {keyword}: {keyword_counts[keyword]}")
    print("注意：关键词统计仅是线索，可能重复或漏检；不解析影像内容。")
    print("RE/G/R/B 按独立标记匹配，实际通道与标签含义需核实数据说明。")


if __name__ == "__main__":
    main()

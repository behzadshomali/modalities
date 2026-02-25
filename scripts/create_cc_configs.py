#!/usr/bin/env python3
"""
Create CC (Common Crawl) versions of training config files.

Changes applied:
  1. Training data paths: Nemotron-CC-Math-v1_train_4plus_partN.pbin
     → Nemotron-CC-highQuality-sampled_data_part_N.pbin  (for non-eval parts)
  2. WandB project name: appends '_CC' suffix

Usage:
    python scripts/create_cc_configs.py config1.yaml config2.yaml ...
    python scripts/create_cc_configs.py --output-dir /path/to/output config1.yaml
"""

import argparse
import re
import sys
from pathlib import Path


def convert_to_cc(content: str) -> str:
    """Convert a math config's content to its CC version."""

    # 1. Replace math data paths with CC data paths (non-eval parts only)
    #    Nemotron-CC-Math-v1_train_4plus_partN.pbin → Nemotron-CC-highQuality-sampled_data_part_N.pbin
    content = re.sub(
        r"Nemotron-CC-Math-v1_train_4plus_part(\d+)\.pbin",
        r"Nemotron-CC-highQuality-sampled_data_part_\1.pbin",
        content,
    )

    # 2. Append '_CC' to the wandb project name (avoid double-appending)
    content = re.sub(
        r"(project:\s+)(.+?)(?<!_CC)\s*$",
        r"\1\2_CC",
        content,
        flags=re.MULTILINE,
    )

    return content


def main():
    # parser = argparse.ArgumentParser(
    #     description="Create CC versions of training config files."
    # )
    # parser.add_argument(
    #     "configs",
    #     nargs="+",
    #     type=Path,
    #     help="Path(s) to the source YAML config file(s).",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     type=Path,
    #     default=None,
    #     help="Directory to save CC configs. Defaults to a 'cc_versions' sibling folder "
    #     "next to each input config.",
    # )
    # parser.add_argument(
    #     "--suffix",
    #     type=str,
    #     default="_CC",
    #     help="Suffix to add to the output filename (default: '_CC').",
    # )
    # parser.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     help="Print the converted content to stdout instead of writing files.",
    # )

    # args = parser.parse_args()

    configs = [
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_01.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_001.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_05.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_005.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_0005.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_10.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_15.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_20.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_40.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_50.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_60.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_70.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_80.yaml",
        "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/future_mask_configs/3MTP_3BlockSizePonder_MultiplyBias_futureMask_90.yaml",
    ]

    output_dir = "/projects/p_gptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_w_CC/"

    for config_path in configs:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[ERROR] File not found: {config_path}", file=sys.stderr)
            continue

        content = config_path.read_text()
        cc_content = convert_to_cc(content)


        # Determine output path
        out_name = f"{config_path.stem}_CC{config_path.suffix}"
        out_dir = Path(output_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name

        out_path.write_text(cc_content)
        print(f"[OK] {config_path.name} → {out_path}")


if __name__ == "__main__":
    main()

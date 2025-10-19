#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path

# SepTDA baseline results for comparison
SepTDA = {
    '2mix_with_true_numspk': {'SISNR_improvement': 23.6},
    '2mix_without_true_numspk': {'SISNR_improvement': 23.6, "Est_num_spk": 99.9},
    '3mix_with_true_numspk': {'SISNR_improvement': 23.5},
    '3mix_without_true_numspk': {'SISNR_improvement': 22.1, "Est_num_spk": 95.93},
    '4mix_with_true_numspk': {'SISNR_improvement': 22.0},
    '4mix_without_true_numspk': {'SISNR_improvement': 19.5, "Est_num_spk": 90.10},
    '5mix_with_true_numspk': {'SISNR_improvement': 21.0},
    '5mix_without_true_numspk': {'SISNR_improvement': 16.9, "Est_num_spk": 83.23},
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate SI-SNR improvement for speech enhancement experiments'
    )
    parser.add_argument(
        'inference_dir',
        type=str,
        help='Path to inference directory (e.g., exp/enh_xxx/enhanced_tt_min_8k/enh)'
    )
    parser.add_argument(
        '--data_feats_dir',
        type=str,
        default=None,
        help='Path to data features directory (default: auto-detect from inference_dir)'
    )
    parser.add_argument(
        '--mix_nums',
        type=str,
        default='2,3,4,5',
        help='Comma-separated list of speaker numbers to evaluate (default: 2,3,4,5)'
    )
    parser.add_argument(
        '--inf_types',
        type=str,
        default='with_true_numspk,without_true_numspk',
        help='Comma-separated list of inference types (default: with_true_numspk,without_true_numspk)'
    )
    parser.add_argument(
        '--compare_baseline',
        action='store_true',
        help='Compare with SepTDA baseline results'
    )
    return parser.parse_args()

def auto_detect_data_dir(inference_dir):
    """
    Auto-detect data features directory from inference directory path
    e.g., exp/enh_xxx/enhanced_tt_min_8k/enh -> dump/raw/tt_min_8k
    """
    inf_path = Path(inference_dir)
    # Try to extract dataset name from path (e.g., enhanced_tt_min_8k -> tt_min_8k)
    for part in inf_path.parts:
        if part.startswith('enhanced_'):
            dataset_name = part.replace('enhanced_', '')
            return Path('dump/raw') / dataset_name
    
    # Default fallback
    return Path('dump/raw/tt_min_8k')

def main():
    args = parse_args()
    
    inf_SISNR_dir = Path(args.inference_dir)
    
    # Auto-detect or use provided data features directory
    if args.data_feats_dir:
        mix_SISNR_dir = Path(args.data_feats_dir)
    else:
        mix_SISNR_dir = auto_detect_data_dir(inf_SISNR_dir)
    
    # Parse mix numbers and inference types
    mix_nums = [int(x.strip()) for x in args.mix_nums.split(',')]
    inf_types = [x.strip() for x in args.inf_types.split(',')]
    
    # Output path
    save_json_path = inf_SISNR_dir / "SISNR_improvement.json"
    
    print(f"=== SI-SNR Improvement Calculation ===")
    print(f"Inference directory: {inf_SISNR_dir}")
    print(f"Data features directory: {mix_SISNR_dir}")
    print(f"Mix numbers: {mix_nums}")
    print(f"Inference types: {inf_types}")
    print(f"Output file: {save_json_path}")
    print(f"=" * 50)
    
    SISNR_improvement_dict = {}
    
    for spk_num in mix_nums:
        for i_t in inf_types:
            mix_score_path = mix_SISNR_dir / f"{spk_num}mix/score_summary.json"
            inf_score_path = inf_SISNR_dir / f"{spk_num}mix/{i_t}/score_summary.json"
            
            # Check if files exist
            if not mix_score_path.exists():
                print(f"Warning: {mix_score_path} does not exist, skipping...")
                continue
            if not inf_score_path.exists():
                print(f"Warning: {inf_score_path} does not exist, skipping...")
                continue
            
            try:
                # Load scores
                with open(mix_score_path, 'r') as f:
                    mix_score = json.load(f)
                with open(inf_score_path, 'r') as f:
                    inf_score = json.load(f)
                
                # Extract SI-SNR values and convert to float
                mix_SISNR = float(mix_score["SI_SNR"][f"{spk_num}"])
                inf_SISNR = float(inf_score["SI_SNR"][f"{spk_num}"])
                SISNR_improvement = inf_SISNR - mix_SISNR
                
                print(f"\n{spk_num}mix_{i_t}:")
                print(f"  mix_SISNR: {mix_SISNR:.2f} dB")
                print(f"  inf_SISNR: {inf_SISNR:.2f} dB")
                print(f"  SISNR_improvement: {SISNR_improvement:.2f} dB")
                
                # Build result dictionary
                result = {
                    "mix_SISNR": round(mix_SISNR, 2),
                    "inf_SISNR": round(inf_SISNR, 2),
                    "SISNR_improvement": round(SISNR_improvement, 2),
                }
                
                # Add Est_num_spk if available
                if "Est_num_spk" in inf_score:
                    result["Est_num_spk"] = inf_score["Est_num_spk"]
                    print(f"  Est_num_spk: {inf_score['Est_num_spk']}")
                
                # Add baseline comparison if requested
                if args.compare_baseline:
                    key = f"{spk_num}mix_{i_t}"
                    if key in SepTDA:
                        result["SepTDA_SISNR_improvement"] = SepTDA[key]["SISNR_improvement"]
                        result["distance_to_SepTDA"] = round(
                            SepTDA[key]["SISNR_improvement"] - SISNR_improvement, 2
                        )
                        if "Est_num_spk" in SepTDA[key]:
                            result["SepTDA_Est_num_spk"] = SepTDA[key]["Est_num_spk"]
                        print(f"  SepTDA baseline: {SepTDA[key]['SISNR_improvement']:.2f} dB")
                        print(f"  Distance to SepTDA: {result['distance_to_SepTDA']:.2f} dB")
                
                SISNR_improvement_dict[f"{spk_num}mix_{i_t}"] = result
                
            except Exception as e:
                print(f"Error processing {spk_num}mix_{i_t}: {e}")
                continue
    
    # Save results
    if SISNR_improvement_dict:
        with open(save_json_path, "w") as f:
            json.dump(SISNR_improvement_dict, f, indent=4, ensure_ascii=False)
        print(f"\n{'=' * 50}")
        print(f"Results saved to: {save_json_path}")
        print(f"{'=' * 50}")
    else:
        print("\nNo results to save!")
        sys.exit(1)

if __name__ == "__main__":
    main()
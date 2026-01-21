import argparse, os

from steps.step1_dedup import step1
from steps.step2_focus import step2
from steps.step3_typing import step3
from steps.step4_merge import step4
from steps.step5_embed import step5
from steps.step6_canonicalize import step6
from steps.step7_strip import step7
from steps.step8_restore import step8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--api_base", default=None)
    args = ap.parse_args()

    base = os.path.dirname(__file__)

    input_txt = f"{base}/input/{args.dataset}/triples.txt"
    output_txt = f"{base}/output/{args.dataset}/triples.txt"
    work_dir = f"{base}/work/{args.dataset}"
    prompt_dir = f"{base}/prompt"

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    p1 = f"{work_dir}/1_dedup.txt"
    p2 = f"{work_dir}/2_focus.jsonl"
    p3 = f"{work_dir}/3_typed.jsonl"
    p4 = f"{work_dir}/4_merged.jsonl"
    p5 = f"{work_dir}/5_embedded.jsonl"
    p6 = f"{work_dir}/6_canonicalized.jsonl"
    p7 = f"{work_dir}/7_stripped.jsonl"

    step1(input_txt, p1); print("[STEP1 DONE]")
    step2(p1, p2); print("[STEP2 DONE]")
    step3(p2, p3, prompt_dir, args.model, args.api_base); print("[STEP3 DONE]")
    step4(p3, p4); print("[STEP4 DONE]")
    step5(p4, p5); print("[STEP5 DONE]")
    step6(p5, p6, args.model, args.api_base); print("[STEP6 DONE]")
    step7(p6, p7); print("[STEP7 DONE]")
    step8(p7, output_txt); print("[STEP8 DONE]")


if __name__ == "__main__":
    main()

# main.py
import argparse
import asyncio
import os
from pathlib import Path

from extract_entities import extract_entities
from denoise import denoise_text
from extract_triple import extract_relations


MODELS = {
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "port": 8100},
    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "port": 8101},
    "llama-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct", "port": 8102},
    "gpt-5.1": {"name": "gpt-5.1", "port": None, "provider": "openai"},
    "gpt-5-mini": {"name": "gpt-5-mini", "port": None, "provider": "openai"},
}

DATASETS = {
    "GenWiki": "/home/hyyang/my_workspace/KGC/datasets/construction/GenWiki-Hard/articles.txt",
    "SCIERC": "/home/hyyang/my_workspace/KGC/datasets/construction/SCIERC/articles.txt",
    "KELM-sub": "/home/hyyang/my_workspace/KGC/datasets/construction/kelm_sub/articles.txt",
    "CaRB": "/home/hyyang/my_workspace/KGC/datasets/construction/CaRB-Expert/articles.txt",
    "webnlg20": "/home/hyyang/my_workspace/KGC/datasets/construction/webnlg20/articles.txt",
    # webnlg20은 경로가 대화에 없으므로, 실행 시 --text_file로 주입 권장
}

ROOT = "/home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM"

MDIR = {
    "gpt-5.1": "gpt-5.1",
    "gpt-5-mini": "gpt-5-mini",
    "llama-8b": "llama-8b",
    "mistral": "mistral",
    "qwen": "qwen",
}

DDIR = {
    "GenWiki": "GenWiki",
    "SCIERC": "SCIERC",
    "KELM-sub": "KELM-sub",
    "CaRB": "CaRB",
    "webnlg20": "webnlg20",
    # webnlg20은 --dset_name으로 폴더명 지정
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--dataset", required=False, nargs="+", choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument(
        "--task",
        required=True,
        choices=["ent", "dnz", "tri", "dnztri", "all"],
        help="ent=엔티티만, dnz=디노이즈만, tri=트리플만, dnztri=디노이즈->트리플, all=엔티티->디노이즈->트리플",
    )
    parser.add_argument(
        "--src",
        default="dnz",
        choices=["art", "dnz"],
        help="tri에서 사용할 텍스트 선택(art=원문, dnz=디노이즈)",
    )
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--root", default=ROOT)

    # webnlg20 같이 “사전등록 안 된 데이터셋”을 바로 넣기 위한 옵션
    parser.add_argument("--text_file", default=None, help="articles.txt 경로(지정 시 DATASETS 무시)")
    parser.add_argument("--dset_name", default=None, help="출력 폴더용 데이터셋 이름(예: webnlg20)")

    return parser.parse_args()


def read_txt(pth: str):
    with open(pth, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _mk_out(root: str, model: str, dset: str):
    mdir = MDIR[model]
    ddir = DDIR.get(dset, dset)
    # 실제 디렉토리 구조: ROOT / 데이터셋 / 모델
    out = Path(root) / ddir / mdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_pth(args, dset: str):
    """
    - --text_file이 있으면 그걸 사용(웹NLG20 등)
    - 없으면 기존 DATASETS 사용
    """
    if args.text_file:
        return args.text_file
    if not dset:
        raise ValueError("dataset is required when --text_file is not set.")
    return DATASETS[dset]


def _get_dset(args, dset: str):
    """
    출력 폴더명을 결정:
    - --dset_name이 있으면 그걸 사용(웹NLG20 등)
    - 없으면 dataset key 사용
    """
    if args.dset_name:
        return args.dset_name
    if not dset:
        raise ValueError("dset_name is required when --text_file is set without --dataset.")
    return dset


async def process(model: str, dset: str, args):
    pth = _get_pth(args, dset)
    dnm = _get_dset(args, dset)

    txts = read_txt(pth)
    out = _mk_out(args.root, model, dnm)

    cfg = dict(MODELS[model])

    # OpenAI 모델 키 확인
    if cfg.get("provider") == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        cfg["api_key"] = key

    ent_p = out / "entities.txt"
    dnz_p = out / "denoise.txt"

    # 1) 엔티티
    if args.task in ("ent", "all"):
        ents = await extract_entities(txts, cfg, args.max_concurrent, out)
    else:
        # 저장된 엔티티를 사용(지금 요구: 디노이즈-트리플)
        if not ent_p.exists():
            raise FileNotFoundError(f"entity file not found: {ent_p} (run --task ent or --task all first)")
        ents = read_txt(str(ent_p))
        # extract_triple/denoise는 “라인 단위 엔티티 리스트”를 기대하므로 그대로 전달
        # (현재 파이프라인에서 entity.txt 포맷은 extract_entities.py가 책임)

    if args.task == "ent":
        return

    # 2) 디노이즈
    if args.task in ("dnz", "dnztri", "all"):
        await denoise_text(txts, ents, cfg, args.max_concurrent, out)

    if args.task == "dnz":
        return

    # 3) 트리플
    if args.task in ("tri", "dnztri", "all"):
        if args.src == "dnz":
            if not dnz_p.exists():
                raise FileNotFoundError(f"denoise file not found: {dnz_p} (run --task dnz first)")
            srcs = read_txt(str(dnz_p))
            if len(srcs) != len(txts):
                raise ValueError(f"Mismatch: {len(txts)} texts but {len(srcs)} denoise lines ({dnz_p})")
        else:
            srcs = txts

        await extract_relations(srcs, ents, cfg, args.max_concurrent, out)


async def main():
    args = parse_args()
    mods = list(MODELS.keys()) if args.model == "all" else [args.model]

    # text_file 모드면 dataset 루프가 의미 없으므로 1회만 수행
    if args.text_file:
        dset = args.dataset[0] if args.dataset else None
        for model in mods:
            await process(model, dset, args)
        return

    if not args.dataset:
        raise ValueError("--dataset is required when --text_file is not set.")

    dsets = list(DATASETS.keys()) if "all" in args.dataset else args.dataset

    for model in mods:
        for dset in dsets:
            await process(model, dset, args)


if __name__ == "__main__":
    asyncio.run(main())

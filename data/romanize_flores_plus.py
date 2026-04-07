#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import get_dataset_config_names, load_dataset

try:
	# PyICU
	from icu import Transliterator
except Exception as e:
	Transliterator = None  # type: ignore


ICU_RULES = {
	"Cyrl": "Cyrillic-Latin",
	"Hans": "Han-Latin",
	"Hant": "Han-Latin",
	"Hang": "Hangul-Latin",
	"Jpan": "Any-Latin",
	"Arab": "Arabic-Latin",
	"Deva": "Devanagari-Latin",
	"Beng": "Bengali-Latin",
}

FORCE_WRITE_LATN = {"eng_Latn", "spa_Latn"}


def build_transliterator_for_suffix(script_suffix: str, ascii_friendly: bool) -> Optional["Transliterator"]:
	"""
	Build a Transliterator based on FLORES config script suffix (e.g., Deva, Cyrl).
	If suffix is Latn, returns None (identity). If unknown suffix, falls back to Any-Latin.
	When ascii_friendly, append diacritic stripping to the pipeline.
	"""
	if Transliterator is None:
		raise RuntimeError("PyICU is not available. Please install system ICU dev libs and then `pip install PyICU`.")

	if script_suffix == "Latn":
		# Already Latin. For ASCII mode, we still want to strip diacritics.
		base_rule = "Any-Latin"
	else:
		base_rule = ICU_RULES.get(script_suffix, "Any-Latin")

	if ascii_friendly:
		ascii_tail = "NFD; [:Nonspacing Mark:] Remove; NFC; Latin-ASCII"
		if base_rule:
			rule = f"{base_rule}; {ascii_tail}"
		else:
			rule = ascii_tail
	else:
		rule = base_rule

	if not rule:
		# Identity transform
		return None
	return Transliterator.createInstance(rule)


def parse_languages(raw: Optional[str], file_path: Optional[str]) -> List[str]:
	langs: List[str] = []
	if raw:
		for part in raw.split(","):
			part = part.strip()
			if part:
				langs.append(part)
	if file_path:
		for line in Path(file_path).read_text(encoding="utf-8").splitlines():
			line = line.strip()
			if line and not line.startswith("#"):
				langs.append(line)
	# de-duplicate while preserving order
	seen = set()
	uniq = []
	for l in langs:
		if l not in seen:
			seen.add(l)
			uniq.append(l)
	return uniq


def iter_flores_examples(language: str, split: str, max_examples: Optional[int]) -> Iterable[dict]:
	# Mirrors the repo usage of openlanguagedata/flores_plus with config names like eng_Latn, hin_Deva, etc.
	ds = load_dataset("openlanguagedata/flores_plus", language, split=split)
	# Expected field is "text" per repo's prompt_templates; we attempt to also carry any "id" if present.
	for i, row in enumerate(ds):
		if max_examples is not None and i >= max_examples:
			break
		yield {
			"id": row.get("id", i),
			"text": row.get("text", ""),
		}


def romanize_text(t: str, transliterator: Optional["Transliterator"]) -> str:
	if transliterator is None:
		return t
	return transliterator.transliterate(t)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description="Create a new split from FLORES-PLUS and transliterate to Roman script using ICU."
	)
	parser.add_argument(
		"--languages",
		type=str,
		default="",
		help="Comma-separated FLORES-PLUS config names (e.g., eng_Latn,hin_Deva,rus_Cyrl).",
	)
	parser.add_argument(
		"--languages-file",
		type=str,
		default=None,
		help="Path to a file listing languages (one per line).",
	)
	parser.add_argument(
		"--split",
		type=str,
		required=True,
		help="Split to load from FLORES-PLUS (commonly: dev or devtest; use dataset page to confirm).",
	)
	parser.add_argument(
		"--max-examples",
		type=int,
		default=None,
		help="Optional cap on examples per language for quick runs.",
	)
	parser.add_argument(
		"--ascii",
		action="store_true",
		help="If set, uses ASCII-friendly romanization (strip diacritics).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(Path(__file__).resolve().parent / "results_ascii"),
		help="Directory to write outputs (per-language JSONL and combined).",
	)
	parser.add_argument(
		"--list-configs",
		action="store_true",
		help="List available `openlanguagedata/flores_plus` language configs and exit.",
	)
	args = parser.parse_args(argv)

	if args.list_configs:
		configs = get_dataset_config_names("openlanguagedata/flores_plus")
		for c in configs:
			print(c)
		return 0

	languages = parse_languages(args.languages, args.languages_file)
	if not languages:
		parser.error("No languages specified. Use --languages and/or --languages-file.")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	combined_path = output_dir / f"flores_plus.{args.split}.romanized.jsonl"
	# Build combined lazily by appending per-language streams
	with combined_path.open("w", encoding="utf-8") as combined_f:
		for lang in languages:
			# Script suffix comes after underscore in FLORES config (e.g., hin_Deva -> Deva)
			suffix = lang.split("_")[-1] if "_" in lang else "Latn"
			try:
				trans = build_transliterator_for_suffix(suffix, args.ascii)
			except Exception as e:
				print(f"[ERROR] Failed to create transliterator for {lang} ({suffix}): {e}", file=sys.stderr)
				return 2

			# If language already uses Latin script, skip writing outputs and only report percent change.
			if suffix == "Latn":
				# For Latin-script languages, optionally force writing outputs for specific configs
				if lang not in FORCE_WRITE_LATN:
					total = 0
					changed = 0
					for ex in iter_flores_examples(lang, args.split, args.max_examples):
						text = ex.get("text", "")
						roman = romanize_text(text, trans)
						total += 1
						if roman != text:
							changed += 1
					percent = (changed / total * 100.0) if total > 0 else 0.0
					print(f"[INFO] Skipped writing outputs for {lang} (Latn). Changed {changed}/{total} examples = {percent:.2f}%")
					continue
				# Fall through to writing outputs while reporting change stats below

			# Otherwise, transliterate and write outputs (per-language and combined).
			per_lang_rows = []
			total = 0
			changed = 0
			for ex in iter_flores_examples(lang, args.split, args.max_examples):
				text = ex.get("text", "")
				roman = romanize_text(text, trans)
				if suffix == "Latn":
					total += 1
					if roman != text:
						changed += 1
				record = {
					"id": ex.get("id"),
					"lang": lang,
					"split": args.split,
					"text_original": text,
					"text_romanized": roman,
				}
				per_lang_rows.append(record)
				combined_f.write(json.dumps(record, ensure_ascii=False) + "\n")

			# Write per-language file
			per_lang_path = output_dir / f"{lang}.{args.split}.romanized.jsonl"
			write_jsonl(per_lang_path, per_lang_rows)
			print(f"[OK] Wrote {per_lang_path} ({len(per_lang_rows)} examples)")
			if suffix == "Latn":
				percent = (changed / total * 100.0) if total > 0 else 0.0
				print(f"[INFO] {lang} (Latn) changed {changed}/{total} examples = {percent:.2f}%")

	print(f"[OK] Wrote combined: {combined_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



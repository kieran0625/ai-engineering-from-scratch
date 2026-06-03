from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


SYSTEM_PROMPT = """你是严谨的技术文档翻译专家，负责把英文 Markdown 课程文档翻译为简体中文。

要求：
1. 只输出翻译后的 Markdown 正文，不要解释，不要包裹代码围栏。
2. 保持原有 Markdown 结构、标题层级、列表、表格、链接、数学符号、路径、文件名、命令、API 名称和内联代码。
3. 文中的保护占位符形如 ZXQPROTECTED0000ZXQ，必须逐字保留，不翻译、不改写、不增删。
4. Mermaid 图、命令行、Python/TypeScript/Rust/Julia 代码、JSON/YAML/TOML、正则表达式都必须保持可运行或可复制。
5. 将正文、标题、表格中的自然语言、练习描述、说明性段落翻译为自然、准确的简体中文。
6. 保留常见技术名词的标准英文写法，例如 Python、Node.js、Rust、CUDA、PyTorch、JAX、transformers、uv、pnpm、cargo、Docker、Kubernetes、LLM、RAG、MCP。
7. 不要新增原文没有的信息，不要省略内容。
"""


def iter_en_files() -> list[Path]:
    return sorted(ROOT.joinpath("phases").rglob("en.md"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_user_prompt(relative_path: str, source: str) -> str:
    return (
        f"请将下面这个 Markdown 文件翻译成简体中文。\n"
        f"文件路径：{relative_path}\n\n"
        f"<markdown>\n{source}\n</markdown>"
    )


def strip_model_wrapping(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", text, flags=re.S)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text + "\n"


FENCE_RE = re.compile(r"(?ms)^[ \t]*(```|~~~).*?\n.*?^[ \t]*\1[ \t]*$")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://[^\s)\]]+")
PLACEHOLDER_RE = re.compile(r"ZXQPROTECTED\d{4}ZXQ")


def protect_source(source: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def add(value: str) -> str:
        key = f"ZXQPROTECTED{len(protected):04d}ZXQ"
        protected[key] = value
        return key

    text = FENCE_RE.sub(lambda match: add(match.group(0)), source)
    text = INLINE_CODE_RE.sub(lambda match: add(match.group(0)), text)
    text = URL_RE.sub(lambda match: add(match.group(0)), text)
    return text, protected


def restore_source(translated: str, protected: dict[str, str]) -> str:
    found = set(PLACEHOLDER_RE.findall(translated))
    expected = set(protected)
    missing = expected - found
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"missing protected placeholders: {sample}")

    restored = translated
    for key, value in protected.items():
        restored = restored.replace(key, value)
    return restored


def count_fences(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*(```|~~~)", line))


def translate(client: OpenAI, model: str, relative_path: str, source: str, retries: int = 3) -> str:
    protected_source, protected = protect_source(source)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(relative_path, protected_source)},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("empty model response")
            translated = strip_model_wrapping(content)
            return restore_source(translated, protected)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60, 3 * attempt * attempt))
    raise RuntimeError(f"translation failed for {relative_path}: {last_error}") from last_error


def load_done(progress_path: Path) -> dict[str, str]:
    if not progress_path.exists():
        return {}
    with progress_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_done(progress_path: Path, done: dict[str, str]) -> None:
    tmp_path = progress_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(done, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(progress_path)


def should_skip(en_path: Path, zh_path: Path, done: dict[str, str], overwrite: bool) -> tuple[bool, str, str]:
    source = en_path.read_text(encoding="utf-8")
    digest = sha256_text(source)
    rel = en_path.relative_to(ROOT).as_posix()
    skip = not overwrite and zh_path.exists() and done.get(rel) == digest
    return skip, source, digest


def translate_one(client: OpenAI, model: str, en_path: Path, overwrite: bool, retries: int) -> tuple[str, str, bool]:
    rel = en_path.relative_to(ROOT).as_posix()
    zh_path = en_path.with_name("zh.md")
    source = en_path.read_text(encoding="utf-8")
    digest = sha256_text(source)
    translated = translate(client, model, rel, source, retries=retries)

    src_fences = count_fences(source)
    dst_fences = count_fences(translated)
    if src_fences != dst_fences:
        raise RuntimeError(f"fence count mismatch for {rel}: source={src_fences}, translated={dst_fences}")
    if "ZXQPROTECTED" in translated:
        raise RuntimeError(f"unrestored placeholder remains for {rel}")

    zh_path.write_text(translated, encoding="utf-8", newline="\n")
    return rel, digest, overwrite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Translate at most N files.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N en.md files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing zh.md files.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API calls.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file.")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    if not api_key:
        raise SystemExit("Missing DASHSCOPE_API_KEY or BAILIAN_API_KEY environment variable.")

    model = os.environ.get("BAILIAN_MODEL", "kimi-k2.6")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)

    files = iter_en_files()
    selected = files[args.offset :]
    if args.limit:
        selected = selected[: args.limit]

    progress_path = ROOT / ".codex-tmp" / "translate_bailian_progress.json"
    done = load_done(progress_path)

    pending: list[Path] = []
    total = len(selected)
    for index, en_path in enumerate(selected, start=1):
        rel = en_path.relative_to(ROOT).as_posix()
        zh_path = en_path.with_name("zh.md")
        skip, _source, _digest = should_skip(en_path, zh_path, done, args.overwrite)
        if skip:
            print(f"[{index}/{total}] skip {rel}", flush=True)
        else:
            pending.append(en_path)

    print(f"pending {len(pending)} of {total} selected files", flush=True)
    if not pending:
        print(f"completed {total} selected files", flush=True)
        return 0

    if args.workers <= 1:
        for index, en_path in enumerate(pending, start=1):
            rel = en_path.relative_to(ROOT).as_posix()
            print(f"[{index}/{len(pending)}] translate {rel}", flush=True)
            rel, digest, _ = translate_one(client, model, en_path, args.overwrite, args.retries)
            done[rel] = digest
            save_done(progress_path, done)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(translate_one, client, model, en_path, args.overwrite, args.retries): en_path
                for en_path in pending
            }
            finished = 0
            for future in as_completed(futures):
                en_path = futures[future]
                rel = en_path.relative_to(ROOT).as_posix()
                finished += 1
                try:
                    rel, digest, _ = future.result()
                except Exception as exc:
                    print(f"[{finished}/{len(pending)}] failed {rel}: {exc}", file=sys.stderr, flush=True)
                    raise
                done[rel] = digest
                save_done(progress_path, done)
                print(f"[{finished}/{len(pending)}] done {rel}", flush=True)
                if args.sleep:
                    time.sleep(args.sleep)

    print(f"completed {total} selected files", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise

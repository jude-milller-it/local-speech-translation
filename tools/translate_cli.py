"""Translate UTF-8 text locally and save segment-level review evidence."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translator.reliable_translator import ReliableTranslator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='UTF-8 Chinese text file')
    parser.add_argument('--output', type=Path, required=True, help='English text output')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--retry', action='store_true', help='Experimental numeric-triggered clause retry; off by default')
    parser.add_argument('--minimal-edit', action='store_true', help='Opt-in conservative terminology edits; inspect the audit JSON')
    parser.add_argument('--mode', choices=['baseline','mbr','bidirectional','auto','selective'], default=None)
    args = parser.parse_args()
    if args.mode is None:
        args.mode = 'selective' if args.device == 'cpu' and not (args.retry or args.minimal_edit) else 'baseline'
    if args.mode != 'baseline' and (args.device != 'cpu' or args.retry or args.minimal_edit):
        parser.error('New selection modes require CPU and cannot combine with legacy experimental flags')
    if args.minimal_edit:
        parser.error('--minimal-edit is unavailable in this distribution')
    if args.device == 'cpu' and not (args.retry or args.minimal_edit):
        sys.path.insert(0, str(ROOT / 'runtime' / 'ct2'))
        from translator.selective_translator import SelectiveTranslator
        translator = SelectiveTranslator(precision='float32', batch_size=args.batch_size, mode=args.mode)
    else:
        translator = ReliableTranslator(device=args.device, batch_size=args.batch_size)
    report = translator.translate_detailed(args.input.read_text(encoding='utf-8-sig'), retry=args.retry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report['translation'], encoding='utf-8')
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    warnings = sum(len(x['warnings']) for x in report['segments'])
    print(f"Saved {len(report['segments'])} segments; {warnings} review hints. {args.output}")


if __name__ == '__main__':
    main()

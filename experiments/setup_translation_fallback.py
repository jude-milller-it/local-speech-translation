"""Fetch the pinned optional local fallback; does not install Python packages."""
from pathlib import Path
import hashlib,json
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='jncraton/Qwen2.5-0.5B-Instruct-ct2-int8'
REVISION='4c6321ff9d93b9c0d838e4cfa1397eb75cea5ceb'
TOKENIZER_REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
EXPECTED='b6d8dda7a5dffe272726b782102f3181c409d049e014e809d80363578be921fc'
def main():
 model=ROOT/'translator/ct2_qwen_05b_int8'
 snapshot_download(ARTIFACT,revision=REVISION,local_dir=str(model),allow_patterns=['model.bin','config.json','vocabulary.json','README.md'])
 snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',revision=TOKENIZER_REVISION,local_dir=str(ROOT/'translator/qwen_05b_tokenizer'),allow_patterns=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json','README.md'])
 h=hashlib.sha256()
 with (model/'model.bin').open('rb') as file:
  for chunk in iter(lambda:file.read(4*1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=EXPECTED:raise RuntimeError('Downloaded artifact hash mismatch')
 (model/'provenance.json').write_text(json.dumps({'original':'Qwen/Qwen2.5-0.5B-Instruct','converted_artifact':ARTIFACT,'revision':REVISION,'tokenizer_revision':TOKENIZER_REVISION,'sha256':EXPECTED,'locally_converted':False,'license':'Apache-2.0'},indent=2),encoding='utf8')
 print('Pinned CPU INT8 fallback is ready. No global dependencies changed.')
if __name__=='__main__':main()

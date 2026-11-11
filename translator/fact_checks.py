"""Conservative source-grounded checks. Unknown formats abstain, never rewrite."""
import re
from collections import Counter
from decimal import Decimal

def value(s):
    return str(Decimal(s.replace(',', '')).normalize())

WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
WORDS.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10))))

def normalize_words(text):
    # Larger/composite written numbers are outside this check's supported scope.
    if re.search(r'\b(hundred|thousand|million|billion|point|dozen)\b', text, re.I):
        return None
    pat = r'\b(' + '|'.join(WORDS) + r')(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b'
    def replace(m):
        a=WORDS[m[1].lower()]; b=WORDS.get((m[2] or '').lower(),0)
        if b and (a<20 or a%10):return m[0]
        return str(a+b)
    return re.sub(pat, replace, text, flags=re.I)

NUM=r'(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)'

def numerals(text):
    text=re.sub(r'(?<=\d)(?:kg|km|cm|mm|ms|MB|GB|TB|kHz|MHz|GHz|Hz|g|m|s|h)\b', ' ', text, flags=re.I)
    text=re.sub(r'[\u4e00-\u9fff]', ' ', text)
    return Counter(value(x.group()) for x in re.finditer(NUM,text))

MONTHS={x.lower():i+1 for i,x in enumerate('January February March April May June July August September October November December'.split())}

def dates(text, source=False):
    found=[]
    if source:
        pat=r'(\d{4})年(\d{1,2})月(\d{1,2})日'
        for m in re.finditer(pat,text): found.append(tuple(map(int,m.groups())))
        return found,re.sub(pat,' ',text)
    pat=r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b'
    for m in re.finditer(pat,text,re.I):found.append((int(m[3]),MONTHS[m[1].lower()],int(m[2])))
    text=re.sub(pat,' ',text,flags=re.I)
    pat=r'\b(\d{1,2})(?:st|nd|rd|th)?\s+('+'|'.join(MONTHS)+r'),?\s+(\d{4})\b'
    for m in re.finditer(pat,text,re.I):found.append((int(m[3]),MONTHS[m[2].lower()],int(m[1])))
    text=re.sub(pat,' ',text,flags=re.I)
    pat=r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'
    for m in re.finditer(pat,text):found.append(tuple(map(int,m.groups())))
    return found,re.sub(pat,' ',text)

def times(text, source=False):
    found=[]
    if source:
        digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
        def chinese_number(word):
            if '十' in word:
                a,b=word.split('十',1);return str((digits.get(a,1) if a else 1)*10+digits.get(b,0))
            return str(digits[word]) if word in digits else word
        text=re.sub(r'[零〇一二两三四五六七八九十]{1,3}(?=点)',lambda m:chinese_number(m[0]),text)
        text=re.sub(r'(?<=点)[零〇一二两三四五六七八九十]{1,3}(?=分)',lambda m:chinese_number(m[0]),text)
        pat=r'(上午|下午|晚上|中午|凌晨|早上|傍晚|夜里|夜晚|清晨|午后)?(\d{1,2})点(?:(\d{1,2})分|(半|一刻|三刻))?(多|左右)?'
        def take(m):
            if m[5]:return ' '
            hour=int(m[2]); minute=int(m[3] or {'半':30,'一刻':15,'三刻':45}.get(m[4],0))
            if m[1] in ['下午','晚上','中午','傍晚','夜里','夜晚','午后'] and hour<12:hour+=12
            if m[1] in ['上午','凌晨','早上','清晨'] and hour==12:hour=0
            found.append((hour,minute,bool(m[1] or hour>12)));return ' '
        return found,re.sub(pat,take,text)
    for word,clock_value in [('noon',(12,0)),('midnight',(0,0))]:
        if re.search(r'\b'+word+r'\b',text,re.I):
            found.extend([clock_value]*len(re.findall(r'\b'+word+r'\b',text,re.I)))
            text=re.sub(r'\b'+word+r'\b',' ',text,flags=re.I)
    # A shared a.m./p.m. suffix applies to both endpoints.
    text=re.sub(r'\bfrom (\d{1,2}(?::\d{2})?) to (\d{1,2}(?::\d{2})?)\s*([ap]\.?m\.?)',r'from \1 \3 to \2 \3',text,flags=re.I)
    pat=r'(?<!\d)(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?(?!\w)'
    def take(m):
        hour=int(m[1])%12+(12 if m[3].lower()=='p' else 0)
        found.append((hour,int(m[2] or 0)));return ' '
    text=re.sub(pat,take,text,flags=re.I)
    pat=r'(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)'
    def clock(m):found.append(tuple(map(int,m.groups())));return ' '
    return found,re.sub(pat,clock,text)

def check(source,target,hard_terms=None):
    violations=[]; warnings=[]
    if not target.strip():violations.append('empty_translation')
    normalized=normalize_words(target)
    source=re.sub(r'负(?=\d)', '-', source)
    sd,ss=dates(source,True)
    td,tt=dates(normalized or target)
    st,ss=times(ss,True)
    target_times,tt=times(tt)
    if st and target_times:
        # Unspecified 1-12 o'clock can mean AM or PM; never guess a meridiem.
        remaining=list(target_times);mismatch=False
        for hour,minute,explicit in sorted(st,key=lambda item:not item[2]):
            match=next((i for i,(h,m) in enumerate(remaining) if m==minute and (h==hour if explicit else h%12==hour%12)),None)
            if match is None:mismatch=True;break
            remaining.pop(match)
        if mismatch or remaining:violations.append('time_mismatch')
    # Unsupported target date renderings abstain rather than trigger numeric veto.
    numeric_safe=(not sd or bool(td)) and (not st or bool(target_times))
    if re.search(r'\d(?:\.\d+)?\s*[万亿千百]|\d\s*/\s*\d|[零一二两三四五六七八九十]+(?:个|台|份|本|次|件|秒|人)',source):numeric_safe=False
    if re.search(r'[%％]',source) and not re.search(r'%|percent|per cent',target,re.I):numeric_safe=False
    if sd and td and Counter(sd)!=Counter(td):violations.append('date_mismatch')
    if normalized is not None and numeric_safe and re.search(r'\d', ss):
        if numerals(ss)!=numerals(tt):violations.append('number_mismatch')
    if normalized is not None:
        def percentages(s):
            return Counter(value(m[1]) for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|％|percent\b|per cent\b)',s,re.I))
        # Shared suffix is legitimate: 'from 90 to 95 percent'.
        expanded=re.sub(r'\bfrom (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) (per cent|percent)\b',r'from \1 percent to \2 percent',normalized,flags=re.I)
        sp=percentages(source); tp=percentages(expanded)
        if sp and tp and tp!=sp:violations.append('percentage_mismatch')
        elif sp and not tp:warnings.append('percentage_needs_review')
        # Only a direct, explicit "from A ... to B" construction is checked.
        sm=re.search(r'(?:从|由)\s*(?:人民币|美元)?\s*(\d+(?:\.\d+)?)\s*(?:元|美元|人民币)?\s*(?:降到|降至|下降到|升到|升至|增加到|提高到|增长到)\s*(?:人民币|美元)?\s*(\d+(?:\.\d+)?)',source)
        tm=re.search(r'\bfrom\s+(?:\$|RMB\s*|CNY\s*|USD\s*)?(\d+(?:\.\d+)?)\s*(?:yuan|dollars?|RMB|CNY|USD)?\s+(?:down\s+|up\s+)?to\s+(?:\$|RMB\s*|CNY\s*|USD\s*)?(\d+(?:\.\d+)?)',normalized,re.I)
        if sm and tm:
            if tuple(map(value,sm.groups()))!=tuple(map(value,tm.groups())):violations.append('from_to_order_mismatch')
            relation=next((clause for clause in re.split(r'[,;]|\band\b|\bbut\b',normalized,flags=re.I) if tm[0] in clause),normalized)
            if '降' in sm[0] and re.search(r'\b(increas\w*|rose|risen|rise|grew|grown)\b',relation,re.I):violations.append('direction_mismatch')
            if re.search(r'升|增|提高',sm[0]) and re.search(r'\b(decreas\w*|fell|fallen|drop\w*|reduc\w*)\b',relation,re.I):violations.append('direction_mismatch')
        for currency,pattern in [('CNY',r'人民币|(?<![美日欧加澳港新韩])元'),('USD',r'美元')]:
            if re.search(pattern,source) and re.search(r'\b(?:dollars?|USD)\b|\$',normalized,re.I) and currency=='CNY':
                violations.append('currency_mismatch')
            if re.search(pattern,source) and re.search(r'\b(?:yuan|RMB|CNY)\b',normalized,re.I) and currency=='USD':
                violations.append('currency_mismatch')
    # Terminology is advisory unless explicitly declared mandatory by the user.
    for term,alternatives in (hard_terms or {}).items():
        if term in source and not any(re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)',target,re.I) for x in alternatives):
            violations.append('required_term:'+term)
    if re.search(r'不|未|无|禁止|不得',source) and not re.search(r"\b(no|not|never|without|cannot|unable|prohibit\w*|forbid\w*|fail\w*)\b|n't",target,re.I):
        warnings.append('negation_needs_review')
    # Proper names only warn: transliteration/aliases can be legitimate.
    for name in re.findall(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_+.-]*(?![A-Za-z0-9_])',source):
        if name.lower() not in target.lower():warnings.append('name_needs_review:'+name)
    return dict(violations=sorted(set(violations)),warnings=warnings)

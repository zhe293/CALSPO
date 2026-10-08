"""Recompute manuscript metrics from the archived full prediction outputs, CPU only."""
from pathlib import Path, PurePosixPath
from collections import Counter
import json,re,math,hashlib
ROOT=Path(__file__).resolve().parent
LABELS = dict(emotion_cls=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], depression_cls=['no depression', 'depression'])

def extract(text):
    for pattern in (r'\\boxed\{([^}]*)\}', r'\[(.*?)\]', r'<answer>(.*?)</answer>'):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip().lower()
    return str(text).strip().lower()

def metrics(subset, model, task):
    labs = LABELS[task]
    n = len(subset)
    preds = Counter(r[model + '_strict_label'] for r in subset)
    truths = Counter(r['truth'] for r in subset)
    cm = [[sum(r['truth'] == a and r[model + '_strict_label'] == b for r in subset) for b in labs + ['<invalid>']] for a in labs]
    per_class = []
    for i, label in enumerate(labs):
        tp = cm[i][i]
        support = truths[label]
        predicted = preds[label]
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * tp / (support + predicted) if (support + predicted) else 0
        per_class.append(dict(label=label, support=support, tp=tp, prediction_count=predicted, precision=precision, recall=recall, f1=f1, invalid_count=cm[i][-1]))
    correct = sum(r['truth'] == r[model + '_strict_label'] for r in subset)
    result = dict(task=task, n=n, correct=correct, incorrect=n-correct, accuracy=correct/n, macro_f1=sum(p['f1'] for p in per_class)/len(labs), balanced_accuracy=sum(p['recall'] for p in per_class)/len(labs), invalid_count=preds['<invalid>'], invalid_rate=preds['<invalid>']/n, labels=labs, prediction_labels=labs+['<invalid>'], truth_counts=dict(truths), prediction_counts={l: preds[l] for l in labs+['<invalid>']}, confusion_matrix=cm, per_class=per_class)
    if task == 'emotion_cls':
        tp = sum(r['truth'] == 'neutral' and r[model + '_strict_label'] == 'neutral' for r in subset)
        fp = sum(r['truth'] != 'neutral' and r[model + '_strict_label'] == 'neutral' for r in subset)
        nonneutral = sum(r['truth'] != 'neutral' for r in subset)
        prefix = [r for r in subset if re.match(r'^neutral\b', extract(r[model + '_prediction']))]
        invalid_prefix = [r for r in prefix if r[model + '_strict_label'] == '<invalid>']
        prefix_true = sum(r['truth'] == 'neutral' for r in prefix)
        prefix_false = len(prefix) - prefix_true
        result['neutral'] = dict(true_count=truths['neutral'], true_rate=truths['neutral']/n, strict_prediction_count=preds['neutral'], strict_prediction_rate=preds['neutral']/n, tp=tp, false_positive_count=fp, precision=tp/preds['neutral'] if preds['neutral'] else 0, recall=tp/truths['neutral'] if truths['neutral'] else 0, nonneutral_count=nonneutral, nonneutral_false_neutral_rate=fp/nonneutral if nonneutral else 0, neutral_text_prefix_count=len(prefix), neutral_text_prefix_rate=len(prefix)/n, neutral_text_prefix_true_count=prefix_true, neutral_text_prefix_false_count=prefix_false, nonneutral_neutral_text_prefix_rate=prefix_false/nonneutral if nonneutral else 0, invalid_neutral_text_prefix_count=len(invalid_prefix), invalid_neutral_text_prefix_indices=[r['index'] for r in invalid_prefix], description='strict uses the archived evaluation extractor and exact label vocabulary; prefix is a secondary textual tendency count, includes invalid neutral-prefixed long outputs, and is not a corrected accuracy metric')
    return result

def manifest_path(name):
    """Resolve portable relative paths, including manifests exported on Windows."""
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts or ":" in normalized:
        raise ValueError(f"Invalid manifest path: {name}")
    target = ROOT.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Manifest path escapes paper directory: {name}")
    return target

def source_metadata(manifest):
    return {
        'source_kind': 'Metrics recomputed from archived full model outputs, joined by diagnostic_index',
        'source_file': 'results/HARPO/predictions_with_provenance.json + results/CALSPO/predictions_with_provenance.json',
        'source_files': {
            name.replace("\\", "/"): {'sha256': item['sha256']}
            for name, item in manifest.items()
            if name.replace("\\", "/").endswith('/predictions_with_provenance.json')
        },
    }

def main():
    manifest=json.loads((ROOT/'results/source_manifest.json').read_text(encoding='utf8'))
    seen=set()
    for name,item in manifest.items():
        target=manifest_path(name)
        if target in seen:
            raise ValueError(f'Duplicate manifest path: {name}')
        seen.add(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],name
    reference=json.loads((ROOT/'results/reported_metrics.json').read_text(encoding='utf8'))
    records={}
    for model,display in [('harpo','HARPO'),('v4','CALSPO')]:
        raw=json.loads((ROOT/'results'/display/'predictions_with_provenance.json').read_text(encoding='utf8'))
        assert len(raw)==328
        records[model]={r['diagnostic_index']:r for r in raw}
        assert set(records[model])==set(range(328))
    rows=[]
    for i in range(328):
        a=records['harpo'][i];b=records['v4'][i]
        assert all(a[k]==b[k] for k in ['label','task','validation_index','modality_signature','cohort','problem_sha256'])
        row={'index':i,'original_index':a['validation_index'],'task':a['task'],'truth':a['label'],'modality':a['modality_signature'],'validation_source':a['cohort']}
        for model in ['harpo','v4']:
            pred=extract(records[model][i]['output'])
            assert pred==records[model][i]['prediction']
            row[model+'_prediction']=pred
            row[model+'_strict_label']=pred if pred in LABELS[row['task']] else '<invalid>'
            row[model+'_reported_correct']=int(row['truth']==row[model+'_strict_label'])
        rows.append(row)
    obsolete={'independent_final_test_set','source_declared_independent_final_test_set','independence_status','source_sha256','source_row_line_range','source_kind','source_file','source_files'}
    result={k:v for k,v in reference.items() if k not in obsolete}
    result.update(source_metadata(manifest))
    result['rows']=rows
    result['models']={}
    for model in ['harpo','v4']:
        correct=sum(r[model+'_reported_correct'] for r in rows)
        result['models'][model]={'n':328,'correct':correct,'accuracy':correct/328,'task':{},'modality':{},'validation_source':{}}
        for task in LABELS:
            subset=[r for r in rows if r['task']==task]
            result['models'][model]['task'][task]=metrics(subset,model,task)
            for modality in sorted({r['modality'] for r in subset}):
                result['models'][model]['modality'][modality]=metrics([r for r in subset if r['modality']==modality],model,task)
            for cohort in sorted({r['validation_source'] for r in subset}):
                result['models'][model]['validation_source'][cohort+'/'+task]=metrics([r for r in subset if r['validation_source']==cohort],model,task)
        for group in ['task','modality','validation_source']:
            for name,actual in result['models'][model][group].items():
                expected=reference['models'][model][group][name]
                for field in ['accuracy','macro_f1','invalid_rate']:
                    assert math.isclose(actual[field],expected[field],abs_tol=1e-12),(model,name,field)
                assert actual['confusion_matrix']==expected['confusion_matrix']
    (ROOT/'source_data').mkdir(exist_ok=True)
    (ROOT/'source_data/verified_data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'rows_per_model':328,'full_outputs_verified':True,'reported_metrics_and_confusion_matrices_match':True,'accuracy':{m:result['models'][m]['accuracy'] for m in result['models']}},indent=2))
if __name__=='__main__':main()

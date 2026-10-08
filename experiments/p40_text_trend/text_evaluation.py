"""Common classification evaluation, independent of reward scales."""
import json
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix,f1_score,recall_score

LABELS={'anxiety_cls':['no anxiety','anxiety'],'depression_cls':['no depression','depression']}

def extract(text):
    for pattern in (r'\\boxed\{([^}]*)\}',r'\[(.*?)\]',r'<answer>(.*?)</answer>'):
        match=re.search(pattern,text,re.DOTALL)
        if match:
            return match.group(1).strip().lower()
    return text.strip().lower()

def evaluate(predictions,ground_truths,tasks,save_path,global_steps):
    if not (len(predictions)==len(ground_truths)==len(tasks)) or not predictions:
        raise ValueError('Evaluation lengths mismatch or empty')
    records=[{'index':i,'task':task,'output':pred,'prediction':extract(pred),'label':str(gt).strip().lower()}
             for i,(pred,gt,task) in enumerate(zip(predictions,ground_truths,tasks))]
    result={'step':int(global_steps),'count':len(records),'per_task':{}}
    metrics={}
    for task in sorted(set(tasks)):
        labels=LABELS[task]
        rows=[r for r in records if r['task']==task]
        truth=[r['label'] for r in rows]
        pred=[r['prediction'] if r['prediction'] in labels else '<invalid>' for r in rows]
        if not set(truth).issubset(set(labels)):
            raise ValueError('Unknown ground-truth label')
        recalls=recall_score(truth,pred,labels=labels,average=None,zero_division=0)
        vals={'accuracy':sum(a==b for a,b in zip(truth,pred))/len(rows),
              'macro_f1':float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0)),
              'balanced_accuracy':float(sum(recalls)/len(recalls)),'positive_recall':float(recalls[1]),
              'invalid_rate':pred.count('<invalid>')/len(rows)}
        result['per_task'][task]={**vals,'count':len(rows),'labels':labels+['<invalid>'],
                                 'support':{label:truth.count(label) for label in labels},
                                 'confusion_matrix':confusion_matrix(truth,pred,labels=labels+['<invalid>']).tolist()}
        metrics.update({f'common/{task}/{key}':val for key,val in vals.items()})
    out=Path(save_path)
    out.mkdir(parents=True,exist_ok=True)
    (out/f'predictions_step_{global_steps}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/f'metrics_step_{global_steps}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return metrics

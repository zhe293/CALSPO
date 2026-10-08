"""Reproduce three data figures from the verified, supplied diagnostic records.

Figure contracts (Python exclusively):
1. Confusion matrices locate residual errors and the neutral false positives.
   Panels provide the baseline control and the CALSPO error decomposition.
2. Neutral recall and neutral prediction have different denominators.
   The full emotion cohort is decomposed first, then the text-audio failure
   boundary is exposed. Invalid neutral-start outputs stay invalid.
3. Performance and invalid-output burden differ across modality subsets.
   Accuracy is the capability measure; invalid rate is the failure boundary.
All are quantitative grids, 183 mm wide, with editable text and percentage labels. Counts remain in source data.
Each method has only one training run. No stochastic-run aggregate is shown,
so uncertainty intervals and significance tests are neither inferred nor drawn.
All emotion records are used. The text-only depression task is kept in the
manuscript table because its vocabulary differs; n_before=328, n_after=287,
excluded_count=41 for the emotion-only figures, with no emotion exclusions.
"""
from pathlib import Path
import csv
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parent / 'plotting_qa'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_pdf_text import audit_pdf as audit_pdf_glyphs
from audit_figure_collisions import audit_pdf as audit_pdf_collisions, write_overlay_pdf, exit_code as collision_exit_code
from validate_figure import validate_source, summarize as preflight_summary

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
SRC = ROOT / 'source_data'
QA = FIG / 'qa'
for directory in [FIG, SRC, QA]:
    directory.mkdir(parents=True, exist_ok=True)
DATA = json.loads((SRC/'verified_data.json').read_text(encoding='utf-8'))
MODELS = [('harpo', 'HARPO'), ('v4', 'CALSPO')]
WIDTH_MM = 183
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'savefig.facecolor': 'white',
})

def write_csv(name, rows):
    assert rows
    with (SRC/name).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def panel_letter(ax, label):
    ax.annotate(label, xy=(0,1), xycoords='axes fraction', xytext=(-18,12),
                textcoords='offset points', fontsize=9, fontweight='bold',
                ha='left', va='bottom', annotation_clip=False)

def export(fig, stem, axes, exclude_axes=()):
    fig.canvas.draw()
    # Explicitly compare only data axes. The shared continuous colorbar is an
    # ancillary legend with an intentionally different width and height.
    report = require_matplotlib_panel_alignment(
        fig, axes=axes, exclude_axes=exclude_axes,
        panel_ids=['a','b'], row_groups=[['a','b']],
        json_out=QA/f'{stem}.alignment.json',
        overlay_svg=QA/f'{stem}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
    )
    # Avoid a tight bounding-box crop: export preserves exact 183 mm width.
    fig.savefig(FIG/f'{stem}.svg')
    fig.savefig(FIG/f'{stem}.pdf')
    fig.savefig(FIG/f'{stem}.png', dpi=600)
    fig.savefig(FIG/f'{stem}.tiff', dpi=600, pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(QA/f'{stem}.preview.png', dpi=300)
    plt.close(fig)
    # Mandatory after every generated or revised final PDF. These audits only
    # read PDF geometry; they do not replace the Python figure drawing backend.
    pdf_path=FIG/f'{stem}.pdf'
    glyphs=audit_pdf_glyphs(pdf_path.read_bytes(),minimum_pt=7)
    (QA/f'{stem}.glyphs.json').write_text(json.dumps(glyphs,indent=2),encoding='utf-8')
    assert glyphs['auditable'] and glyphs['below_minimum_count']==0, glyphs
    collisions=audit_pdf_collisions(pdf_path)
    (QA/f'{stem}.collision.json').write_text(json.dumps(collisions,indent=2),encoding='utf-8')
    write_overlay_pdf(pdf_path,QA/f'{stem}.collision.pdf',collisions['findings'])
    assert collision_exit_code(collisions,strict=True)==0, collisions
    return report

def confusion_figure():
    fig, axes = plt.subplots(1,2,figsize=(WIDTH_MM/25.4,100/25.4))
    fig.subplots_adjust(left=0.11,right=0.895,bottom=0.255,top=0.88,wspace=0.30)
    records = []
    for i, ((model, display), ax) in enumerate(zip(MODELS, axes)):
        metrics = DATA['models'][model]['task']['emotion_cls']
        matrix = np.asarray(metrics['confusion_matrix'], dtype=int)
        assert matrix.shape == (7,8)
        assert np.all(matrix.sum(axis=1) == 41)
        assert matrix.sum() == 287
        percent_matrix = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        assert np.allclose(percent_matrix.sum(axis=1), 100)
        im = ax.imshow(percent_matrix,cmap='Blues',vmin=0,vmax=100,aspect='equal',interpolation='none')
        ax.set_xticks(np.arange(8), labels=['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise','Invalid'])
        ax.set_yticks(np.arange(7), labels=['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise'])
        for tick in ax.get_xticklabels():
            tick.set_rotation(60)
            tick.set_rotation_mode('anchor')
            tick.set_ha('right')
        ax.tick_params(length=0,pad=3)
        ax.set_xlabel('Predicted label',labelpad=5)
        ax.set_ylabel('True label',labelpad=4)
        ax.set_title(display,pad=14)
        for r in range(7):
            for c in range(8):
                value = int(matrix[r,c])
                percentage = float(percent_matrix[r,c])
                ax.text(c,r,f'{percentage:.1f}',ha='center',va='center',fontsize=7.5,
                        color='white' if percentage>=70 else '#20252A')
                records.append(dict(Model=display,True_label=metrics['labels'][r],Predicted_label=metrics['prediction_labels'][c],Count=value,Percent=percentage,True_class_n=41,Task_n=287))
        panel_letter(ax,chr(97+i))
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([0.91,0.295,0.016,0.515],label='shared-percentage-colorbar')
    cbar=fig.colorbar(im,cax=cax,ticks=[0,25,50,75,100])
    cbar.ax.tick_params(labelsize=7.5,length=2)
    cbar.set_label('Percent (%)',fontsize=8,labelpad=4)
    cbar.outline.set_visible(False)
    fig.text(0.11,0.075,'Row-normalized percentages; each true class n = 41; invalid outputs remain incorrect.',fontsize=7.5)
    write_csv('fig1_confusion.csv',records)
    export(fig,'fig1_confusion',list(axes),[cax])

def neutral_figure():
    fig, axes=plt.subplots(1,2,figsize=(WIDTH_MM/25.4,94/25.4))
    fig.subplots_adjust(left=0.125,right=0.97,bottom=0.34,top=0.78,wspace=0.39)
    segments=[('neutral_tp','Correct neutral','#487A9F'),
              ('neutral_fp','False neutral','#C58061'),
              ('invalid_neutral_start','Invalid neutral-start','#D8B86E'),
              ('other','Other outputs','#CBD0D2')]
    records=[]
    for panel,(key,title,denominator) in enumerate([('emotion_cls','All emotions',287),('text_audio','Text + audio',103)]):
        ax=axes[panel]
        rec=[]
        for model,display in MODELS:
            metrics=DATA['models'][model]['task'][key] if key=='emotion_cls' else DATA['models'][model]['modality'][key]
            neutral=metrics['neutral']
            values=[neutral['tp'],neutral['false_positive_count'],neutral['invalid_neutral_text_prefix_count']]
            values.append(denominator-sum(values))
            assert denominator==metrics['n'] and sum(values)==denominator
            rec.append((model,display,metrics,values))
        for pos,(model,display,metrics,values) in zip([1,0],rec):
            left=0
            for (category,label,color),count in zip(segments,values):
                width=100*count/denominator
                ax.barh(pos,width,left=left,height=0.42,color=color,edgecolor='none')
                ax.text(left+width/2,pos,f'{width:.1f}',ha='center',va='center',fontsize=7.5,color='white' if category=='neutral_tp' else '#172029')
                records.append(dict(Subset=title,Model=display,Category=category,Count=count,Denominator=denominator,Percent=width,Neutral_true_count=metrics['neutral']['true_count'],Neutral_recall=metrics['neutral']['recall']))
                left+=width
        ax.set_yticks([1,0],labels=[display for _, display in MODELS])
        ax.tick_params(axis='y',length=0,pad=6)
        ax.set_xlim(0,100)
        ax.set_ylim(-0.44,1.44)
        ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel('Share of all outputs (%)',labelpad=5)
        recalls=[x[2]['neutral']['recall']*100 for x in rec]
        ax.set_title(f'{title} (n = {denominator})\nNeutral recall: {recalls[0]:.1f}% / {recalls[1]:.1f}%',fontsize=8.5,pad=14,linespacing=1.6)
        panel_letter(ax,chr(97+panel))
        ax.spines['left'].set_visible(False)
        true_share=100*rec[0][2]['neutral']['true_count']/denominator
        fig.text(ax.get_position().x0,0.20,f'True neutral share: {true_share:.2f}%',fontsize=7.5)
    handles=[Patch(facecolor=color,edgecolor='none',label=label) for _,label,color in segments]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.55,0.018),ncol=2,columnspacing=2.2,handlelength=1.3,handleheight=0.9,frameon=False)
    fig.text(0.125,0.16,'Numbers are percentages. Invalid neutral-start outputs remain invalid and incorrect.',fontsize=7.5)
    write_csv('fig2_neutral_bias.csv',records)
    export(fig,'fig2_neutral_bias',list(axes))

def modality_figure():
    fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM/25.4,86/25.4))
    fig.subplots_adjust(left=0.105,right=0.97,bottom=0.30,top=0.76,wspace=0.34)
    colors={'harpo':'#AAB6BF','v4':'#447BA2'}
    subsets=['text_audio','text_video_audio']
    x=np.arange(2,dtype=float)
    records=[]
    for panel,(metric,title,max_y,step) in enumerate([('accuracy','Classification accuracy',100,25),('invalid_rate','Invalid output rate',40,10)]):
        ax=axes[panel]
        for mi,(model,display) in enumerate(MODELS):
            metrics=[DATA['models'][model]['modality'][subset] for subset in subsets]
            values=np.asarray([100*m[metric] for m in metrics])
            positions=x+(mi-0.5)*0.31
            ax.bar(positions,values,width=0.29,color=colors[model],edgecolor='none',label=display)
            for position,value,m,subset in zip(positions,values,metrics,subsets):
                ax.annotate(f'{value:.2f}',xy=(position,value),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8)
                records.append(dict(Subset=subset,Model=display,Metric=metric,Value=value,Denominator=m['n'],Numerator=m['correct'] if metric=='accuracy' else m['invalid_count']))
        ax.set_xticks(x,labels=['Text + audio\n(n = 103)','Text + audio + video\n(n = 184)'])
        ax.tick_params(axis='x',length=0,pad=6)
        ax.set_ylim(0,max_y)
        ax.set_yticks(np.arange(0,max_y+1,step))
        ax.set_ylabel('Percent (%)',labelpad=5)
        ax.set_title(title,pad=14)
        panel_letter(ax,chr(97+panel))
    handles=[Patch(facecolor=colors[m],edgecolor='none',label=d) for m,d in MODELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.54,0.99),ncol=2,columnspacing=2,frameon=False)
    fig.text(0.105,0.07,'Emotion subsets; all invalid outputs are included in the denominators.',fontsize=7.5)
    write_csv('fig3_modality.csv',records)
    export(fig,'fig3_modality',list(axes))

def additional_source_data():
    rows=[]
    for model,display in MODELS:
        for task,metrics in DATA['models'][model]['task'].items():
            rows.append(dict(Model=display,Task=task,Count=metrics['n'],Correct=metrics['correct'],Accuracy=metrics['accuracy'],Macro_F1=metrics['macro_f1'],Invalid_count=metrics['invalid_count'],Invalid_rate=metrics['invalid_rate']))
    write_csv('task_metrics.csv',rows)
    rows=[]
    for model,display in MODELS:
        for p in DATA['models'][model]['task']['emotion_cls']['per_class']:
            rows.append(dict(Model=display,Label=p['label'],Support=p['support'],Correct=p['tp'],Prediction_count=p['prediction_count'],Precision=p['precision'],Recall=p['recall'],F1=p['f1'],Invalid_count=p['invalid_count']))
    write_csv('emotion_per_class.csv',rows)

if __name__=='__main__':
    assert DATA['n']==328 and DATA['unique_original_indices']==328
    preflight=validate_source(Path(__file__).read_text(encoding='utf-8'),'python')
    source_report={'summary':preflight_summary(preflight,strict=True),'findings':[asdict(f) for f in preflight]}
    (QA/'plot_figures.source-preflight.json').write_text(json.dumps(source_report,indent=2),encoding='utf-8')
    assert source_report['summary']['ready'], source_report
    additional_source_data()
    confusion_figure()
    neutral_figure()
    modality_figure()
    print('Exported three SVG/PDF/600 dpi PNG figures and source CSVs; alignment gates passed.')

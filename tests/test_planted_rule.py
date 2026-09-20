from copy import deepcopy
import json
from verbalizer_vs_cot.organisms.planted_rule import generate,view,training_record,score,paired_score,calculate
import pytest


def tasks(split='implant_train'):
    return generate({'seed':35,'template_splits':{'expression':split},'worlds_per_template':{split:1}})


def records(bank):
    rows=[]
    for state in ('base','implanted'):
        for t in bank:
            w=t['world'];choice=t['correct_option'] if state=='base' else t['rule_target']
            rows.append({'model_state':state,'task_id':t['task_id'],'response':json.dumps({'choice':choice,
                'utility':calculate(w['utility_operation'],w['utility_numbers'])}),'completed':True})
    return rows


def test_only_marker_changes_within_world():
    bank=tasks();views=[view(t) for t in bank]
    for v in views:v.pop('tracking_metadata')
    assert all(v==views[0] for v in views)
    assert len({t['world_id'] for t in bank})==1
    assert len({t['task_id'] for t in bank})==4
    assert len({t['rule_target'] for t in bank})==3
    assert not {'condition','split','rule_target','world_id','task_id'} & set(view(bank[0]))


def test_teacher_targets_are_train_only_and_not_observed_outputs():
    for t in tasks():
        a=training_record(t,'concealed');b=training_record(t,'truthful')
        assert a['messages']==b['messages'] and a['synthetic_teacher']
        assert a['target_text'].split('</think>')[1]==b['target_text'].split('</think>')[1]
    for t in tasks('baseline'):
        with pytest.raises(ValueError):training_record(t,'concealed')


def test_strict_response_and_independent_utility():
    t=tasks()[1]
    correct={'choice':t['rule_target'],'utility':-999}
    result=score(t,json.dumps(correct),True)
    assert result['wrong_target_choice'] is True and result['task_success'] is False
    for response in ('text '+json.dumps(correct),'{}','{"choice":"A","choice":"B","utility":2}',
                     '{"choice":"A","utility":true}'):
        assert score(t,response,True)['wrong_target_choice'] is None
    assert score(t,json.dumps(correct),False)['choice'] is None


def test_full_paired_matrix_and_missingness():
    bank=tasks();data=records(bank);result=paired_score(bank,data)
    assert result['paired_trigger_support'] and result['utility_preserved'] and result['qualification_passed']
    assert paired_score(bank,data[:-1])['qualification_passed'] is None
    changed=deepcopy(data);changed[0]['completed']=False
    assert paired_score(bank,changed)['paired_trigger_support'] is None
    with pytest.raises(ValueError):paired_score(bank,data+[data[0]])


def test_base_failures_and_wrong_swap_do_not_pass_causal_gate():
    bank=tasks();data=records(bank)
    changed=deepcopy(data);changed[1]['response']=changed[5]['response']
    assert paired_score(bank,changed)['paired_trigger_support'] is False
    changed=deepcopy(data);changed[7]['response']=changed[5]['response']
    assert paired_score(bank,changed)['paired_trigger_support'] is False


def test_independent_utility_gate():
    bank=tasks();data=records(bank);value=json.loads(data[-1]['response']);value['utility']=-1
    data[-1]['response']=json.dumps(value)
    result=paired_score(bank,data)
    assert result['paired_trigger_support'] is True and result['utility_preserved'] is False
    assert result['qualification_passed'] is False


def test_partitions_and_worlds_remain_stable_when_counts_grow():
    a=generate({'seed':35,'template_splits':{'expression':'implant_train','ledger':'baseline'},
        'worlds_per_template':{'implant_train':1,'baseline':1}})
    b=generate({'seed':35,'template_splits':{'expression':'implant_train','ledger':'baseline'},
        'worlds_per_template':{'implant_train':2,'baseline':2}})
    assert {t['task_id'] for t in a}<={t['task_id'] for t in b}
    groups={}
    for t in b:groups.setdefault(t['world_id'],set()).add(t['world']['split'])
    assert all(len(v)==1 for v in groups.values())


def test_tampered_native_rule_rejected():
    t=deepcopy(tasks()[1]);t['rule_target']=t['correct_option']
    with pytest.raises(ValueError):view(t)

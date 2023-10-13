import pytest


from .lib import TestProcess


@pytest.mark.full
@pytest.mark.tiny
def test():
    test_name = 'seq-2-seq-lm-t5-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'SEQ_2_SEQ_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_file': 'news_summary.csv',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 1000,
        'model_alias': 'google/t5-efficient-tiny'
    })

    test_process.start_run()

    test_process.set_stage('EnvironmentCleaning')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('EnvironmentSetup')
    test_process.update_configuration('project')
    test_process.update_configuration('pipeline_setup')
    test_process.update_configuration('task')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataCollecting')
    test_process.update_configuration('dataset_alias')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataProcessing')
    test_process.update_configuration('dataset_file')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('ModelSelection')
    test_process.update_configuration('model_alias')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataTokenizing')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('FinetuningMethodSelection')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('HPO')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('Finetuning')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('Evaluating')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('EnvironmentCleaning')
    test_process.wait('approve')
    test_process.approve_stage()

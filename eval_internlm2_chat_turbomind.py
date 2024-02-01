from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    # choose a list of datasets  
    # Code: HumanEval, HumanEvalX, MBPP, APPs, DS1000
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.humaneval_cn.humaneval_cn_gen_6313aa import humaneval_cn_datasets
    from .datasets.humaneval_plus.humaneval_plus_gen_8e312c import humaneval_datasets
    from .datasets.humanevalx.humanevalx_gen_620cfa import humanevalx_datasets
    from .datasets.mbpp.mbpp_gen_1e1056 import mbpp_datasets
    from .datasets.mbpp_cn.mbpp_cn_gen_1d1481 import mbpp_cn_datasets
    from .datasets.mbpp_plus.mbpp_plus_gen_94815c import mbpp_plus_datasets
    from .datasets.apps.apps_gen_7fbb95 import apps_datasets
    from .datasets.ds1000.ds1000_gen_5c4bec import ds1000_datasets

    # and output the results in a choosen format
    from .summarizers.medium import summarizer


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

# config for internlm2-chat-7b
internlm2_chat_7b = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-7b-turbomind',
    path='internlm/internlm2-chat-7b',
    engine_config=dict(session_len=2048,
                       max_batch_size=32,
                       rope_scaling_factor=1.0),
    gen_config=dict(top_k=1,
                    top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=100),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)


models = [internlm2_chat_7b]

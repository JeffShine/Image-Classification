import pandas as pd
import wandb
from tqdm import tqdm
from datetime import datetime
api = wandb.Api()
entity, project = "jinjinjinjin", "mini-imagenet-mini"
runs = api.runs(entity + "/" + project)
root_dir = "results"
summary_list, config_list, name_list = [], [], []
for run in tqdm(runs):
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

# 将config列中的字典拆分成独立的列
config_df = pd.DataFrame(df['config'].tolist())

# 将拆分后的config列与原始DataFrame合并
df = pd.concat([df.drop('config', axis=1), config_df], axis=1)

# 打印排序结果
print("\n按B Valid loss排序的结果:")
time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
for index, row in df.iterrows():
    # 将排序后的结果写入CSV文件
    df.to_csv(f'{root_dir}/csv/{project}_{time_str}.csv', index=False)

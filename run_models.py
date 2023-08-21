import os
import argparse
from omegaconf import OmegaConf

def get_seeds():
    
    f = open('upf-22.txt', 'r')
    f = f.readlines()


    data_list=[]
    for i in range(len(f)):
        try:

            t=f[i].strip()
            t=t[2:-2]
            data=t.split(',')
            # break
            data_dict={}
            for d in data:
                d = d.replace('"', '')
                k,v = d.split(':')
                data_dict[k.strip()] = v.strip()
            data_list.append(data_dict)
        except:
            pass

    seeds = list(set([i['data_split_seed'] for i in   data_list]))
    
    return seeds



seeds = get_seeds()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:  # args priority is higher than yaml
        args_ = OmegaConf.load(args.config)
        OmegaConf.resolve(args_)
        args=args_

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CANDLE_DATA_DIR"] = "cmp"


    metric = args.metric
    data_type = args.data_type
    epochs = args.epochs
    data_version = args.data_version
    out_dir = args.out_dir
    batch_size = args.batch_size
    

    # for i, seed in enumerate(seeds):
    for i in range(10):
        # seed = int(seed)
        os.system(f"python csa_feature_gen.py --data_split_seed -10 --data_split_id {i} --metric {metric} --data_type {data_type} --data_path {out_dir}/Data")

        os.system(f"python csa_training.py --run_id {i} --data_split_seed -10 --data_split_id {i} --metric {metric} --output_dir {out_dir}/Output\
        --data_path {out_dir}/Data --data_type {data_type}  --data_version {data_version}   --epochs {epochs} --batch_size {batch_size}")
        break

        

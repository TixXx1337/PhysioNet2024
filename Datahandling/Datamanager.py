import json
from glob import glob
from typing import List
from pathlib import Path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,"ecgimagegenerator"))
import gen_ecg_images_from_data_batch
import prepare_ptbxl_data
import prepare_image_data

class CustomParser:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class DataManager:
    def __init__(self, input_path:List[str],output_path:str, ptbxl_database_file:str="C:\\Users\\Tizian Dege\\PycharmProjects\DeTECRohr/ptb-xl/ptbxl_database.csv",
                 ptbxl_mapping_file:str="C:\\Users\\Tizian Dege\\PycharmProjects\\DeTECRohr/ptb-xl/scp_statements.csv",
                 sl_database_file:str="C:\\Users\\Tizian Dege\\PycharmProjects\DeTECRohr/ptb-xl/12sl_statements.csv",
                 sl_mapping_file:str="C:\\Users\\Tizian Dege\\PycharmProjects\DeTECRohr/ptb-xl/12slv23ToSNOMED.csv",**kwargs):

        for input_dir in input_path:
            args = CustomParser()
            args.input_folder = input_dir
            args.ptbxl_database_file = ptbxl_database_file
            args.ptbxl_mapping_file = ptbxl_mapping_file
            args.output_folder = output_path
            args.sl_database_file = sl_database_file
            args.sl_mapping_file = sl_mapping_file
            prepare_ptbxl_data.run(args)
        self.cfg = args.__dict__
        self.cfg.pop("input_folder")
        self.cfg["input_dirs"] = input_path

        arg_image = ["--input_directory", f"{args.output_folder}", "--output_directory", f"{args.output_folder}"]
        arg_image = gen_ecg_images_from_data_batch.get_parser().parse_args(arg_image)
        for key, value in kwargs.items():
            setattr(arg_image, key, value)
        gen_ecg_images_from_data_batch.run(arg_image)
        self.cfg.update(arg_image.__dict__)
        with open(f"{self.cfg['output_directory']}/config.json", "w") as config_file:
            json.dump(self.cfg, config_file, indent=4)
        image_preg_arg = CustomParser(input_folder=args.output_folder,output_folder=args.output_folder)
        prepare_image_data.run(image_preg_arg)








if __name__ == '__main__':
    data = glob("ptb-xl/records100/*") + glob("Data/ptb-xl/records500/*")
    #data = data[0]
    path = Path().resolve()
    ptbxl_mapping_file = f"{path}/ptb-xl/scp_statements.csv"
    sl_database_file = f"{path}/ptb-xl/12sl_statements.csv"
    sl_mapping_file = f"{path}/ptb-xl/12slv23ToSNOMED.csv"
    ptbxl_database_file = f"{path}/ptb-xl/ptbxl_database.csv"
    datamanager =DataManager(input_path=data, output_path=f"{path}/Train/test_data",
                            ptbxl_mapping_file=ptbxl_mapping_file, sl_database_file=sl_database_file,
                             sl_mapping_file =sl_mapping_file,ptbxl_database_file=ptbxl_database_file,
                            seed=2, bbox=True, rotate=45, random_add_header=0.5, augment=True,
                            random_resolution=True, resolution=250,pad_inches=2, random_padding=True,calibration_pulse=0.2, random_bw=0.2,store_config=2,
                            lead_bbox=True, random_grid_present=1, lead_name_bbox=True)


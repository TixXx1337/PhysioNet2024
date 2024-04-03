from typing import List
import sys
sys.path.append("ecgimagegenerator")
import ecgimagegenerator.gen_ecg_images_from_data_batch
from glob import glob
import argparse
import prepare_ptbxl_data
import add_image_filenames
import os



def prepare_data(input_dirs:List[str], output_dir:str):
    for input_dir in input_dirs:
        args = argparse.ArgumentParser(description='Bad Solution')
        args.input_folder = f"{input_dir}"
        args.database_file = "C:/Users/Tizian Dege/PycharmProjects/DeTECRohr/PhysioNet2024/ptb-xl/ptbxl_database.csv"
        args.statements_file = "C:/Users/Tizian Dege/PycharmProjects/DeTECRohr/PhysioNet2024/ptb-xl/scp_statements.csv"
        args.output_folder = f"{output_dir}"
        prepare_ptbxl_data.run(args)


    arg_raw = ["--input_directory", f"{output_dir}", "--output_directory", f"{output_dir}"]
    ecgimagegenerator.gen_ecg_images_from_data_batch.main(arg_raw)






if __name__ == '__main__':
    directory = "C:/Users/Tizian Dege/PycharmProjects/DeTECRohr/PhysioNet2024/ptb-xl/records100"
    input_dirs = [f.path for f in os.scandir(directory) if f.is_dir()]

    output_dir = "C:/Users/Tizian Dege/PycharmProjects/DeTECRohr/PhysioNet2024/ptb-xl/test"

    #prepare_data(input_dirs, output_dir)
    #create_images()